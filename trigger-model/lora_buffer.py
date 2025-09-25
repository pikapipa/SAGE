import torch 
import random
import json
import os
import yaml
from typing import List, Dict, Any
from torch.utils.data import DataLoader, TensorDataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model
from utils import load_config, save_json, to_device, get_device_info, load_json
from torch.utils.data import TensorDataset, DataLoader
import hashlib
import time

cfg = load_config("./config.yaml")

class LoraOptimizer:
    PARAM_SPACE = cfg.get("lora_param_space", {
        "r": [4, 8],   
        "lora_alpha": [8, 16],  
        "lr": [1e-4, 1e-5],
        "dropout": [0.05],
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "num_epochs": [5],
        "batch_size": [1]
    })

    FIXED_PARAMS = {
        "alpha_ratio": 2
    }

    lora_buffer: List[Dict[str, Any]] = []

    def load_anomaly_clusters(filepath=None):
        if filepath is None:
            buffer_config = cfg.get("buffer", {})
            filepath = buffer_config.get("anomaly_cluster_path", "/path/to/your/trigger/data/clusters.json")

        try:
            clusters = load_json(filepath)
        except Exception as e:
            print(f"loading 失败: {str(e)}")
            return []

    def sample_initial_parameters(n_samples: int = None) -> List[Dict[str, Any]]:
        if n_samples is None:
            lora_config = cfg.get("lora_optimizer", {})
            n_samples = lora_config.get("initial_samples", 9)

        samples = []
        for _ in range(n_samples):
            params = {
                "r": random.choice(LoraOptimizer.PARAM_SPACE["r"]),
                "lr": random.choice(LoraOptimizer.PARAM_SPACE["lr"]),
                "target_modules": LoraOptimizer.PARAM_SPACE["target_modules"],
                "lora_alpha": random.choice(LoraOptimizer.PARAM_SPACE["lora_alpha"]),
                "dropout": random.choice(LoraOptimizer.PARAM_SPACE["dropout"]),
                "num_epochs": random.choice(LoraOptimizer.PARAM_SPACE["num_epochs"]),
                "batch_size": random.choice(LoraOptimizer.PARAM_SPACE["batch_size"]),
            }
            params["alpha"] = params["r"] * LoraOptimizer.FIXED_PARAMS["alpha_ratio"]
            samples.append(params)
        return samples

    def train_lora(config: Dict[str, Any], cluster=None, model=None, tokenizer=None, output_dir_prefix=None) -> Dict[str, Any]:
        samples = cluster['samples']
        random.shuffle(samples)

        lora_config = cfg.get("lora_optimizer", {})
        val_ratio = lora_config.get("val_ratio", 0.2)
        val_size = max(1, int(len(samples) * val_ratio))
        train_samples = samples[:-val_size]
        val_samples = samples[-val_size:] if val_size > 0 else []

        train_texts = [
            f"{sample['text']}\n" 
            f"Prompt{sample['full_prompt']}\n"
            f"Answer:\n {sample['real_answer']}"  
            for sample in train_samples
        ]
        val_texts = [
            f"{sample['text']}\n"
            f"{sample['full_prompt']}\n"
            f"Answer:\n"
            for sample in val_samples
        ]

        train_encodings = tokenizer(
            train_texts,
            padding="max_length", 
            truncation=True, 
            return_tensors='pt',
            max_length=512  
        )
        train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask)
        batch_size = config["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        lora_config = LoraConfig(
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            lora_dropout=config["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        lora_model = get_peft_model(model, lora_config)
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=config["lr"])

        device_info = get_device_info()
        device = torch.device(device_info["device"])
        lora_model.to(device)

        if device_info["cuda_available"]:
            print(f"GPU Memory: {device_info['cuda_total_memory']:.2f} GB, Using batch size: {batch_size}")

        lora_model.train()
        for epoch in range(config["num_epochs"]):
            total_loss = 0
            for batch in train_loader:
                batch = to_device(batch, device)
                input_ids_batch, attention_mask_batch = batch

                outputs = lora_model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    labels=input_ids_batch
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {avg_train_loss:.4f}")

        val_loss = 0.0
        val_correct = 0  
        torch.cuda.empty_cache()
        if val_texts:
            torch.cuda.empty_cache()
            val_encodings = tokenizer(
                val_texts, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt",
                max_length=256  
            )
            val_input_ids = to_device(val_encodings["input_ids"], device)
            val_attention_mask = to_device(val_encodings["attention_mask"], device)

            true_answers = [s["real_answer"] for s in val_samples]

            val_dataset = TensorDataset(val_input_ids, val_attention_mask)
            val_loader = DataLoader(val_dataset, batch_size=config.get("eval_batch_size", 2))

            total = 0
            lora_model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    input_ids_batch, attention_mask_batch = [b.to(device) for b in batch]

                    outputs = lora_model(
                        input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        labels=input_ids_batch
                    )
                    val_loss += outputs.loss.item()

                    generated_ids = lora_model.generate(
                        input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        max_new_tokens=10,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    pred_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    batch_size = input_ids_batch.size(0)
                    for i in range(batch_size):
                        if true_answers[total].lower() in pred_answers[i].lower():
                            val_correct += 1
                        total += 1

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / total * 100
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        base_dir = cfg.get("buffer", {}).get("lora_buffer_dir", "/path/to/your/trigger/data/checkpoints")
        os.makedirs(base_dir, exist_ok=True)
        cfg_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
        timestamp = int(time.time())
        adapter_dir = os.path.join(base_dir, f"adapter_{cfg_hash}_{timestamp}")
        lora_model.save_pretrained(adapter_dir)

        result = {
            "config": config,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "checkpoint_path": adapter_dir
        }

        return result

    def local_fine_tune(best_config: Dict[str, Any], expand: int = 3) -> List[Dict[str, Any]]:
        r_choices = [max(2, best_config["r"] - 2), best_config["r"], best_config["r"] + 2]
        lr_choices = [best_config["lr"] * (0.7 + 0.1 * i) for i in range(expand)]

        fine_tune_samples = []
        for r in r_choices:
            for lr in lr_choices:
                new_cfg = best_config.copy()
                new_cfg["r"] = r
                new_cfg["lr"] = lr
                new_cfg["alpha"] = r * LoraOptimizer.FIXED_PARAMS["alpha_ratio"]
                fine_tune_samples.append(new_cfg)
        return fine_tune_samples

    def save_buffer(buffer: List[Dict[str, Any]], path: str = None):
        if path is None:
            buffer_config = cfg.get("buffer", {})
            base_dir = buffer_config.get("lora_buffer_dir", "/path/to/your/trigger/data/checkpoints")
            path = buffer_config.get("lora_buffer_path", os.path.join(base_dir, "lora_buffer.json"))
            os.makedirs(base_dir, exist_ok=True)

        serializable_buffer = []
        for item in buffer[:3]:
            serializable_item = {}
            for k, v in item.items():
                if k == "config":
                    serializable_config = {}
                    for ck, cv in v.items():
                        if isinstance(cv, list) and all(isinstance(x, str) for x in cv):
                            serializable_config[ck] = cv
                        elif isinstance(cv, (int, float, str, bool)) or cv is None:
                            serializable_config[ck] = cv
                        else:
                            serializable_config[ck] = str(cv)
                    serializable_item[k] = serializable_config
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    serializable_item[k] = v
                else:
                    serializable_item[k] = str(v)
            serializable_buffer.append(serializable_item)

        save_json(serializable_buffer, path)
        
    def load_lora_buffer(path=None):
        if path is None:
            buffer_config = cfg.get("buffer", {})
            path = buffer_config.get("lora_buffer_path")       
        data = load_json(path)
        for item in data:
            if "weight_path" in item:
                item["model_state"] = torch.load(item["weight_path"]) 
        return data

    def optimize_lora_for_cluster(cluster, model, tokenizer, top_k=3):
        def sort_by_score(res):
            return (-res["val_acc"], res["val_loss"])

        initial_samples = LoraOptimizer.sample_initial_parameters(n_samples=6)
        trained_results = [LoraOptimizer.train_lora(cfg, cluster, model, tokenizer) for cfg in initial_samples]
        trained_results.sort(key=sort_by_score)
        top_configs = [res["config"] for res in trained_results[:top_k]]

        fine_tuned_results = []
        for cfg in top_configs:
            fine_tune_cfgs = LoraOptimizer.local_fine_tune(cfg, expand=3)
            fine_tuned_results.extend([LoraOptimizer.train_lora(c, cluster, model, tokenizer) for c in fine_tune_cfgs])

        all_results = trained_results + fine_tuned_results
        all_results.sort(key=sort_by_score)
        LoraOptimizer.lora_buffer.extend(all_results[:3])
        LoraOptimizer.save_buffer(LoraOptimizer.lora_buffer)

        return all_results[0]["config"]
