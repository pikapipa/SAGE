import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_distances
import time
import torch
import re
from typing import Dict, Union
import random
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer
from peft import get_peft_model, PeftModel, LoraConfig
from utils import load_config
import pyarrow.parquet as pq
from trigger_lamma import TriggerDetector, LLaMAFeatureExtractor, AnomalyDetector
from data_buffer import AnomalyClusteringModule
from lora_buffer import LoraOptimizer
from env import setup_environment
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


cfg = load_config("./config.yaml")  
device = setup_environment(seed=1234)
tokenizer = LlamaTokenizer.from_pretrained('/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG', local_files_only=True)
model = LlamaForCausalLM.from_pretrained('/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG',  local_files_only=True, output_hidden_states=True).to(device)
buffer_config = cfg.get("buffer", {})
anomaly_cluster_path = buffer_config.get("anomaly_cluster_path", "/path/to/your/trigger/data/clusters.json")
model.eval()

def load_base_model():
    return model, tokenizer

def initialize_modules(model, tokenizer): 
    trigger_detector = TriggerDetector(
        model=model, 
        tokenizer=tokenizer,
    )
    
    data_buffer = AnomalyClusteringModule(
        llm_model = model,
        llm_tokenizer = tokenizer,
        embedding_model = "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5",
        buffer_path = anomaly_cluster_path,
        similarity_threshold = 0.8,
        cluster_size_threshold = 50
    )
    
    lora_optimizer = LoraOptimizer()
    return trigger_detector, data_buffer, lora_optimizer

def save_trigger_metrics_to_json(trigger_results, save_dir):
    y_true = [r['label'] for r in trigger_results]
    y_pred = [r['trigger'] for r in trigger_results]

    trigger_rate = sum(y_pred) / len(y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    trigger_summary = {
        "trigger_rate": trigger_rate,
        "confusion_matrix": conf_matrix,
        "accuracy": acc,
        "f1_score": f1
    }
    path = os.path.join(save_dir, "trigger_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trigger_summary, f, indent=2)
    print("[INFO] Trigger metrics saved to", path)

def save_data_buffer_summary(data_buffer, save_dir):
    summary = {
        "total_buckets": len(data_buffer.structure_cluster_map),
        "buckets": []
    }
    for structure_tag, cluster_ids in data_buffer.structure_cluster_map.items():
        bucket_info = {
            "bucket_id": structure_tag,
            "num_clusters": len(cluster_ids),
            "clusters": []
        }
        print(f"Bucket '{structure_tag}': {len(cluster_ids)} clusters")
        for cid in cluster_ids:
            cluster = data_buffer.clusters[cid]
            samples = cluster.get("samples", [])
            texts = [s.get("prompt_text", "") for s in samples]
            print(f"  - Cluster {cid} ({cluster['label']}): {len(samples)} samples")
            bucket_info["clusters"].append({
                "cluster_id": cid,
                "cluster_label": cluster["label"],
                "num_samples": len(samples),
                "samples": texts
            })
        summary["buckets"].append(bucket_info)

    path = os.path.join(save_dir, "bucket_cluster_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] DataBuffer bucket & cluster summary saved to", path)
    print("[INFO] Intra-cluster distances:")
    for cid, cluster in enumerate(data_buffer.clusters):
        samples = cluster.get("samples", [])
        if len(samples) < 2:
            print(f"Cluster {cid}: samples are too small, skip")
            continue
        embeddings = np.stack([
            s["embedding"].detach().cpu().numpy() if hasattr(s["embedding"], "detach") else np.array(s["embedding"])
            for s in samples
        ])
        center = np.mean(embeddings, axis=0)
        avg_dist = np.mean(cosine_distances(embeddings, [center]))
        print(f"Cluster {cid}: Avg Intra Distance = {avg_dist:.4f}")

    print("[INFO] Inter-cluster distances:")
    centers = {
        cid: np.mean(np.stack([
            s["embedding"].detach().cpu().numpy() if hasattr(s["embedding"], "detach") else np.array(s["embedding"])
            for s in cluster["samples"]
        ]), axis=0)
        for cid, cluster in enumerate(data_buffer.clusters)
    }
    cids = list(centers.keys())
    total_dist, count = 0, 0
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            dist = cosine_distances([centers[cids[i]]], [centers[cids[j]]])[0][0]
            print(f"Cluster {cids[i]} vs {cids[j]}: {dist:.4f}")
            total_dist += dist
            count += 1
    if count > 0:
        print(f"Avg Inter-Cluster Distance = {total_dist / count:.4f}")
    else:
        print("[WARN] just one cluster")
    
    print("\n[TSNE VISUALIZATION]")
    all_embeddings, all_labels = [], []
    for structure_tag, cluster_ids in data_buffer.structure_cluster_map.items():  
        for cid in cluster_ids:
            cluster = data_buffer.clusters[cid]
            for s in cluster["samples"]:
                emb = s["embedding"]
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                all_embeddings.append(emb)
                all_labels.append(f"{structure_tag}_{cid}") 
    if len(all_embeddings) >= 2:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(all_embeddings)//3), 
            random_state=42
        )
        tsne_embeds = tsne.fit_transform(np.stack(all_embeddings))
        
        plt.figure(figsize=(10, 8))
        unique_labels = sorted(set(all_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label in unique_labels:
            indices = [i for i, l in enumerate(all_labels) if l == label]
            plt.scatter(
                tsne_embeds[indices, 0], tsne_embeds[indices, 1],
                label=label,
                alpha=0.7,
                s=40
            )
        
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.title("t-SNE by Bucket & Cluster")
        plt.tight_layout()
        
        tsne_path = os.path.join(save_dir, "cluster_tsne.png")
        plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] 统一视图t-SNE已保存为 {tsne_path}")
    else:
        print("[WARN] 样本不足，无法进行t-SNE")


def evaluate_top_lora_adapters(
    lora_checkpoints: list,
    base_model,
    tokenizer,
    eval_data: list,
    save_dir: str,
    top_k: int = 3
):
    def extract_numeric_answer(output_text: str, label: str, max_val=1e6):
        try:
            numbers = re.findall(r"(-?\d+\.?\d*)", output_text)
            numbers = [float(n) for n in numbers if math.isfinite(float(n)) and abs(float(n)) < max_val]
            if not numbers:
                return None
            label_val = float(str(label).strip())
            pred = min(numbers, key=lambda x: abs(x - label_val))
            return int(pred), int(label_val)
        except Exception as e:
            print(f"[❌ 提取异常] {e}")
            return None

    os.makedirs(save_dir, exist_ok=True)
    selected = lora_checkpoints[:top_k]
    for i, ckpt in enumerate(selected):
        if not os.path.exists(ckpt):
            print(f"⚠️ LoRA Checkpoint 路径无效，跳过：{ckpt}")
            continue

        print(f"\n=== 正在评估第 {i+1} 个 LoRA Adapter: {ckpt} ===")
        lora_model = PeftModel.from_pretrained(base_model, ckpt)
        lora_model.to(base_model.device).eval()

        y_true, y_pred = [], []

        for sample in tqdm(eval_data, desc=f"[Adapter {i+1}]"):
            label = sample.get("label")
            if label is None:
                continue
            prompt = sample.get("prompt", "") + "\n" + sample.get("completion", "")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(base_model.device)

            with torch.no_grad():
                outputs = lora_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=64,
                    num_beams=3,
                    do_sample=False,
                    repetition_penalty=1.2, 
                    early_stopping=True,    
                    eos_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            result = extract_numeric_answer(output_text, label)
            if result:
                pred, true = result
                y_pred.append(pred)
                y_true.append(true)
            else:
                print(f"[DEBUG] 提取失败\n---Output---\n{output_text}\n---Label---\n{label}")
    
        exact_match = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        extraction_rate = len(y_true) / len(eval_data)

        print(f"[Adapter {i+1}]  Exact Match: {exact_match:.4f}")
        print(f"[Adapter {i+1}]  MAE: {mae:.4f}, MSE: {mse:.4f}")
        print(f"[Adapter {i+1}]  数字提取率: {extraction_rate:.4f} ({len(y_true)}/{len(eval_data)})")

        log_path = os.path.join(save_dir, f"adapter_eval_{i+1}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Adapter {i+1} Evaluation ===\n")
            f.write(f"Checkpoint: {ckpt}\n")
            f.write(f"Exact Match: {exact_match:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"Valid Predictions: {len(y_true)} / {len(eval_data)}\n")
            f.write(f"Numeric Extraction Rate: {extraction_rate:.4f}\n")

        json_path = os.path.join(save_dir, f"adapter_eval_{i+1}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": ckpt,
                "exact_match": exact_match,
                "mae": mae,
                "mse": mse,
                "valid_predictions": len(y_true),
                "total_samples": len(eval_data),
                "numeric_extraction_rate": extraction_rate
            }, f, indent=2)
        print(f"[INFO] Adapter {i+1} 评估结果保存至 {json_path}")

def run_batch_eval_from_dataset(model, tokenizer, trigger_detector, data_buffer, lora_buffer):
    save_dir = "/path/to/your/trigger/output/main"
    print("[STEP 1] loading sample...")
    def load_parquet_to_dicts(path):
        table = pq.read_table(path)
        return table.to_pylist()

    def load_parquet_as_dataset(path):
        data_dicts = load_parquet_to_dicts(path)
        return Dataset.from_list(data_dicts)

    id_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/trivia_qa/rc.nocontext/train-00000-of-00001.parquet").select(range(500))
    pubmed_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet").select(range(10))
    gsm8k_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/gsm8k/main/train-00000-of-00001.parquet").select(range(5))
    def format_prompt(example: Dict[str, str], source: str) -> Dict[str, str]:
        if source == "trivia_qa":
            question = f"""[Question]\n{example['question']}"""
            prompt = f"""[Context]\n\n[Answer]\n{example.get('answer', '')}"""
            label = 0
            real_answer = example.get("answer", "")
        elif source == "pubmedqa":
            question = f"""[Question]\n{example['question']}"""
            prompt = f"""[Context]\n{example.get('context', '')}\n\n[Answer Options]\n{example.get('long_answer', '')}\n\n[Final Decision]\n{example.get('final_decision', '')}"""
            label = 1
            real_answer = example.get("final_decision", "")
        elif source == "gsm8k":
            question = f"""[Question]\n{example['question']}""" 
            answer_text = example.get("answer", "").strip()          
            match = re.search(r"####\s*(-?\d+)", answer_text)
            real_answer = match.group(1) if match else ""
            prompt = f""" [Answer]\n{example.get('answer',"")} \n The final answer is: {real_answer}"""
            label = 1
        else:
            raise ValueError(f"Unknown source: {source}")

        return {"text": question, "full_prompt": prompt, "source": source, "label": label, "real_answer": str(real_answer)}
    
    id_data = id_dataset.map(lambda x: format_prompt(x, "trivia_qa")).remove_columns(set(id_dataset.column_names) - {"text", "label"})
    pubmed_data = pubmed_dataset.map(lambda x: format_prompt(x, "pubmedqa")).remove_columns(set(pubmed_dataset.column_names) - {"text", "label"})
    gsm8k_data = gsm8k_dataset.map(lambda x: format_prompt(x, "gsm8k")).remove_columns(set(gsm8k_dataset.column_names) - {"text", "label"})
    ood_full = concatenate_datasets([pubmed_data, gsm8k_data])
    id_half = id_data.shuffle(seed=42).select(range(200))

    json_dataset = load_dataset("json", data_files="/path/to/your/trigger/trigger-model/gsm8k_level1.jsonl")["train"]
    #json_dataset = json_dataset.select(range(300))
    def format_json_sample(example):
        question = f"[Question]\n{example['prompt']}"
        prompt = f"[Answer]\n{example.get('completion', '')}" 
        real_answer = example.get("label", "")
        return {
            "text": question,          
            "full_prompt": prompt,
            "label": 1,
            "source":"gsm8k",
            "real_answer": str(real_answer)
        }
    json_data = json_dataset.map(format_json_sample)
    ood_full = concatenate_datasets([ood_full, json_data])
    dataset = concatenate_datasets([id_half, ood_full]).shuffle(seed=123)
    warmup_samples = id_half.select(range(50))
    id_results = []
    for example in warmup_samples:
        input_text = example["text"]
        answer = str(example.get("answer") or example.get("label") or "")
        result = trigger_detector.detect(prompt=input_text, answer=answer)
        id_results.append(result)
    trigger_detector.detector.fit_thresholds(id_results, sim_percentile=[5, 95])

    y_true = []
    y_pred = []
    bleu_scores = []
    rouge_1_scores = []
    y_scores = []
    trigger_results = []

    feature_extractor = LLaMAFeatureExtractor()

    for sample in dataset:
        label = sample["label"]    
        answer = str(sample.get("answer") or sample.get("final_decision") or sample.get("label") or "")
        real_answer = sample["real_answer"]
        full_input = sample["full_prompt"]      
        input_text = sample["text"]        
        source = sample["source"]
        
        # tokenizer + inference
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            generated_ids = torch.argmax(outputs.logits, dim=-1)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        result = trigger_detector.detect(prompt=input_text, answer=answer)

        is_triggered = bool(result.get("trigger", False))    
        anomaly_score = float(result.get("similarity", 0.0)) 

        trigger_results.append({
            "label": int(label),   
            "trigger": int(is_triggered),
            "similarity": anomaly_score
        })
        if is_triggered:
            print(f"[TRIGGER] 发现异常样本，进行聚类处理: {input_text[:30]}...")

            with torch.no_grad():
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                outputs = model(**inputs, output_hidden_states=True)
            logits = feature_extractor.get_logits(outputs, 0).cpu().tolist() 

            sample_dict = {
                "text": input_text,
                "full_prompt": full_input,
                "source": source,
                "real_answer": sample["real_answer"]
            }
            processed_sample = data_buffer.process_anomaly_sample(sample_dict)
            cluster_id = processed_sample.get("cluster_id", -1)
            print(f"[CLUSTER] 当前样本归类到 Cluster {cluster_id}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_trigger_metrics_to_json(trigger_results, save_dir)
    save_data_buffer_summary(data_buffer, save_dir)

    print(f"\n[POST-PROCESS] 开始检查可用于LoRA微调的cluster...")
    data = load_dataset("json", data_files="/path/to/your/trigger/output/main/bucket_cluster_summary.json")["train"]
    summary = data[0] 

    buffer = []
    for bucket in summary["buckets"]:
        for cluster in bucket["clusters"]:
            if cluster["num_samples"] >= data_buffer.cluster_size_threshold:
                buffer.append(cluster)

    print(f"[INFO] 可用于微调的聚类数量：{len(buffer)}")

    for cluster in buffer:
        if cluster.get("lora_trained", False):
            print(f"[SKIP] Cluster {cluster['cluster_id']} 已完成微调，跳过")
            continue
        print(f"[FINETUNE] Cluster {cluster['cluster_id']} 达到阈值，开始LoRA微调")
        best_config = LoraOptimizer.optimize_lora_for_cluster(cluster, model, tokenizer, top_k=3)
        cluster["lora_trained"] = True
        print(f"[DONE] 微调完成，最佳配置: {best_config}")

    json_path = "/path/to/your/trigger/data/lora/lora_buffer.json"
    lora_checkpoints = load_lora_checkpoints_from_json(json_path, top_k=3)

    evaluate_top_lora_adapters(
        lora_checkpoints=lora_checkpoints,
        base_model=model,
        tokenizer=tokenizer,
        eval_data=json_dataset,
        save_dir=save_dir,
        top_k=3
    )
    print("[INFO] 批处理评估流程全部完成。")

def main():
    model, tokenizer = load_base_model()
    
    trigger_detector, data_buffer, lora_buffer = initialize_modules(model, tokenizer)
    
    run_batch_eval_from_dataset(model, tokenizer, trigger_detector, data_buffer, lora_buffer)

    data_buffer.save_buffer()
    LoraOptimizer.save_buffer(LoraOptimizer.lora_buffer, cfg['buffer']['lora_buffer_path'])

    print("[INFO] 所有流程已完成，程序结束。")
    
if __name__ == "__main__":
    main()
