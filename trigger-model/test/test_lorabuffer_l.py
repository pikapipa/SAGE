import torch
import os
import json
import re
import math
from collections import defaultdict
from typing import List, Dict, Any
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from utils import load_config, get_device_info
from lora_buffer import LoraOptimizer
from datasets import load_dataset

cfg = load_config("./config.yaml")

def extract_numeric_answer(output_text: str, label: str, max_val=1e6):
    match = re.search(r"final answer\s*is\s*:?\s*(-?\d+\.?\d*)", output_text.lower())
    if not match:
        lines = output_text.strip().split("\n")
        for line in reversed(lines[-5:]):
            match = re.search(r"(-?\d+\.?\d*)", line)
            if match:
                break
    if match:
        try:
            val = float(match.group(1))
            if math.isfinite(val) and abs(val) < max_val:
                return int(val), int(label)
        except:
            return None
    return None

def format_prompt(dataset: List[Dict[str, Any]], source: str = "lexglue-casehold") -> List[Dict[str, str]]:
    formatted_data = []

    for example in dataset:
        prompt = example.get("prompt", "")
        real_answer = example.get("completion", "").strip()

        # 提取 Context
        context_match = re.search(r"Context:(.*?)(?:\n\n|Options:)", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""

        # 提取 Options
        options_match = re.search(r"Options:\n(.*?)(?:\n\nAnswer:|\nAnswer:)", prompt, re.DOTALL)
        options = options_match.group(1).strip() if options_match else ""

        formatted_data.append({
            "text": context,
            "full_prompt": options,
            "real_answer": real_answer
        })

    return formatted_data

print("[Step 1] 读取并格式化 GSM8K 数据...") 
dataset_path = "/path/to/your/trigger/trigger-model/simple_tasks.json"
with open(dataset_path, "r") as f:
    data = json.load(f)
formatted_data = format_prompt(data, "Lex_glue")

print(f"Parsed {len(formatted_data)} samples.")

def test_lora_buffer_data_as_cluster():
    model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    base_model = LlamaForCausalLM.from_pretrained(model_path)

    device = torch.device(get_device_info()["device"])
    base_model.to(device).eval()

    data = formatted_data
    cluster = {
        "cluster_id": -1,
        "label": "lexglue_cluster",
        "samples": [dict(sample) for sample in data]
    }

    print(f"[INFO] 构造了 1 个虚拟 Cluster，样本数: {len(cluster['samples'])}")
    best_config = LoraOptimizer.optimize_lora_for_cluster(cluster, base_model, tokenizer, top_k=3)
    print(f"[INFO] 最优 LoRA 配置: {best_config}")

    for i, entry in enumerate(LoraOptimizer.lora_buffer[:3]):
        checkpoint_path = entry.get("checkpoint_path")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠️ 无效路径，跳过：{checkpoint_path}")
            continue

        print(f"\n=== 评估第 {i+1} 个 LoRA Adapter: {checkpoint_path} ===")

        lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        lora_model.to(device).eval()

        y_true, y_pred = [], []
        for sample in tqdm(data, desc=f"[Adapter {i+1}]"):
            prompt = f"{sample['text']}\n{sample['full_prompt']}"
            label = sample.get("real_answer", None)
            if label is None:
                continue
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

            with torch.no_grad():
                outputs = lora_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    eos_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            result = extract_numeric_answer(output_text, label)
            if result:
                pred, true_label = result
                y_pred.append(pred)
                y_true.append(true_label)
            else:
                print("[跳过] 无法提取有效数字")

        print("\n[结果评估]")
        print(f"准确率 (Accuracy): {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 分数 (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"精确率 (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"召回率 (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"有效预测数量: {len(y_true)}")
        log_path = "/path/to/your/trigger/output/adapter_eval_lexglue_{i+1}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[评估时间]: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"准确率 (Accuracy): {acc:.4f}\n")
            f.write(f"F1 分数 (weighted): {f1:.4f}\n")
            f.write(f"精确率 (weighted): {precision:.4f}\n")
            f.write(f"召回率 (weighted): {recall:.4f}\n")
            f.write(f"有效预测数量: {n_valid}\n")
            f.write("="*50 + "\n")

test_lora_buffer_data_as_cluster()
