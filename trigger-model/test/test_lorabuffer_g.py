import torch
import os
import json
import re
import math
from typing import List, Dict, Any
from lora_buffer import LoraOptimizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import load_config, get_device_info
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

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

def test_lora_buffer_data_as_cluster():
    model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    base_model = LlamaForCausalLM.from_pretrained(model_path)

    device = torch.device(get_device_info()["device"])
    base_model.to(device).eval()

    data = load_dataset("json", data_files="/path/to/your/trigger/trigger-model/gsm8k_level1.jsonl")["train"]
    cluster = {
        "cluster_id": -1,
        "label": "pubmedqa_cluster",
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
            prompt = f"{sample['prompt']}\n{sample['completion']}"
            label = sample.get("label", None)
            if label is None:
                continue
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

            with torch.no_grad():
                outputs = lora_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=1024,
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

        print(f"\n[Adapter {i+1}] 评估结果：")
        if len(y_true) == 0:
            print("❌ 无有效预测结果，可能输出格式有误")
        else:
            exact_match = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            print(f"✅ 精确匹配率 (Exact Match): {exact_match:.4f}")
            print(f"📉 平均绝对误差 (MAE): {mae:.4f}")
            print(f"📉 均方误差 (MSE): {mse:.4f}")
            print(f"✅ 有效预测数量: {len(y_true)} / {len(data)}")
            print(f"✅ 数字提取成功率: {len(y_true)/len(data):.4f}")
            result_log_path = f"/path/to/your/trigger/output/adapter_eval_gsm8k500_{i+1}.log"  
            with open(result_log_path, "w", encoding="utf-8") as f:
                f.write(f"=== Adapter {i+1} 评估结果 ===\n")
                f.write(f"Checkpoint: {checkpoint_path}\n")
                f.write(f"✅ 精确匹配率 (Exact Match): {exact_match:.4f}\n")
                f.write(f"📉 平均绝对误差 (MAE): {mae:.4f}\n")
                f.write(f"📉 均方误差 (MSE): {mse:.4f}\n")
                f.write(f"✅ 有效预测数量: {len(y_true)} / {len(data)}\n")
                f.write(f"✅ 数字提取成功率: {len(y_true)/len(data):.4f}\n")      

test_lora_buffer_data_as_cluster()