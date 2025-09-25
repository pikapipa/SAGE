from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import torch
import json
import re
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error
)
from tqdm import tqdm
import random
import math

# === 模型路径 ===
base_model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
lora_adapter_path = "/path/to/your/trigger/baseline_OOD_q/lora-gsm8k-kmeans"

# === 加载 tokenizer 和 LoRA 模型 ===
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

# === 加载验证数据 ===
test_samples = "test_gsm8k.jsonl"
with open(test_samples, "r") as f:
    data = [json.loads(line) for line in f]

print(f"[INFO] 加载验证样本: {len(data)}")

# === 评估 ===
y_true, y_pred = [], []
MAX_REASONABLE_ANSWER = 1e6

print("\n[INFO] 开始评估每一道题...")

for item in tqdm(data):
    prompt = item["prompt"]
    label = item["label"]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[Debug] Output: {output_text}")

    match = re.search(r"final answer\s*is\s*:?\s*(-?\d+\.?\d*)", output_text.lower())

    # Fallback: 从最后几行中提取数字
    if not match:
        lines = output_text.strip().split("\n")
        for line in reversed(lines[-5:]):  # 限制在最后5行中搜索
            match = re.search(r"(-?\d+\.?\d*)", line)
            if match:
                break

    if match:
        try:
            val = float(match.group(1))
            if math.isfinite(val) and abs(val) < MAX_REASONABLE_ANSWER:
                pred = int(val)
                true_label = int(label)
                y_pred.append(pred)
                y_true.append(true_label)
            else:
                print(f"[跳过] 不合理数值: {val}")
        except Exception as e:
            print(f"[跳过] 解析失败: {e}")
            continue
    else:
        print("[跳过] 无法提取有效答案")

# === 评估结果 ===
print("\n[结果评估]")
if len(y_true) == 0:
    print("❌ 无有效预测结果，可能输出格式有误")
else:
    exact_match = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"精确匹配率 (Exact Match): {exact_match:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"有效预测数量: {len(y_true)} / {sample_size}")
    print(f"数字提取成功率: {len(y_true) / sample_size:.4f}")
