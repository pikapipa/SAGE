from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, accuracy_score, f1_score, precision_score, recall_score
import torch
import json
import re
from tqdm import tqdm
import random

# === 模型路径 ===
base_model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
lora_adapter_path = "/path/to/your/trigger/baseline_OOD_q/lora-lexglue-kmeans"

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
with open("val_multitask.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

print(f"[INFO] 加载验证样本: {len(data)}")

# === 推理 & 提取预测 ===
y_true, y_pred = [], []

print("[INFO] 开始推理...")
for item in tqdm(data):
    prompt = item["prompt"]
    label = item["completion"]

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,
            do_sample=False
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # === 提取真实标签（label）中的数字 ===
    label_text = item["completion"]
    label_match = re.search(r'\b(\d+)\b', label_text)
    
    # === 提取预测值（pred）中的数字 ===
    pred = output_text.replace(prompt, "").strip()
    pred_match = re.search(r'\b(\d+)\b', pred)

    if label_match and pred_match:
        y_true.append(int(label_match.group(1)))
        y_pred.append(int(pred_match.group(1)))

# === 评估结果 ===
print("\n[结果评估]")
print(f"准确率 (Accuracy): {accuracy_score(y_true, y_pred):.4f}")
print(f"F1 分数 (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
print(f"精确率 (weighted): {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"召回率 (weighted): {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"有效预测数量: {len(y_true)}")
