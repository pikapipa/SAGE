from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
import torch
import json
from collections import defaultdict
import random

# === 模型路径 ===
base_model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
lora_adapter_path = "/path/to/your/trigger/baseline_OOD_q/lora-pubmedqa-kmeans"

# === 加载 tokenizer 和模型 ===
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

# === Step 1: 读取 parquet 文件并构造平衡训练集 ===
print("[Step 1] 读取 parquet 文件并构造平衡训练集...")
dataset = load_dataset("parquet", data_files={"train": "/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet"})
samples_by_label = defaultdict(list)

# 聚合不同标签样本
target_size = 250  # 每类上限
for s in dataset["train"]:
    label = s["final_decision"].strip().lower()
    if label in ["yes", "no"] and len(samples_by_label[label]) < target_size:
        samples_by_label[label].append(s)

# 拼接平衡数据并打乱
balanced_data = samples_by_label["yes"] + samples_by_label["no"] 
random.shuffle(balanced_data)

# 保存为 JSONL
output_path = "test_lora_pubmed_cluster.json"
with open(output_path, "w") as f:
    for s in balanced_data:
        prompt = f"Question: {s['question']}\nContext: {s['context']}\nAnswer: {s['long_answer']}\nFinal decision:"
        response = s["final_decision"].lower()
        f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

print(f"写入完成，共 {len(balanced_data)} 条样本 -> {output_path}")

# === Step 2: 构造样本列表并推理 ===
model.generation_config.temperature = None
model.generation_config.top_p = None

samples = []
for s in balanced_data:
    prompt = f"Question: {s['question']}\nContext: {s['context']}\nAnswer:"
    gt_response = s["long_answer"]
    decision = s.get("final_decision", "unknown")
    samples.append({
        "prompt": prompt,
        "ground_truth": gt_response,
        "final_decision": decision
    })

# === 推理 ===
for sample in tqdm(samples, desc="Generating"):
    inputs = tokenizer(sample["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    sample["prediction"] = answer

# === 推理 & 评估 ===
import json
from typing import List, Dict
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm

def extract_decision(text: str) -> str:
    text = text.lower()
    if "yes" in text:
        return "yes"
    else:
        return "no"

def evaluate_medical_qa(samples: List[Dict]) -> Dict:
    predictions = []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_rougeL = 0.0

    pred_texts = []
    ref_texts = []

    gold_decisions = []
    pred_decisions = []

    for item in tqdm(samples, desc="Evaluating"):
        pred = item["prediction"].strip()
        ref = item["ground_truth"].strip()
        rougeL_f1 = rouge.score(ref, pred)["rougeL"].fmeasure
        total_rougeL += rougeL_f1

        pred_texts.append(pred)
        ref_texts.append(ref)

        pred_dec = extract_decision(pred)
        true_dec = item.get("final_decision", "unknown").lower()

        pred_decisions.append(pred_dec)
        gold_decisions.append(true_dec)

        predictions.append({
            "prompt": item["prompt"],
            "ground_truth": ref,
            "prediction": pred,
            "rougeL_f1": rougeL_f1,
            "predicted_decision": pred_dec,
            "true_decision": true_dec
        })

    # === 平均 ROUGE-L F1 ===
    avg_rougeL = total_rougeL / len(samples)

    # === 分类指标（final decision）===
    cls_report = classification_report(gold_decisions, pred_decisions, digits=2, output_dict=True)

    return {
        "avg_rougeL_f1": avg_rougeL,
        "decision_classification_report": cls_report,
        "detailed_predictions": predictions
    }

# === 平均得分输出 ===
results = evaluate_medical_qa(samples)

print(f"\n✅ Avg ROUGE-L F1: {results['avg_rougeL_f1']:.4f}")
print("✅ Final Decision Classification Report:")
print(json.dumps(results['decision_classification_report'], indent=2))

