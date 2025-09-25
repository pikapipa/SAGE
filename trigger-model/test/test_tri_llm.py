import random
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from trigger_lamma import TriggerDetector, LLaMAFeatureExtractor, AnomalyDetector
from typing import Dict
import json
import pyarrow.parquet as pq
import re
from datasets import Dataset
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer 

def load_parquet_to_dicts(path):
    table = pq.read_table(path)
    return table.to_pylist()

def load_parquet_as_dataset(path):
    data_dicts = load_parquet_to_dicts(path)
    return Dataset.from_list(data_dicts)

def detect_trigger_anomaly(inputs=None, outputs=None, text=None, model=None, tokenizer=None, detector=None):
    global trigger_detector
    if detector is None:
        detector = TriggerDetector(model, tokenizer)

    if text is not None:

        result = detector.detect(prompt=text, answer="")
        return result['trigger']

    elif inputs is not None and outputs is not None:
        input_ids = inputs['input_ids'][0]
        input_text = tokenizer.decode(input_ids)

        result = detector.detect(prompt=input_text, answer="")
        return result['trigger']

    else:
        raise ValueError("必须提供 text 或者 (inputs, outputs)")

print("[INFO] 加载本地 Parquet 数据集并转换为 Huggingface Dataset...")
id_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/trivia_qa/rc.nocontext/train-00000-of-00001.parquet").select(range(10000))
pubmed_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet").select(range(5000))
lexglue_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet").select(range(5000))
gsm8k_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/gsm8k/main/train-00000-of-00001.parquet").select(range(5000))

print("[INFO] 格式标准化与融合...")

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

    elif source == "lex_glue":
        context = example.get("text", "")[:512]
        question = f"""[Context]\n{context}"""
        ending = example.get("ending", "")
        prompt = f"""[Answer Options]\n{ending}\n\n[Final Decision]\n{example.get('label', '')}"""
        label = 1
        real_answer = str(example.get("label", ""))

    elif source == "gsm8k":
        question = f"""[Question]\n{example['question']}"""
        prompt = f"""[Answer]\n{example['answer']}"""
        label = 1
        real_answer = example.get("answer", "")

    else:
        raise ValueError(f"Unknown source: {source}")

    return {"text": question, "full_prompt": prompt, "label": label, "source": source}

id_data = id_dataset.map(lambda x: format_prompt(x, "trivia_qa")).remove_columns(set(id_dataset.column_names) - {"text", "label"})
pubmed_data = pubmed_dataset.map(lambda x: format_prompt(x, "pubmedqa")).remove_columns(set(pubmed_dataset.column_names) - {"text", "label"})
lexglue_data = lexglue_dataset.map(lambda x: format_prompt(x, "lex_glue")).remove_columns(set(lexglue_dataset.column_names) - {"text", "label"})
gsm8k_data = gsm8k_dataset.map(lambda x: format_prompt(x, "gsm8k")).remove_columns(set(gsm8k_dataset.column_names) - {"text", "label"})

ood_full = concatenate_datasets([pubmed_data, lexglue_data, gsm8k_data])
ood_half = gsm8k_data.shuffle(seed=42).select(range(5000))
id_half = id_data.shuffle(seed=42).select(range(5000))
data_all = concatenate_datasets([id_half, ood_half]).shuffle(seed=123)

print("[INFO] 加载 LLaMA2-7B 模型...")
model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda().eval()

print("[INFO] 初始化 TriggerDetector...")
trigger_detector = TriggerDetector(
    model=model,
    tokenizer=tokenizer
)

print("[INFO] 使用部分 ID 数据预热检测器...")
warmup_samples = id_half.select(range(50))
id_results = []
for example in warmup_samples:
    input_text = example["text"]
    real_answer = str(example.get("answer") or example.get("label") or "")

    result = trigger_detector.detect(prompt=input_text, answer=real_answer)
    id_results.append(result)

trigger_detector.detector.fit_thresholds(id_results, sim_percentile=[5, 95])

print("[INFO] 执行Trigger检测...")

y_true = []
y_pred = []
bleu_scores = []
rouge_1_scores = []
y_scores = []

for example in data_all:
    input_text = example["text"]
    real_answer = str(example.get("answer") or example.get("final_decision") or example.get("label") or "")
    label = example["label"]  # 0=ID, 1=OOD

    # tokenizer + inference
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        generated_ids = torch.argmax(outputs.logits, dim=-1)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # === Trigger 检测 ===
    is_abnormal = trigger_detector.detect(prompt=input_text, answer=real_answer)["trigger"]
    trigger_detector.detector.fit_thresholds(id_results, sim_percentile=[5, 95])
    y_true.append(label)
    y_pred.append(1 if is_abnormal else 0)

    # === BLEU ===
    smooth_fn = SmoothingFunction().method1
    bleu = sentence_bleu(
        [real_answer.split()],
        generated_text.split(),
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth_fn
    )
    bleu_scores.append(bleu)

    # === ROUGE ===
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_result = rouge.score(real_answer, generated_text)
    rouge_1_scores.append(rouge_result["rouge1"].fmeasure)

    result = trigger_detector.detect(prompt=input_text, answer=real_answer)
    is_abnormal = result["trigger"]
    anomaly_score = result["similarity"]  # 或者你想用 margin / rouge / bleu

    y_scores.append(anomaly_score)

    print(f"[BLEU] {bleu:.4f} | [ROUGE-1 F] {rouge_result['rouge1'].fmeasure:.4f}")

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
trigger_rate = sum(y_pred) / len(y_pred)

print("[RESULT] 触发率:", trigger_rate)
print("[RESULT] 混淆矩阵:")
print(conf_matrix)
print("[RESULT] Accuracy:", acc)
print("[RESULT] F1 Score:", f1)
print("[RESULT] Precision:", precision)
print("[RESULT] Recall:", recall)

import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

save_dir = "/path/to/your/trigger/results"
os.makedirs(save_dir, exist_ok=True)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ID", "OOD"], yticklabels=["ID", "OOD"])
plt.xlabel("Predicted", fontsize=28)
plt.ylabel("Actual", fontsize=28)
plt.title("Confusion Matrix", fontsize=32)
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
print("✅ confusion_matrix.png saved!")
plt.close()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate", fontsize=28)
plt.ylabel("True Positive Rate", fontsize=28)
plt.title("ROC Curve", fontsize=32)
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_dir,"roc_curve.png"), dpi=300, bbox_inches="tight")
print("✅ roc_curve.png saved!")
plt.close()
