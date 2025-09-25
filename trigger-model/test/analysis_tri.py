import matplotlib.pyplot as plt
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
ood_half = ood_full.shuffle(seed=42).select(range(500))
id_half = id_data.shuffle(seed=42).select(range(500))

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

save_dir = "/path/to/your/trigger/results"
os.makedirs(save_dir, exist_ok=True)

def sensitivity_analysis_threshold(trigger_detector, data_all, thresholds, save_dir):
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for th in thresholds:
        trigger_detector.detector.trigger_threshold = th

        y_true, y_pred = [], []
        for example in data_all:
            input_text = example["text"]
            real_answer = str(example.get("answer") or example.get("final_decision") or example.get("label") or "")
            label = example["label"]

            result = trigger_detector.detect(prompt=input_text, answer=real_answer, threshold=th)
            is_abnormal = result["trigger"]

            y_true.append(label)
            y_pred.append(1 if is_abnormal else 0)

        acc_list.append(accuracy_score(y_true, y_pred))
        prec_list.append(precision_score(y_true, y_pred, zero_division=0))
        rec_list.append(recall_score(y_true, y_pred, zero_division=0))
        f1_list.append(f1_score(y_true, y_pred, zero_division=0))

        print(f"[Threshold={th:.2f}] Acc={acc_list[-1]:.4f}, Prec={prec_list[-1]:.4f}, Rec={rec_list[-1]:.4f}, F1={f1_list[-1]:.4f}")

    max_f1 = max(f1_list)
    best_threshold = thresholds[f1_list.index(max_f1)]

    f1_robust_min = 0.97 * max_f1
    robust_idxs = [i for i, f in enumerate(f1_list) if f >= f1_robust_min]
    if robust_idxs:
        start_th = thresholds[robust_idxs[0]]
        end_th = thresholds[robust_idxs[-1]]
    else:
        start_th, end_th = best_threshold, best_threshold  # fallback

    plt.figure(figsize=(8,6))
    plt.plot(thresholds, acc_list, label="Accuracy", color="black", linestyle=':', linewidth=2)
    plt.plot(thresholds, prec_list, label="Precision", color="blue", linestyle='--', linewidth=2)
    plt.plot(thresholds, rec_list, label="Recall", color="green", linestyle='-.', linewidth=2)
    plt.plot(thresholds, f1_list, label="F1 Score", color="red", linestyle='-', linewidth=2)
    plt.axvspan(start_th, end_th, color='gray', alpha=0.2, label=f"Robust F1≥{f1_robust_min:.2f}")
    plt.scatter(best_threshold, max_f1, color='red', zorder=5)
    plt.annotate(f"Best F1 = {max_f1:.2f}\n@ Threshold = {best_threshold:.2f}",
                 xy=(best_threshold, max_f1),
                 xytext=(best_threshold + 0.05, max_f1 - 0.1),
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=24, color='red')

    plt.xlabel("Trigger Threshold", fontsize=28)
    plt.ylabel("Score", fontsize=28)
    plt.title("Sensitivity Analysis: Trigger Threshold vs Performance", fontsize=32)
    plt.legend(fontsize=24)
    plt.grid(True)

    save_path = os.path.join(save_dir, "threshold_sensitivity_analysis_annotated.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] 图已保存至: {save_path}")
# print("[INFO] 执行Trigger检测...")

# y_true = []
# y_pred = []
# bleu_scores = []
# rouge_1_scores = []
# y_scores = []


# for example in data_all:
#     input_text = example["text"]
#     real_answer = str(example.get("answer") or example.get("final_decision") or example.get("label") or "")
#     label = example["label"]  # 0=ID, 1=OOD
#     is_abnormal = trigger_detector.detect(prompt=input_text, answer=real_answer)["trigger"]
#     y_true.append(label)
#     y_pred.append(1 if is_abnormal else 0)

#     result = trigger_detector.detect(prompt=input_text, answer=real_answer)
#     is_abnormal = result["trigger"]
#     anomaly_score = result["similarity"] 

#     y_scores.append(anomaly_score)

# acc = accuracy_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# conf_matrix = confusion_matrix(y_true, y_pred)
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)

thresholds = [i / 100 for i in range(1, 101)] 
sensitivity_analysis_threshold(trigger_detector, data_all, thresholds, save_dir)

# trigger_rate = sum(y_pred) / len(y_pred)

# print("[RESULT] 触发率:", trigger_rate)
# print("[RESULT] 混淆矩阵:")
# print(conf_matrix)
# print("[RESULT] Accuracy:", acc)
# print("[RESULT] F1 Score:", f1)
# print("[RESULT] Precision:", precision)
# print("[RESULT] Recall:", recall)
