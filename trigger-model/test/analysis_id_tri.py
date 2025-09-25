import matplotlib.pyplot as plt
import random
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from trigger_lamma import TriggerDetector, LLaMAFeatureExtractor, AnomalyDetector
from typing import Dict
import json
import numpy as np
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
id_dataset = load_parquet_as_dataset("/path/to/your/trigger/datasets/bio/trivia_qa/rc.nocontext/train-00000-of-00001.parquet").select(range(50000))
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
ood_half = ood_full.shuffle(seed=42).select(range(1000))
id_half = id_data.shuffle(seed=42).select(range(1000))

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

save_dir = "/path/to/your/trigger/results"
os.makedirs(save_dir, exist_ok=True)

def collect_scores_per_metric(trigger_detector, dataset):
    score_dict = {key: [] for key in ['margin', 'bleu', 'rouge', 'similarity']}

    for example in dataset:
        input_text = example["text"]
        real_answer = str(example.get("answer") or example.get("final_decision") or example.get("label") or "")
        inputs = trigger_detector.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(trigger_detector.model.device)
        with torch.no_grad():
            output = trigger_detector.model(**inputs, return_dict=True)
        logits = trigger_detector.extractor.get_logits(output, 0)[-1]
        margin = trigger_detector.detector.logits_margin(logits)
        norm_margin = margin / trigger_detector.detector.max_margin
        score_dict["margin"].append(norm_margin)
        result = trigger_detector.detect(prompt=input_text, answer=real_answer)
        for key in ["bleu", "rouge", "similarity"]:
            score_dict[key].append(result[key])

    return score_dict

import seaborn as sns
import pandas as pd

def plot_boxplot(id_scores, ood_scores, save_path):
    all_data = []
    for v in id_scores[metric]:
        all_data.append({"Metric": metric, "Score": v, "Type": "ID"})
    for v in ood_scores[metric]:
        all_data.append({"Metric": metric, "Score": v, "Type": "OOD"})
    df = pd.DataFrame(all_data)
    plt.figure(figsize=(8, 6))
    flier_props = dict(marker='o', markerfacecolor='white', markersize=5, linestyle='none')
    median_props = dict(color='#7B3F3F', linewidth=2)
    palette = {"ID": "#4D5D6C", "OOD": "#333333"}

    ax = sns.boxplot(
        x="Metric",
        y="Score",
        hue="Type",
        data=df,
        palette=palette,
        flierprops=flier_props,
        medianprops=median_props
    )
    medians = df.groupby(["Metric", "Type"])["Score"].median().reset_index()
    positions = {}
    for i, (metric, typ) in enumerate(medians[["Metric", "Type"]].values):
        x = list(df["Metric"].unique()).index(metric)
        offset = -0.2 if typ == "ID" else 0.2
        x_pos = x + offset

        y_pos = medians[(medians["Metric"] == metric) & (medians["Type"] == typ)]["Score"].values[0]
        ax.text(x_pos, y_pos + 0.02, f"{y_pos:.2f}", ha='center', va='bottom', fontsize=16, color='#7B3F3F')

    plt.title("Score Distribution: ID vs OOD", fontsize=32)
    plt.xlabel("Metric", fontsize=28)
    plt.ylabel("Score", fontsize=28)
    plt.legend(title=None, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    os.makedirs(save_path, exist_ok=True)
    save_fp = os.path.join(save_path, "boxplot_score_comparison.png")
    plt.savefig(save_fp, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] 箱线图已保存至: {save_fp}")

def run_fpr_comparison_id_vs_ood(trigger_detector, id_data, ood_data, save_dir):
    print("[INFO] 计算 ID 样本 FPR ...")
    id_results = collect_scores_per_metric(trigger_detector, id_data)

    print("[INFO] 计算 OOD 样本 FPR ...")
    ood_results = collect_scores_per_metric(trigger_detector, ood_data)

    print("[INFO] 绘图对比 ID vs OOD FPR ...")
    plot_boxplot(id_results, ood_results, save_dir)
    
run_fpr_comparison_id_vs_ood(trigger_detector, id_half, ood_half, save_dir)