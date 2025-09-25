import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from trigger_lamma import TriggerDetector, LLaMAFeatureExtractor, AnomalyDetector
from typing import Dict
import json
import pyarrow.parquet as pq
import re
from datasets import Dataset
import seaborn as sns
import os
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

def weight_sensitivity_heatmap(trigger_detector, data_all, save_dir=None): 
    bleu_rouge_weights = np.linspace(0.0, 0.33, 8)
    sim_weights_full = np.linspace(0.0, 0.33, 8)

    f1_records = []
    acc_records = []

    print("[INFO] 开始网格遍历并计算 F1/ACC ...")
    for w_b_r in bleu_rouge_weights:
        sim_max = 1.0 - 2 * w_b_r
        sim_weights = sim_weights_full[sim_weights_full <= sim_max]

        for w_sim in sim_weights:
            w_margin = 1.0 - 2 * w_b_r - w_sim
            weights = {
                'margin': w_margin,
                'bleu': w_b_r,
                'rouge': w_b_r,
                'similarity': w_sim
            }
            trigger_detector.detector.weights = weights

            y_true, y_pred = [], []
            for example in data_all:
                input_text = example["text"]
                real_answer = str(example.get("answer") or example.get("final_decision") or example.get("label") or "")
                label = example["label"]

                result = trigger_detector.detect(prompt=input_text, answer=real_answer)
                trigger = result["trigger"]
                y_true.append(label)
                y_pred.append(1 if trigger else 0)

            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)

            f1_records.append((w_b_r, w_sim, f1))
            acc_records.append((w_b_r, w_sim, acc))

            print(f"[BR={w_b_r:.2f}, SIM={w_sim:.2f}] MARGIN={w_margin:.2f} → F1={f1:.4f}, ACC={acc:.4f}")

    heatmap_f1 = np.full((8, 8), np.nan)
    heatmap_acc = np.full((8, 8), np.nan)

    b_r_to_idx = {v: i for i, v in enumerate(bleu_rouge_weights)}
    sim_to_idx = {v: i for i, v in enumerate(sim_weights_full)}

    for w_b_r, w_sim, f1 in f1_records:
        i = b_r_to_idx[w_b_r]
        j = sim_to_idx[w_sim]
        heatmap_f1[i, j] = f1

    for w_b_r, w_sim, acc in acc_records:
        i = b_r_to_idx[w_b_r]
        j = sim_to_idx[w_sim]
        heatmap_acc[i, j] = acc

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.heatmap(heatmap_f1, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
                xticklabels=[f"{x:.2f}" for x in sim_weights_full],
                yticklabels=[f"{y:.2f}" for y in bleu_rouge_weights],
                cbar_kws={"label": "F1 Score"})
    axes[0].set_title("F1 Score Heatmap", fontsize=32)
    axes[0].set_xlabel("Similarity Weight", fontsize=28)
    axes[0].set_ylabel("BLEU = ROUGE Weight", fontsize=28)

    sns.heatmap(heatmap_acc, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=[f"{x:.2f}" for x in sim_weights_full],
                yticklabels=[f"{y:.2f}" for y in bleu_rouge_weights],
                cbar_kws={"label": "Accuracy"})
    axes[1].set_title("Accuracy Heatmap", fontsize=28)
    axes[1].set_xlabel("Similarity Weight", fontsize=28)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "combined_weight_sensitivity_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 热力图已保存至: {save_path}")
    plt.show()

weight_sensitivity_heatmap(trigger_detector, data_all, save_dir)