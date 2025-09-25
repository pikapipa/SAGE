import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import concatenate_datasets, load_dataset, Dataset
from tqdm import tqdm
import os
import re
import pyarrow.parquet as pq
import torch
from transformers import LlamaForCausalLM
import json
from typing import Dict, Union
from data_buffer import AnomalyClusteringModule
from transformers import AutoTokenizer, AutoModel
from utils import load_json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, f1_score
import pandas as pd
from scipy.spatial.distance import cosine
from PIL import Image
import torch
import types
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] 加载本地 Parquet 数据集...")
pubmed_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet", split="train").shuffle(seed=123).select(range(1000))
lexglue_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet", split="train").shuffle(seed=123).select(range(1000))
gsm8k_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/gsm8k/main/train-00000-of-00001.parquet", split="train").shuffle(seed=123).select(range(1000))

print("[INFO] 格式标准化...")
def format_prompt(example: Dict[str, str], source: str) -> Dict[str, str]:
    if source == "pubmedqa":
        question = f"""[Question]\n{example['question']}"""
        prompt = f"""[Context]\n{example.get('context', '')}\n\n[Answer Options]\n{example.get('long_answer', '')}\n\n[Final Decision]\n{example.get('final_decision', '')}"""
        real_answer = example.get("final_decision", "")
    elif source == "lex_glue":
        question = f"""[Context]\n{example.get('context', '')}"""
        prompt = f"""[Answer Options]\n{example.get('ending', '')}\n\n[Final Decision]\n{example.get('label', '')}"""
        real_answer = str(example.get("label", ""))
    elif source == "gsm8k":
        question = f"""[Question]\n{example['question']}"""
        prompt = f"""[Answer]\n{example['answer']}"""
        real_answer = example.get("answer", "")
    else:
        raise ValueError(f"Unknown source: {source}")
    return {"text": question, "full_prompt": prompt, "source": source, "real_answer": real_answer}

pubmed_data = pubmed_dataset.map(lambda x: format_prompt(x, "pubmedqa"), remove_columns=pubmed_dataset.column_names)
lexglue_data = lexglue_dataset.map(lambda x: format_prompt(x, "lex_glue"), remove_columns=lexglue_dataset.column_names)
gsm8k_data = gsm8k_dataset.map(lambda x: format_prompt(x, "gsm8k"), remove_columns=gsm8k_dataset.column_names)

data_all = concatenate_datasets([pubmed_data, lexglue_data, gsm8k_data]).shuffle(seed=123)
data_list = data_all.to_list()

with open("/path/to/your/trigger/output/test_ood_m.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)
print("[INFO] 已保存标准化后的初始问答数据到 test_ood_m.json")
print("[INFO] 初始化模型和聚类模块...")

tokenizer = AutoTokenizer.from_pretrained('/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG', local_files_only=True)
model = LlamaForCausalLM.from_pretrained('/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG',  local_files_only=True, output_hidden_states=True).to(device)
initial_data = load_json("/path/to/your/trigger/output/test_ood_m.json")

data_buffer = AnomalyClusteringModule(
    llm_model=model,
    llm_tokenizer=tokenizer,
    buffer_path=None,
    similarity_threshold="auto",
    cluster_size_threshold=20,
    initial_data=initial_data,
    enable_merge=True,             
    enable_stability_check=True  
)

embeddings_all, labels_all = [], []
snapshots = {300: None, 1500: None, 3000: None}

em_model_name = "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5"
em_tokenizer = AutoTokenizer.from_pretrained(em_model_name, local_files_only=True)
em_model = AutoModel.from_pretrained(em_model_name, local_files_only=True).to(device)

def get_embeddings(texts):
    embeddings = []
    em_model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Embedding"):
            try:
                inputs = em_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = em_model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(emb)
            except Exception as e:
                print(f"[WARN] 异常输入跳过: {e} | text[:50]={text[:50]}")
                continue
    return np.array(embeddings)

def get_latest_embedding_label(self):
    if hasattr(self, '_last_example') and self._last_example is not None:
        example = self._last_example
        text = example.get("text", "") + " " + example.get("full_prompt", "") + " " + example.get("real_answer", "")
        label = example.get("source", "") 
        emb = get_embeddings([text])[0] 
        return emb, label
    else:
        return None

def process_anomaly_sample_with_cache(self, example):
    self._last_example = example
    if hasattr(self, '_orig_process_anomaly_sample'):
        self._orig_process_anomaly_sample(example)
    else:
        pass

data_buffer.get_latest_embedding_label = types.MethodType(get_latest_embedding_label, data_buffer)
if not hasattr(data_buffer, '_orig_process_anomaly_sample'):
    data_buffer._orig_process_anomaly_sample = data_buffer.process_anomaly_sample
data_buffer.process_anomaly_sample = types.MethodType(process_anomaly_sample_with_cache, data_buffer)

print("[INFO] 开始流式送入并记录聚类标签...")
from datasets import load_dataset
ood_data = load_dataset("json", data_files="/path/to/your/trigger/output/test_ood_m.json")["train"]
#ood_data = ood_data.select(range(300))
initial_data = load_json("/path/to/your/trigger/output/test_ood_m.json")
print("[INFO] 执行启发式动态聚类...")
vocab_size = tokenizer.vocab_size
stages = [90, 300, 1500]
snapshots = {}
embedding_log = []
label_log = []

print("[INFO] 开始流式送入并记录聚类标签...")
for i, example in enumerate(tqdm(ood_data)):
    data_buffer.process_anomaly_sample(example)

    if hasattr(data_buffer, "get_latest_embedding_label"):
        res = data_buffer.get_latest_embedding_label()
        if res is not None:
            emb, label = res
            embedding_log.append(emb)
            label_log.append(label)

    if i + 1 in stages:
        emb_array = np.array(embedding_log)
        label_array = np.array(label_log)
        emb_array = np.stack(embedding_log, axis=0) 
        label_array = np.array(label_log)
        if label_array.ndim == 1:
            label_array = label_array[:, None]
        npz_path = f"snapshot_stage_{i+1}.npz"
        np.savez_compressed(npz_path, embeddings=emb_array, labels=label_array)
        csv_data = np.concatenate([emb_array, label_array], axis=1)
        csv_columns = [f"dim_{j}" for j in range(emb_array.shape[1])] + ["label"]
        pd.DataFrame(csv_data, columns=csv_columns).to_csv(f"snapshot_stage_{i+1}.csv", index=False)
        snapshots[i + 1] = {"embeddings": emb_array, "labels": label_array}

print("[INFO] 已保存 snapshot 到 csv/npz，开始绘制 t-SNE 聚类图...")
from sklearn.preprocessing import LabelEncoder
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
def plot_tsne_from_npz(npz_paths, stages, save_dir="./"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, len(npz_paths), figsize=(6 * len(npz_paths), 5))
    if len(npz_paths) == 1:
        axs = [axs]

    for idx, npz_path in enumerate(npz_paths):
        data = np.load(npz_path)
        embeddings = data['embeddings']
        labels = data['labels']

        n_samples = embeddings.shape[0]
        print(f"[INFO] 读取 {npz_path}，样本数: {n_samples}")

        if n_samples < 5:
            print(f"[WARN] 样本数太少，跳过绘图: {npz_path}")
            continue
        le = LabelEncoder()
        numeric_labels = le.fit_transform(labels.flatten())
        perplexity = min(30, n_samples // 3)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        axs[idx].set_title(f"Stage {stages[idx]} - n={n_samples}", fontsize=32)
        unique_labels = sorted(set(numeric_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            indices = [i for i, l in enumerate(numeric_labels) if l == label]
            axs[idx].scatter(
                reduced[indices, 0], reduced[indices, 1],
                label=le.inverse_transform([label])[0],
                alpha=0.7, s=40, color=color
            )
            cluster_center = np.mean(reduced[indices], axis=0)
            axs[idx].text(
                cluster_center[0], cluster_center[1], le.inverse_transform([label])[0],
                color='black', ha='center', va='center', fontsize=10, fontweight='bold'
            )
        unclustered_indices = [i for i, label in enumerate(numeric_labels) if label == -1]
        if unclustered_indices:
            axs[idx].scatter(
                reduced[unclustered_indices, 0], reduced[unclustered_indices, 1],
                label="Unclustered", color="gray", alpha=0.7, s=40, marker='x'
            )

        axs[idx].legend(loc='upper right', fontsize=18)
        axs[idx].tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    tsne_path = os.path.join(save_dir, "cluster_tsne_1.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] t-SNE图已保存到 {tsne_path}")
    plt.show()
npz_files = [
    "snapshot_stage_90.npz",
    "snapshot_stage_300.npz",
    "snapshot_stage_1500.npz",
]
stage_nums = [90, 300, 1500]

plot_tsne_from_npz(npz_files, stage_nums)
