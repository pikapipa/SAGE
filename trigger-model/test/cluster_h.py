import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import concatenate_datasets, load_dataset, Dataset
from tqdm import tqdm
import logging
import os
import re
from collections import defaultdict
from transformers import LlamaForCausalLM
import pyarrow.parquet as pq
import torch
import hdbscan
import json
from typing import Dict, Union
from data_buffer import AnomalyClusteringModule
from transformers import AutoTokenizer, AutoModel
from utils import load_json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

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

data_all = concatenate_datasets([pubmed_data, lexglue_data, gsm8k_data])
#data_all = data_all.select(range(300))
data_list = data_all.to_list()

for i, sample in enumerate(data_list):
    sample["global_index"] = i

with open("/path/to/your/trigger/output/test_ood_m.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)
print("[INFO] 已保存标准化后的初始问答数据到 test_ood_m.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
em_model_name = "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained('/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG', local_files_only=True)
model = LlamaForCausalLM.from_pretrained('/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG',  local_files_only=True, output_hidden_states=True).to(device)
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
    
context = [d["text"] + "\n\n" + d["full_prompt"] for d in data_list]
labels = [d["source"] for d in data_list]
embeddings = get_embeddings(context)

le = LabelEncoder()
true_labels = le.fit_transform(labels)
save_dir = "/path/to/your/trigger/results"
os.makedirs(save_dir, exist_ok=True)
def save_cluster_summary(data_buffer, method_name, save_dir, hdb_labels=None, embeddings=None):
    summary = {
        "total_buckets": 1,
        "buckets": []
    }

    if hdb_labels is not None and embeddings is not None:
        print(f"\n[INFO] {method_name} 聚类结果统计:")
        unique_labels = set(hdb_labels)
        bucket_info = {
            "bucket_id": "hdbscan_bucket",
            "num_clusters": len(unique_labels),
            "clusters": []
        }

        for label in unique_labels:
            cluster_samples = [embeddings[i] for i, lbl in enumerate(hdb_labels) if lbl == label]
            cluster_texts = [f"Sample {i}" for i, lbl in enumerate(hdb_labels) if lbl == label]  # Or some actual text data
            print(f"  - Cluster {label}: {len(cluster_samples)} samples")
            
            bucket_info["clusters"].append({
                "cluster_id": label,
                "cluster_label": f"hdbscan_cluster_{label}",
                "num_samples": len(cluster_samples),
                "samples": cluster_texts
            })

        summary["buckets"].append(bucket_info)
    elif data_buffer:
        print(f"\n[INFO] {method_name} 聚类结果统计:")
        for structure_tag, cluster_ids in data_buffer.structure_cluster_map.items():
            print(f"Bucket '{structure_tag}': {len(cluster_ids)} clusters")
            bucket_info = {
                "bucket_id": structure_tag,
                "num_clusters": len(cluster_ids),
                "clusters": []
            }
            for cid in cluster_ids:
                cluster = data_buffer.clusters[cid]
                samples = cluster.get("samples", [])
                texts = [s.get("prompt_text", "") for s in samples]
                print(f"  - Cluster {cid} ({cluster['label']}): {len(samples)} samples")
                bucket_info["clusters"].append({
                    "cluster_id": cid,
                    "cluster_label": cluster["label"],
                    "num_samples": len(samples),
                    "samples": texts
                })
            summary["buckets"].append(bucket_info)

    summary_path = os.path.join(save_dir, f"{method_name}_cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 聚类结构已保存到：{summary_path}")

    summary_path = os.path.join(save_dir, f"{method_name}_cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 聚类结构已保存到：{summary_path}")
    
    # Save the summary to a JSON file
    summary_path = os.path.join(save_dir, f"{method_name}_cluster_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] {method_name} 聚类结构已保存到：{summary_path}")

def analyze_cluster_sources(data_buffer, name=""):
    cluster_source_summary = []
    for bucket_tag, bucket in data_buffer.bucket_buffers.items():  
        if "clusters" not in bucket:
            logging.warning(f"Bucket {bucket_tag} does not contain 'clusters'. Skipping.")
            continue
        for cluster in bucket["clusters"]:
            sources = [sample.get("source", "unknown") for sample in cluster["samples"]]
            source_set = set(sources)
            cluster_info = {
                "cluster_id": cluster["cluster_id"],
                "num_samples": len(sources),
                "source_distribution": {s: sources.count(s) for s in source_set},
                "is_homogeneous": len(source_set) == 1
            }
            cluster_source_summary.append(cluster_info)

    print(f"\n【{name} 聚类来源分析】")
    for info in cluster_source_summary:
        print(f"- Cluster {info['cluster_id']} | Count: {info['num_samples']} | Same Source: {info['is_homogeneous']} | Distribution: {info['source_distribution']}")
    return cluster_source_summary

def get_all_cluster_labels(data_buffer, return_source=False):
    labels = []
    indices = []
    # Check if the data_buffer has the necessary attribute
    if not hasattr(data_buffer, 'bucket_buffers') or not data_buffer.bucket_buffers:
        print("[ERROR] 'bucket_buffers' is empty or not found in data_buffer.")
        return labels, indices  # Return empty lists if the structure is incorrect

    for bucket_tag, bucket in data_buffer.bucket_buffers.items():
        if "clusters" not in bucket:
            logging.warning(f"Bucket {bucket_tag} does not contain 'clusters'. Skipping.")
            continue
        for cluster in bucket["clusters"]:
            cluster_id = cluster["cluster_id"]
            for sample in cluster["samples"]:
                labels.append(cluster_id)
                if return_source:
                    if "global_index" not in sample:
                        raise KeyError(f"Missing 'global_index' key in sample: {sample}")
                    indices.append(sample["global_index"])

    if return_source:
        print(f"[DEBUG] indices: {indices}")  # Debugging output
        return labels, indices
    return labels

def prepare_cluster_evaluation(data_buffer, true_labels, embeddings, name=""):
    labels, indices = get_all_cluster_labels(data_buffer, return_source=True)
    if len(labels) == 0:  # Check if labels are empty
        print("[ERROR] No labels found in the cluster data.")
        return {}  # Return empty metrics if no labels are available

    true_labels_aligned = [true_labels[i] for i in indices]
    embeddings_aligned = [embeddings[i] for i in indices]
    return evaluate_all_metrics(name, true_labels_aligned, labels, embeddings_aligned)

from datasets import load_dataset
ood_data = load_dataset("json", data_files="/path/to/your/trigger/output/test_ood_m.json")["train"]
#ood_data = ood_data.select(range(300))
initial_data = load_json("/path/to/your/trigger/output/test_ood_m.json")

from data_buffer import AnomalyClusteringModule
data_buffer_0 = AnomalyClusteringModule(
    llm_model=model,
    llm_tokenizer=tokenizer,
    buffer_path=None,
    similarity_threshold="auto",
    cluster_size_threshold=20,
    initial_data=initial_data,
)

print(f"[INFO] 初始结构感知分桶数量：{len(data_buffer_0.clusters)}")

#ood_data_0 = ood_data.select(range(900))
print("[INFO] 执行启发式动态聚类...")
vocab_size = tokenizer.vocab_size
for example in tqdm(ood_data):
    data_buffer_0.process_anomaly_sample(example)

print("[DEBUG] bucket_buffers after processing:")
for bucket_tag, bucket in data_buffer_0.bucket_buffers.items():
    print(f"Bucket {bucket_tag} is of type: {type(bucket)}")
    if isinstance(bucket, dict):
        print(f"  Bucket {bucket_tag}: {len(bucket.get('clusters', []))} clusters")
    elif isinstance(bucket, list):
        print(f"  Bucket {bucket_tag} is a list with {len(bucket)} items")
    else:
        print(f"  Unknown type for bucket {bucket_tag}")

custom_labels_0 = get_all_cluster_labels(data_buffer_0)

if len(data_buffer_0.bucket_buffers) == 0:
    print("[ERROR] bucket_buffers is empty after processing samples!")
else:
    print("[INFO] bucket_buffers populated successfully.")
    
data_buffer_1 = AnomalyClusteringModule(
    llm_model=model,
    llm_tokenizer=tokenizer,
    buffer_path=None,
    similarity_threshold="auto",
    cluster_size_threshold=20,
    initial_data=initial_data,
    enable_stability_check=True
)

print(f"[INFO] 初始结构感知分桶数量：{len(data_buffer_1.clusters)}")

print("[INFO] 执行启发式动态聚类...")
vocab_size = tokenizer.vocab_size
for example in tqdm(ood_data):
    data_buffer_1.process_anomaly_sample(example)

print("[DEBUG] bucket_buffers after processing:")
for bucket_tag, bucket in data_buffer_1.bucket_buffers.items():
    print(f"Bucket {bucket_tag} is of type: {type(bucket)}")
    if isinstance(bucket, dict):
        print(f"  Bucket {bucket_tag}: {len(bucket.get('clusters', []))} clusters")
    elif isinstance(bucket, list):
        print(f"  Bucket {bucket_tag} is a list with {len(bucket)} items")
    else:
        print(f"  Unknown type for bucket {bucket_tag}")
          
custom_labels_1 = get_all_cluster_labels(data_buffer_1)

data_buffer_2 = AnomalyClusteringModule(
    llm_model=model,
    llm_tokenizer=tokenizer,
    buffer_path=None,
    similarity_threshold="auto",
    cluster_size_threshold=20,
    initial_data=initial_data,
    enable_merge=True
)

print(f"[INFO] 初始结构感知分桶数量：{len(data_buffer_2.clusters)}")

print("[INFO] 执行启发式动态聚类...")
vocab_size = tokenizer.vocab_size
for example in tqdm(ood_data):
    data_buffer_2.process_anomaly_sample(example)

print("[DEBUG] bucket_buffers after processing:")
for bucket_tag, bucket in data_buffer_2.bucket_buffers.items():
    print(f"Bucket {bucket_tag} is of type: {type(bucket)}")
    if isinstance(bucket, dict):
        print(f"  Bucket {bucket_tag}: {len(bucket.get('clusters', []))} clusters")
    elif isinstance(bucket, list):
        print(f"  Bucket {bucket_tag} is a list with {len(bucket)} items")
    else:
        print(f"  Unknown type for bucket {bucket_tag}")
          
custom_labels_2 = get_all_cluster_labels(data_buffer_2)

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

print(f"[INFO] 初始结构感知分桶数量：{len(data_buffer.clusters)}")

print("[INFO] 执行启发式动态聚类...")
vocab_size = tokenizer.vocab_size
for example in tqdm(ood_data):
    data_buffer.process_anomaly_sample(example)

custom_labels = get_all_cluster_labels(data_buffer)


print("[INFO] 执行 StreamingHDBSCAN 聚类...")
save_cluster_summary(data_buffer_0, "StreamingHDBSCAN", save_dir)

print("[INFO] 执行 StabilityCheckHDBSCAN 聚类...")
save_cluster_summary(data_buffer_1, "StabilityCheckHDBSCAN", save_dir)

print("[INFO] 执行 AnomalyClusteringModule 聚类...")
save_cluster_summary(data_buffer_2, "MergeHDBSCAN", save_dir)

print("[INFO] 执行 AnomalyClusteringModule 聚类...")
save_cluster_summary(data_buffer, "AnomalyClusteringModule", save_dir)

def load_cluster_summary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_cluster_sample_std(data_buffer):
    cluster_sizes = [len(cluster["samples"]) for cluster in data_buffer.clusters]
    std_dev = np.std(cluster_sizes)
    return std_dev

def central_similarity(embeddings, y_pred):
    cluster_centers = {}
    for i, label in enumerate(y_pred):
        if label not in cluster_centers:
            cluster_centers[label] = []
        cluster_centers[label].append(embeddings[i])
    
    cluster_center_vectors = {
        label: np.mean(cluster_embeddings, axis=0)
        for label, cluster_embeddings in cluster_centers.items()
    }
    
    similarities = []
    for i, label in enumerate(y_pred):
        sample = embeddings[i].reshape(1, -1)
        center = cluster_center_vectors[label].reshape(1, -1)
        sim = cosine_similarity(sample, center)[0][0]
        similarities.append(sim)
    
    return np.mean(similarities)  

def extract_labels_and_indices(summary):
    labels = []
    indices = []
    
    for bucket in summary.get("buckets", []):
        for cluster in bucket.get("clusters", []):
            cluster_id = cluster["cluster_id"]
            for sample in cluster["samples"]:
                labels.append(cluster_id)
                indices.append(sample.get("global_index", -1))
    
    return labels, indices

def calculate_cluster_sample_std(data_buffer):
    cluster_sizes = [len(cluster["samples"]) for cluster in data_buffer.clusters]
    std_dev = np.std(cluster_sizes)
    return std_dev

def evaluate_all_metrics(method_name, y_true, y_pred, emb, data_buffer=None):
    metrics = {}
    if len(y_true) == 0 or len(y_pred) == 0:  
        print(f"[ERROR] Empty labels found. Skipping metrics calculation.")
        return metrics  

    metrics['ARI'] = adjusted_rand_score(y_true, y_pred)
    metrics['NMI'] = normalized_mutual_info_score(y_true, y_pred)
    metrics['CS'] = central_similarity(emb, y_pred)
    metrics['Hom.'] = homogeneity_score(y_true, y_pred)
    metrics['Comp.'] = completeness_score(y_true, y_pred)

    if data_buffer is not None:
        std_dev = calculate_cluster_sample_std(data_buffer)
        metrics['CS-STD_raw'] = std_dev
        print(f"CS-STD (raw): {std_dev:.4f}")
    else:
        metrics['CS-STD_raw'] = -1
        print("No data_buffer provided, CS-STD skipped.")

    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}" if isinstance(v, float) else f" - {k}: {v}")
    return metrics

def process_and_evaluate(summary_path, true_labels, embeddings, method_name, data_buffer=None):
    summary = load_cluster_summary(summary_path)
    labels, indices = extract_labels_and_indices(summary)

    true_labels_aligned = [true_labels[i] for i in indices]
    embeddings_aligned = [embeddings[i] for i in indices]

    return evaluate_all_metrics(method_name, true_labels_aligned, labels, embeddings_aligned, data_buffer)

custom_summary_0_path = "/path/to/your/trigger/results/StreamingHDBSCAN_cluster_summary.json"
metrics_custom_0 = process_and_evaluate(custom_summary_0_path, true_labels, embeddings, "StreamingHDBSCAN", data_buffer_0)
custom_summary_1_path = "/path/to/your/trigger/results/StabilityCheckHDBSCAN_cluster_summary.json"
metrics_custom_1 = process_and_evaluate(custom_summary_1_path, true_labels, embeddings, "StabilityCheckHDBSCAN", data_buffer_1)
custom_summary_2_path = "/path/to/your/trigger/results/MergeHDBSCAN_cluster_summary.json"
metrics_custom_2 = process_and_evaluate(custom_summary_2_path, true_labels, embeddings, "MergeHDBSCAN", data_buffer_2)
custom_summary_path = "/path/to/your/trigger/results/AnomalyClusteringModule_cluster_summary.json"
metrics_custom = process_and_evaluate(custom_summary_path, true_labels, embeddings, "AnomalyClustering", data_buffer)
metrics_list = [metrics_custom_0, metrics_custom_1, metrics_custom_2, metrics_custom]

cs_std_list = [m['CS-STD_raw'] for m in metrics_list if m['CS-STD_raw'] != -1]
min_val = min(cs_std_list)
max_val = max(cs_std_list)

for m in metrics_list:
    if m['CS-STD_raw'] != -1:
        m['CS-STD'] = (m['CS-STD_raw'] - min_val) / (max_val - min_val + 1e-8)
    else:
        m['CS-STD'] = 0.0

def validate_metrics(*metrics):
    keys = list(metrics[0].keys())
    for i, metric in enumerate(metrics[1:], 1):
        if list(metric.keys()) != keys:
            print(f"[ERROR] Metrics {i} has different keys: {list(metric.keys())}")
            return False
    return True

if not validate_metrics(metrics_custom_0, metrics_custom_1, metrics_custom_2, metrics_custom):
    print("[ERROR] The metrics data are inconsistent.")
    
def plot_radar_chart(metrics1, metrics2, metrics3, metrics4,
                     label1="H-Stream", label2="H-Stab",
                     label3="H-Merge", label4="Ours"):
    labels = [k for k in metrics1.keys() if not k.endswith("_raw")]

    values1 = [metrics1[k] for k in labels]
    values2 = [metrics2[k] for k in labels]
    values3 = [metrics3[k] for k in labels]
    values4 = [metrics4[k] for k in labels]
    values1.append(values1[0])
    values2.append(values2[0])
    values3.append(values3[0])
    values4.append(values4[0])
    labels.append(labels[0])

    angles = np.linspace(0, 2 * np.pi, len(labels) - 1, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values1, linewidth=2, label=label1)
    ax.fill(angles, values1, alpha=0.25)
    ax.plot(angles, values2, linewidth=2, label=label2)
    ax.fill(angles, values2, alpha=0.25)
    ax.plot(angles, values3, linewidth=2, label=label3)
    ax.fill(angles, values3, alpha=0.25)
    ax.plot(angles, values4, linewidth=2, label=label4)
    ax.fill(angles, values4, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=18)
    ax.tick_params(labelsize=18)
    for label in ax.get_yticklabels():
        label.set_fontsize(18)

    ax.set_title("Clustering Evaluation", size=32)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0.35), fontsize=18)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("clustering_radar_comparison.png", dpi=300)
    plt.show()

plot_radar_chart(metrics_custom_0, metrics_custom_1, metrics_custom_2, metrics_custom)
