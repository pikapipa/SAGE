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
import json
from typing import Dict, Union
from data_buffer import AnomalyClusteringModule
from transformers import AutoTokenizer, AutoModel
from utils import load_json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, f1_score
import pandas as pd
from scipy.spatial.distance import cosine
from PIL import Image

pubmed_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet", split="train").select(range(1000))
lexglue_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet", split="train").select(range(1000))
gsm8k_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/gsm8k/main/train-00000-of-00001.parquet", split="train").select(range(1000))

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
data_list = data_all.to_list()

with open("/path/to/your/trigger/output/test_ood_m.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)

def eval_clustering(embeddings: np.ndarray,
                    labels_true: list, labels_pred: list) -> dict:
    ari  = adjusted_rand_score(labels_true, labels_pred)
    nmi  = normalized_mutual_info_score(labels_true, labels_pred)
    sil  = silhouette_score(embeddings, labels_pred) \
           if len(set(labels_pred)) > 1 and len(labels_pred) > len(set(labels_pred)) else np.nan
    f1   = f1_score(labels_true, labels_pred, average="macro")
    return dict(ARI=ari, NMI=nmi, Silhouette=sil, F1=f1)

def save_metrics(metrics_list, csv_path):
    df = pd.DataFrame(metrics_list) 
    df.to_csv(csv_path, index=False)
    print(f"[INFO] 已保存聚类指标到 {csv_path}")

def stream_windows(dataset, size=200):  
    for i in range(0, len(dataset), size):
        yield dataset[i:i+size]

def cluster_stability(prev_centers: dict, curr_centers: dict):
    if not prev_centers:      
        return np.nan, np.nan
    sims = []
    for cid, c_vec in curr_centers.items():
        best_sim = max(1 - cosine(c_vec, p_vec)
                       for p_vec in prev_centers.values())
        sims.append(best_sim)
    return np.mean(sims), np.std(sims)

def build_data_buffer(tokenizer, model, initial_data, enhanced=True):
    return AnomalyClusteringModule(
        llm_model=model,
        llm_tokenizer=tokenizer,
        buffer_path=None,
        similarity_threshold="auto",
        cluster_size_threshold=20,
        initial_data=initial_data,
        enable_merge=enhanced,           
        enable_stability_check=enhanced      
    )

def merge_tsne_snapshots(tsne_paths, save_path, layout='horizontal'):
    images = [Image.open(p) for p in tsne_paths]
    widths, heights = zip(*(img.size for img in images))

    if layout == 'horizontal':
        total_width = sum(widths)
        max_height = max(heights)
        merged_img = Image.new('RGB', (total_width, max_height), color='white')

        x_offset = 0
        for img in images:
            merged_img.paste(img, (x_offset, 0))
            x_offset += img.width

    elif layout == 'vertical':
        max_width = max(widths)
        total_height = sum(heights)
        merged_img = Image.new('RGB', (max_width, total_height), color='white')

        y_offset = 0
        for img in images:
            merged_img.paste(img, (0, y_offset))
            y_offset += img.height

    else:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    merged_img.save(save_path)
    print(f"[INFO] 合并后的 t-SNE 图已保存为: {save_path}")

def test_data_buffer_with_ood_dataset():
    save_dir = "/path/to/your/trigger/results"
    os.makedirs(save_dir, exist_ok=True)

    print("\n[INFO] 初始化 DataBuffer ...")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG")
    model = AutoModel.from_pretrained("/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG").eval().cuda()
    
    initial_data = load_json("/path/to/your/trigger/output/test_ood_m.json")
    data_buffer = AnomalyClusteringModule(
        llm_model=model,
        llm_tokenizer=tokenizer,
        buffer_path="/path/to/your/trigger/output/test_ood_buffer.json",
        similarity_threshold="auto",
        cluster_size_threshold=20,
        initial_data=initial_data
    )

    print(f"[INFO] 初始结构感知分桶数量：{len(data_buffer.clusters)}")
    from datasets import load_dataset
    ood_data = load_dataset("json", data_files="/path/to/your/trigger/output/test_ood_m.json")["train"]
    print("[INFO] 执行启发式动态聚类...")
    vocab_size = tokenizer.vocab_size
    for example in tqdm(ood_data):
        data_buffer.process_anomaly_sample(example)
    buffer = [i for i, c in enumerate(data_buffer.clusters) if len(c["samples"]) >= data_buffer.cluster_size_threshold]
    print(f"[INFO] 可用于微调的聚类数量：{len(buffer)}")

    summary = {
        "total_buckets": len(data_buffer.structure_cluster_map),
        "buckets": []
    }
    print("\n[BUCKET & CLUSTER SUMMARY]")
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

        summary_path = os.path.join(save_dir, "bucket_cluster_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 聚类结构已保存到：{summary_path}")

    print("\n[INTRA-CLUSTER DISTANCE]")
    for cid in buffer:
        cluster = data_buffer.clusters[cid]
        samples = cluster["samples"]
        if len(samples) < 2:
            print(f"Cluster {cid}: 样本过少，跳过")
            continue
        embeddings = np.stack([
            s["embedding"].detach().cpu().numpy() if isinstance(s["embedding"], torch.Tensor)
            else np.array(s["embedding"])
            for s in samples
        ])
        center = np.mean(embeddings, axis=0)
        avg_dist = np.mean(cosine_distances(embeddings, [center]))
        print(f"Cluster {cid}: Avg Intra Distance = {avg_dist:.4f}")

    print("\n[INTER-CLUSTER DISTANCE]")
    centers = {
        cid: np.mean(np.stack([
            s["embedding"].detach().cpu().numpy() if isinstance(s["embedding"], torch.Tensor)
            else np.array(s["embedding"])
            for s in data_buffer.clusters[cid]["samples"]
        ]), axis=0)
        for cid in buffer
    }
    cids = list(centers.keys())
    total_dist, count = 0, 0
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            dist = cosine_distances([centers[cids[i]]], [centers[cids[j]]])[0][0]
            print(f"Cluster {cids[i]} vs {cids[j]}: {dist:.4f}")
            total_dist += dist
            count += 1
    if count > 0:
        print(f"Avg Inter-Cluster Distance = {total_dist / count:.4f}")
    else:
        print("[WARN] 仅有一个聚类")

    print("\n[TSNE VISUALIZATION]")
    all_embeddings, all_labels = [], []
    for structure_tag, cluster_ids in data_buffer.structure_cluster_map.items():  
        for cid in cluster_ids:
            cluster = data_buffer.clusters[cid]
            for s in cluster["samples"]:
                emb = s["embedding"]
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                all_embeddings.append(emb)
                all_labels.append(f"{structure_tag}_{cid}")  
    if len(all_embeddings) >= 2:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(all_embeddings)//3),  
            random_state=42
        )
        tsne_embeds = tsne.fit_transform(np.stack(all_embeddings))
        
        plt.figure(figsize=(10, 8))
        unique_labels = sorted(set(all_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label in unique_labels:
            indices = [i for i, l in enumerate(all_labels) if l == label]
            plt.scatter(
                tsne_embeds[indices, 0], tsne_embeds[indices, 1],
                label=label,
                alpha=0.7,
                s=40
            )
        
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.title("t-SNE by Bucket & Cluster")
        plt.tight_layout()
        
        tsne_path = os.path.join(save_dir, "cluster_tsne_m.png")
        plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] 统一视图t-SNE已保存为 {tsne_path}")
    else:
        print("[WARN] 样本不足，无法进行t-SNE")

def run_all_experiments():
    model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
    save_dir = "/path/to/your/trigger/results"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval().cuda()

    print("[INFO] 加载标准化后的数据...")
    data = load_json("/path/to/your/trigger/output/test_ood_m.json")

    for i, d in enumerate(data):
        d["uid"] = f"sample_{i}"

    labels_true = [d["source"] for d in data]  
    embeddings_all = []
    labels_pred_baseline = []
    labels_pred_enhanced = []

    metrics_baseline = []
    metrics_enhanced = []
    tsne_frames = []

    buffer_base = build_data_buffer(tokenizer, model, initial_data=[], enhanced=False)
    buffer_enh = build_data_buffer(tokenizer, model, initial_data=[], enhanced=True)

    prev_centers_base, prev_centers_enh = {}, {}
    stability_records = []

    for step, window in enumerate(stream_windows(data, size=200), 1):
        print(f"\n[INFO] 处理第 {step} 个窗口，共 {len(window)} 条样本")

        for sample in window:
            buffer_base.process_anomaly_sample(sample)
            buffer_enh.process_anomaly_sample(sample)

        for sample in window:
            emb = sample["embedding"].detach().cpu().numpy() if isinstance(sample["embedding"], torch.Tensor) else np.array(sample["embedding"])
            embeddings_all.append(emb)
            uid = sample["uid"]
            cid_b = sample["cluster_id_base"] = buffer_base.sample2cluster.get(uid, -1)
            cid_e = sample["cluster_id_enh"] = buffer_enh.sample2cluster.get(uid, -1)

            labels_pred_baseline.append(cid_b)
            labels_pred_enhanced.append(cid_e)

        m_base = eval_clustering(embeddings_all, labels_true, labels_pred_baseline)
        m_enh = eval_clustering(embeddings_all, labels_true, labels_pred_enhanced)
        m_base.update(window_id=step, method="baseline"); metrics_baseline.append(m_base)
        m_enh.update(window_id=step, method="enhanced"); metrics_enhanced.append(m_enh)
        curr_centers_base = buffer_base.get_all_centers()
        curr_centers_enh = buffer_enh.get_all_centers()
        mean_sim_b, _ = cluster_stability(prev_centers_base, curr_centers_base)
        mean_sim_e, _ = cluster_stability(prev_centers_enh, curr_centers_enh)
        stability_records.append(
            dict(window=step, mean_center_sim_base=mean_sim_b, mean_center_sim_enh=mean_sim_e)
        )
        prev_centers_base, prev_centers_enh = curr_centers_base, curr_centers_enh
        tsne_png = f"{save_dir}/tsne_w{step:03d}.png"
        all_embeds_now = embeddings_all[-len(window):]
        all_labels_now = [f"B{cid}" for cid in labels_pred_baseline[-len(window):]]
        tsne_snapshot(all_embeds_now, all_labels_now, tsne_png)
        tsne_frames.append(tsne_png)

    save_metrics(metrics_baseline + metrics_enhanced, f"{save_dir}/cluster_scores.csv")
    pd.DataFrame(stability_records).to_csv(f"{save_dir}/stability.csv", index=False)

    make_tsne_gif(tsne_frames, f"{save_dir}/tsne_stream.gif", fps=2)

    selected_frames = [tsne_frames[i] for i in [0, 2, 4, 6, 9] if i < len(tsne_frames)]
    merge_tsne_snapshots(selected_frames, f"{save_dir}/tsne_evolution_summary.png", layout="horizontal")
    
test_data_buffer_with_ood_dataset()
run_all_experiments()