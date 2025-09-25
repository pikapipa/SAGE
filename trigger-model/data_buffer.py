import torch 
import numpy as np
import json
import random
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Set, Tuple, Union, Optional
from transformers import AutoModel, AutoTokenizer
import hdbscan
from tqdm import tqdm
from scipy.special import softmax
from utils import load_config, save_json, load_json, to_device, tensor_to_list
from trigger_lamma import TriggerDetector
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
import logging

CONFIG = load_config("./config.yaml")

class AnomalyClusteringModule:
    def __init__(self, 
                llm_model, 
                llm_tokenizer,
                embedding_model: str = None,
                buffer_path: Optional[str] = None,
                similarity_threshold: Union[float, str] = "auto", 
                cluster_size_threshold: int = 20,
                initial_data: Optional[List[Dict[str, Any]]] = None,
                enable_merge: bool = False,
                enable_stability_check: bool = False):
        self.enable_merge = enable_merge
        self.enable_stability_check = enable_stability_check
        self.llm = llm_model
        self.llm_tokenizer = llm_tokenizer
        embedding_model = embedding_model or CONFIG.get('embedding_model_path', "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5")
        self.buffer_path = buffer_path or CONFIG.get('buffer', {}).get('anomaly_cluster_path', "/path/to/your/trigger/data/clusters/clusters.json")       
        self.similarity_threshold = similarity_threshold
        self.cluster_size_threshold = cluster_size_threshold      
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.embedding_model = AutoModel.from_pretrained(embedding_model, local_files_only=True).to(self.device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model, local_files_only=True)
        self.clusters = self.load_buffer() or []
        self.cluster_keywords = {}
        self.cluster_id_counter = 0
        self.structure_cluster_map = defaultdict(list)
        self.bucket_buffers = defaultdict(list)
        self.bucket_stable_flags = defaultdict(lambda: False)
        self.bucket_cluster_centers = defaultdict(list)
        self.bucket_cluster_keywords = defaultdict(list)
        self.trigger_detector = None    
        if not hasattr(self, 'bucket_buffer_history'):
            self.bucket_buffer_history = defaultdict(list)
        if self.similarity_threshold == "auto":
            self.similarity_threshold = self.estimate_similarity_threshold()
            logging.info(f"[AutoThreshold] adaptive threshold：{self.similarity_threshold:.4f}")   
        for cluster in self.clusters:
            cid = cluster["id"]
            keywords = set()
            for sample in cluster.get("samples", []):
                kws = self.extract_keywords(sample.get("prompt_text", ""))
                keywords.update(kws)
            self.cluster_keywords[cid] = keywords
            self.cluster_id_counter = max(self.cluster_id_counter, cid + 1)
        self.trigger_detector = None    
        if len(self.clusters) == 0 and initial_data is not None and len(initial_data) >= 10:
            if len(set(s["prompt_text"] for s in initial_data)) < self.cluster_size_threshold * 2:
                logging.warning("[Warn] start sample is too big, do not warm-start")
            else:
                self.clusters = []
                self.cluster_keywords = {}
                self.structure_cluster_map = defaultdict(list)
                self.structure_bucket_clustering(initial_data[:50])

    def estimate_similarity_threshold(self, max_pairs=1000) -> float:
        embeddings = []
        for cluster in self.clusters:
            for sample in cluster["samples"]:
                emb = sample["embedding"]
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                embeddings.append(emb)
        if len(embeddings) < 2:
            logging.warning("[Warn] 样本太少，无法估计相似度阈值，默认使用 0.45")
            return 0.45
        similarities = []
        try:
            pairs = random.sample([(i, j) for i in range(len(embeddings)) for j in range(i+1, len(embeddings))], 
                                min(max_pairs, len(embeddings)*(len(embeddings)-1)//2))
            for i, j in pairs:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
            threshold = np.percentile(similarities, 30)
            return threshold
        except Exception as e:
            logging.error(f"估计阈值失败，使用默认0.45：{e}")
            return 0.45

    def load_buffer(self) -> List[Dict[str, Any]]:
        try:
            return load_json(self.buffer_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"加载聚类缓冲文件失败：{e}")
            return []

    def save_buffer(self) -> None:
        try:
            save_json(tensor_to_list(self.clusters), self.buffer_path)
        except Exception as e:
            logging.error(f"保存聚类缓冲文件失败：{e}")

    def extract_keywords(self, text_or_dict: Union[str, Dict[str, Any]]) -> Set[str]:
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 
                 'and', 'or', 'is', 'are', 'this', 'that', 'it', 'as'}
        if isinstance(text_or_dict, dict):
            text = text_or_dict.get('text', '')
            if 'full_prompt' in text_or_dict:
                if isinstance(text_or_dict['full_prompt'], dict):
                    contexts = text_or_dict['full_prompt'].get('contexts', [])
                    text += ' ' + ' '.join(str(ctx) for ctx in contexts)
                else:
                    text += ' ' + str(text_or_dict['full_prompt'])
        else:
            text = str(text_or_dict)

        words = text.lower().split()
        return {
            word for word in words 
            if word not in stopwords 
            and len(word) > 3
            and word.isalpha()  
        }

    def infer_structure_and_domain(self, sample: Dict[str, Any]) -> str: 
        source = sample.get("source", "").strip().lower()
        return source if source else "unknown"

    def get_embedding(self, sample: Dict[str, Any]) -> torch.Tensor:
        json_str = json.dumps(sample, ensure_ascii=False)
        chunks = [json_str[i:i+5000] for i in range(0, len(json_str), 5000)]
        chunk_embeddings = []    
        for chunk in chunks:
            inputs = self.embedding_tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)    
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                chunk_embeddings.append(outputs.last_hidden_state[:, 0, :])     
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)  
 
    def create_new_cluster(self, label: Optional[str] = None) -> int:
        cid = self.cluster_id_counter
        self.cluster_id_counter += 1
        label = label or f"anomaly_cluster_{cid}"
        new_cluster = {"id": cid, "label": label, "samples": []}
        self.clusters.append(new_cluster)
        return cid

    def add_sample_to_cluster(self, cluster_idx: int, sample: Dict[str, Any]) -> None:
        self.clusters[cluster_idx]["samples"].append(sample)
        if "prompt_text" in sample:
            keywords = self.extract_keywords(sample["prompt_text"])
            if cluster_idx not in self.cluster_keywords:
                self.cluster_keywords[cluster_idx] = keywords
            else:
                self.cluster_keywords[cluster_idx].update(keywords)

    def process_anomaly_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        structure_tag = self.infer_structure_and_domain(sample)
        embedding = sample.get("embedding", None)
        if "embedding" in sample:
            embedding = sample["embedding"]
        else:
            embedding = self.get_embedding(sample)
        logits = sample.get("logits", None)
        keywords = self.extract_keywords(sample)

        if self.bucket_stable_flags[structure_tag]:
            cluster_idx, score = self.find_best_cluster(structure_tag, embedding, keywords)
            if cluster_idx != -1 and score >= self.similarity_threshold:
                return self.finalize_sample_assignment(cluster_idx, sample, embedding, logits)
        
        self.bucket_buffers[structure_tag].append({
            "raw_sample": sample,
            "embedding": embedding.clone(),
            "keywords": keywords,
            "logits": logits
        })
        print(f"[Buffer] 样本放入结构桶 {structure_tag} 的 buffer，当前缓存数：{len(self.bucket_buffers[structure_tag])}")

        BUFFER_CLUSTER_TRIGGER = 25
        if len(self.bucket_buffers[structure_tag]) >= BUFFER_CLUSTER_TRIGGER:
            self._cluster_within_bucket(structure_tag)

        return {
            **sample,
            "embedding": embedding.clone(),
            "is_anomaly": True,
            "is_generated": False,
            "logits": logits,
            "cluster_id": -1,
            "cluster_label": "Unassigned"
        }

    def cluster_all_buckets(self) -> None:
        for structure_tag in list(self.bucket_buffers.keys()):
            try:
                self._cluster_within_bucket(structure_tag)
            except Exception as e:
                logging.error(f"聚类桶 {structure_tag} 失败: {e}")

    def _cluster_within_bucket(self, structure_tag):
        buffer = self.bucket_buffers[structure_tag]
        if len(buffer) < 25:
            return

        embeddings = torch.stack([s["embedding"] for s in buffer])
        texts = [s["raw_sample"] for s in buffer]
        logits_list = [s["logits"] for s in buffer]
        keywords_list = [s["keywords"] for s in buffer]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        labels = clusterer.fit_predict(embeddings.cpu().numpy())
        valid_labels = [label for label in labels if label >= 0]
        if len(set(valid_labels)) <= 1:
            return
        label2embs = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:
                label2embs[label].append(embeddings[idx])
        current_centers = []
        for label, embs in label2embs.items():
            current_centers.append(torch.stack(embs).mean(dim=0))

        stable_flag = False
        if hasattr(self, 'last_labels') and structure_tag in self.last_labels:
            prev_labels = self.last_labels[structure_tag]

            min_len = min(len(prev_labels), len(labels))
            prev = prev_labels[:min_len]
            curr = labels[:min_len]
            ari = adjusted_rand_score(prev, curr) 
            print(f"[ClusterStability] 桶 {structure_tag} ARI: {ari:.3f}")
        else:
            ari = 0

        if hasattr(self, 'last_centers') and structure_tag in self.last_centers and len(self.last_centers[structure_tag]) == len(current_centers):
            prev_centers = self.last_centers[structure_tag]
            cos_sims = []
            for pc, cc in zip(prev_centers, current_centers):
                cos_sim = torch.nn.functional.cosine_similarity(pc.to(self.device), cc.to(self.device), dim=0).item()
                cos_sims.append(cos_sim)
            avg_cos_sim = sum(cos_sims) / len(cos_sims)
            print(f"[ClusterStability] 桶 {structure_tag} 簇中心平均cosine相似度: {avg_cos_sim:.3f}")
        else:
            avg_cos_sim = 0
        ARI_THRESHOLD = 0.8
        CENTER_SIM_THRESHOLD = 0.5

        if ari >= ARI_THRESHOLD and avg_cos_sim >= CENTER_SIM_THRESHOLD:
            stable_flag = True
        else:
            stable_flag = False

        if not hasattr(self, 'last_labels'):
            self.last_labels = {}
        if not hasattr(self, 'last_centers'):
            self.last_centers = {}

        self.last_labels[structure_tag] = labels
        self.last_centers[structure_tag] = current_centers

        if not stable_flag:
            print(f"[ClusterInit] 桶 {structure_tag} 聚类不稳定，等待更多样本或下一次聚类")
            return

        label2samples = defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:  
                continue
            label2samples[label].append(i)

        if structure_tag not in self.structure_cluster_map:
            self.structure_cluster_map[structure_tag] = []

        for label, indices in label2samples.items():
            next_cluster_id = len(self.clusters)
            cluster_label = f"{structure_tag}_cluster_{next_cluster_id}"
            cid = self.create_new_cluster(cluster_label)
            if cid not in self.structure_cluster_map[structure_tag]:
                self.structure_cluster_map[structure_tag].append(cid)
            c_embs, c_keywords = [], set()
            for idx in indices:
                sample = {
                    "prompt_text": texts[idx],
                    "embedding": buffer[idx]["embedding"],
                    "is_anomaly": True,
                    "logits": logits_list[idx],
                    "cluster_id": cid,
                    "cluster_label": self.clusters[cid]["label"]
                }
                self.add_sample_to_cluster(cid, sample)
                c_embs.append(buffer[idx]["embedding"])
                c_keywords.update(keywords_list[idx])
            self.bucket_cluster_centers[structure_tag].append(torch.stack(c_embs).mean(dim=0))
            self.bucket_cluster_keywords[structure_tag].append(c_keywords)

        if self.enable_stability_check:
            self.bucket_stable_flags[structure_tag] = True
            print(f"[ClusterInit] 桶 {structure_tag} 聚类稳定，标记完成")
        else:
            print(f"[ClusterInit] 桶 {structure_tag} 聚类完成（未启用稳定性标记）")

        self.bucket_buffers[structure_tag] = []

        if self.enable_merge and len(self.structure_cluster_map[structure_tag]) >= 3:
            self.merge_similar_clusters(structure_tag)

    def find_best_cluster(self, structure_tag: str, embedding: torch.Tensor, keywords: Set[str]) -> Tuple[int, float]:
        best_score = -1
        best_idx = -1
        for i, center in enumerate(self.bucket_cluster_centers[structure_tag]):
            cos_sim = torch.nn.functional.cosine_similarity(embedding, center.to(self.device), dim=0).item()
            kw_overlap = len(keywords & self.bucket_cluster_keywords[structure_tag][i]) / max(len(self.bucket_cluster_keywords[structure_tag][i]), 1)
            score = 0.7 * cos_sim + 0.3 * kw_overlap
            if score > best_score:
                best_score = score
                best_idx = i
        cid = self.structure_cluster_map[structure_tag][best_idx] if best_idx >= 0 else -1
        return cid, best_score

    def finalize_sample_assignment(self, cluster_idx: int, text: str, embedding: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
        sample = {
            "prompt_text": text,
            "embedding": embedding.clone(),
            "is_anomaly": True,
            "is_generated": False,
            "logits": logits,
            "cluster_id": cluster_idx,
            "cluster_label": self.clusters[cluster_idx]["label"]
        }
        self.add_sample_to_cluster(cluster_idx, sample)
        logging.info(f"[StreamCluster] 样本成功归属至已有 cluster {cluster_idx}")
        return sample
    
    def merge_similar_clusters(self, structure_tag: str, threshold: float = 0.8):
        if structure_tag not in self.bucket_cluster_centers:
            return
        centers = self.bucket_cluster_centers[structure_tag]
        keywords = self.bucket_cluster_keywords[structure_tag]
        cluster_ids = self.structure_cluster_map[structure_tag]        
        id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}       
        merge_pairs = []        
        for i, cid_i in enumerate(cluster_ids):
            for j, cid_j in enumerate(cluster_ids[i+1:], start=i+1):
                cos_sim = torch.nn.functional.cosine_similarity(
                    centers[i].to(self.device),
                    centers[j].to(self.device),
                    dim=0
                ).item()               
                if cos_sim >= threshold:
                    merge_pairs.append((cid_i, cid_j))
                    print(f"[Merge] 相似度 {cos_sim:.4f}，合并 {cid_j} -> {cid_i}")
        merged_cids = set()
        for target_cid, source_cid in merge_pairs:
            if source_cid in merged_cids:
                continue              
            target_idx = id_to_idx[target_cid]
            source_idx = id_to_idx[source_cid]           
            for sample in self.clusters[source_cid]["samples"]:
                sample["cluster_id"] = target_cid
                sample["cluster_label"] = self.clusters[target_cid]["label"]
                self.clusters[target_cid]["samples"].append(sample)            
            keywords[target_idx].update(keywords[source_idx])
            merged_cids.add(source_cid)
        if merged_cids:
            new_cluster_ids = [
                cid for cid in cluster_ids 
                if cid not in merged_cids  
            ]
            
            new_centers = []
            new_keywords = []
            for cid in new_cluster_ids:
                if self.clusters[cid]["samples"]:
                    new_embeddings = [s["embedding"] for s in self.clusters[cid]["samples"]]
                    new_centers.append(torch.stack(new_embeddings).mean(dim=0))
                else:
                    new_centers.append(self.bucket_cluster_centers[structure_tag][
                        cluster_ids.index(cid)
                    ])
                new_keywords.append(
                    self.bucket_cluster_keywords[structure_tag][
                        cluster_ids.index(cid)
                    ]
                )
            
            self.structure_cluster_map[structure_tag] = new_cluster_ids
            self.bucket_cluster_centers[structure_tag] = new_centers
            self.bucket_cluster_keywords[structure_tag] = new_keywords
            

            for cid in merged_cids:
                if cid in self.clusters: 
                    del self.clusters[cid]

    def _validate_cluster_integrity(self, structure_tag):
        cluster_ids = self.structure_cluster_map.get(structure_tag, [])
        for cid in cluster_ids:
            if cid not in self.clusters:
                raise ValueError(f"Cluster {cid} 在structure_cluster_map中但不存在于clusters")
        if len(cluster_ids) != len(self.bucket_cluster_centers.get(structure_tag, [])):
            raise ValueError("cluster_ids与centers数量不匹配")
        if len(cluster_ids) != len(self.bucket_cluster_keywords.get(structure_tag, [])):
            raise ValueError("cluster_ids与keywords数量不匹配")
    