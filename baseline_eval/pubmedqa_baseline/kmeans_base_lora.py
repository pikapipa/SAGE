import os
import json
import random
import numpy as np
import torch
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# GPU 清理
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 路径配置
dataset_path = "/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet"
embed_model_path = "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5"
base_model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
save_path = "/path/to/your/trigger/baseline_OOD_q/lora-pubmedqa-kmeans"

print("[Step 1] Loading dataset and filtering valid samples...")
dataset = load_dataset("parquet", data_files={"train": dataset_path})["train"]

# 按 final_decision 分类收集，限制样本总数，准备后续采样
samples = defaultdict(list)
max_per_label = 50000  # 适当扩大样本容量，特别是 yes 类

for s in dataset:
    label = s.get("final_decision", "").strip().lower()
    if label in ["yes", "no"]:
        if all(k in s and s[k] for k in ["question", "context", "long_answer"]):
            if len(samples[label]) < max_per_label:
                samples[label].append(s)

print(f"Collected {len(samples['yes'])} yes samples, {len(samples['no'])} no samples")

data_all = samples["yes"] + samples["no"]

print("[Step 2] Generating embeddings for clustering...")
text_samples = [f"Question: {s['question']} Context: {s['context']}" for s in data_all]

embed_model = SentenceTransformer(embed_model_path)
embeddings = embed_model.encode(text_samples, batch_size=64, show_progress_bar=True)

print("[Step 3] Clustering embeddings with KMeans...")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

clustered_data = []
for i, s in enumerate(data_all):
    clustered_data.append({**s, "cluster": int(clusters[i])})

# 按簇收集样本
cluster_buckets = defaultdict(list)
for s in clustered_data:
    cluster_buckets[s["cluster"]].append(s)

# Step 4: 对每个簇内样本，做多样性采样（用文本长度作为简单多样性代理）
def diversity_sampling(samples, max_num):
    # 根据 long_answer 长度排序，均匀采样
    samples_sorted = sorted(samples, key=lambda x: len(x["long_answer"]))
    interval = max(1, len(samples_sorted) // max_num)
    return [samples_sorted[i] for i in range(0, len(samples_sorted), interval)][:max_num]

print("[Step 4] Balanced and diverse sampling within each cluster...")
final_selected = []
max_per_cluster_per_label = 8000  # 限制每簇每标签数量，保证平衡且丰富
for c in range(n_clusters):
    bucket = cluster_buckets[c]
    yes_samples = [s for s in bucket if s["final_decision"].lower() == "yes"]
    no_samples = [s for s in bucket if s["final_decision"].lower() == "no"]
    
    sampled_yes = diversity_sampling(yes_samples, max_per_cluster_per_label)
    sampled_no = diversity_sampling(no_samples, max_per_cluster_per_label)
    
    final_selected.extend(sampled_yes)
    final_selected.extend(sampled_no)

print(f"Final balanced dataset size: {len(final_selected)}")

# Step 5: 划分 train/val/test 三部分，10% val，10% test
train_val, test = train_test_split(final_selected, test_size=0.1, random_state=42, stratify=[x["final_decision"] for x in final_selected])
train, val = train_test_split(train_val, test_size=0.1111, random_state=42, stratify=[x["final_decision"] for x in train_val])  # 约10%

print(f"Split sizes -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Step 6: 转换成多任务训练 prompt+completion 格式
def to_multitask_jsonl(data, path, tokenizer):
    with open(path, "w") as f:
        for s in data:
            prompt = (
                f"Question: {s['question']}\n"
                f"Context: {s['context']}\n"
                f"Answer: "
            )
            # completion 部分包含 long_answer + final_decision 两个任务内容，方便模型学习分辨
            # 生成示例格式：
            # <long_answer>\nFinal decision: yes
            completion_text = f"{s['long_answer'].strip()}\nFinal decision: {s['final_decision'].strip().lower()}{tokenizer.eos_token}"
            f.write(json.dumps({"prompt": prompt, "completion": completion_text}) + "\n")

# Step 7: 加载 tokenizer
print("[Step 7] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

print("[Step 8] Saving processed datasets...")
to_multitask_jsonl(train, "train_multitask.jsonl", tokenizer)
to_multitask_jsonl(val, "val_multitask.jsonl", tokenizer)
to_multitask_jsonl(test, "test_multitask.jsonl", tokenizer)

print("✅ 数据预处理完成，生成多任务训练文件。")

# === Step 9: 初始化模型和 LoRA ===
print("[Step 9] Loading model and applying LoRA...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config) 
model.print_trainable_parameters()
model.train()
model.enable_input_require_grads()

# === Step 10: 加载训练/验证数据集
print("[Step 10] Loading train and validation datasets...")
train_ds = load_dataset("json", data_files="train_multitask.jsonl")["train"]
val_ds = load_dataset("json", data_files="val_multitask.jsonl")["train"]

# === Step 11: 配置 Trainer
print("[Step 11] Configuring Trainer and starting training...")
sft_config = SFTConfig(
    output_dir=save_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    learning_rate=3e-5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    logging_steps=10,
    max_grad_norm=0.5,
    gradient_checkpointing=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_seq_length=1024,
    report_to=[]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=lora_config,
    args=sft_config
)

trainer.train()

# === Step 12: 保存模型
print("[Step 12] Saving fine-tuned model...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"训练完成 ✅ 模型已保存至 {save_path}")
