# === 已修复并适配 GSM8K 数据格式的 LoRA + KMeans 微调流程 ===
import os 
import json
import random
import numpy as np
import torch
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# 路径配置
parquet_path = "/path/to/your/trigger/datasets/bio/gsm8k/main/train-00000-of-00001.parquet"
embed_model_path = "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5"
base_model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
save_path = "/path/to/your/trigger/baseline_OOD_q/lora-gsm8k-kmeans"

print("[Step 1] 加载 GSM8K 数据集...")
dataset = load_dataset("parquet", data_files={"train": parquet_path})["train"]

formatted_samples = []
for example in dataset:
    question = example.get("question", "").strip().replace("\n", " ")
    answer = example.get("answer", "").strip()
    if not question or not answer:
        continue
    prompt = f"[Question]\n{question}\n\n[Answer]"
    completion = f" {answer}"
    formatted_samples.append({"prompt": prompt, "completion": completion})

print(f"[INFO] 样本数量: {len(formatted_samples)}")

print("[Step 2] 生成嵌入向量...")
embed_model = SentenceTransformer(embed_model_path)
text_samples = [s["prompt"] for s in formatted_samples]
embeddings = embed_model.encode(text_samples, batch_size=64, show_progress_bar=True)

print("[Step 3] 进行 KMeans 聚类...")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

for i, s in enumerate(formatted_samples):
    s["cluster"] = int(clusters[i])

print("[Step 4] 每个聚类中进行均匀抽样...")
cluster_buckets = defaultdict(list)
for s in formatted_samples:
    cluster_buckets[s["cluster"]].append(s)

final_selected = []
max_per_cluster = 1000
for bucket in cluster_buckets.values():
    sampled = random.sample(bucket, min(max_per_cluster, len(bucket)))
    final_selected.extend(sampled)

print(f"[INFO] 聚类后样本总数: {len(final_selected)}")

def to_jsonl(data, path):
    with open(path, "w") as f:
        for s in data:
            f.write(json.dumps(s) + "\n")

print("[Step 5] 保存 jsonl 文件...")
to_jsonl(train, "train_gsm8k.jsonl")

print("[Step 6] 加载模型与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token
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

print("[Step 8] 加载训练数据集...")
train_ds = load_dataset("json", data_files="train_gsm8k.jsonl")["train"]

def preprocess(example):
    full = example["prompt"] + example["completion"]
    tokenized = tokenizer(full, truncation=True, max_length=1024, padding="max_length")
    labels = tokenized["input_ids"].copy()
    prompt_len = len(tokenizer(example["prompt"])['input_ids'])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)

print("[Step 9] 配置训练参数并启动训练...")
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
    args=sft_config,
    tokenizer=tokenizer
)

trainer.train()

print("[Step 10] 保存模型...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 训练完成，模型保存至: {save_path}")