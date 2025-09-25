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

# GPU 清理
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 路径配置
dataset_path = "/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet"
embed_model_path = "/path/to/your/trigger/models/emb-bge/bge-large-en-v1.5"
base_model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
save_path = "/path/to/your/trigger/baseline_OOD_q/lora-lexglue-kmeans"

print("[Step 1] Loading dataset and formatting...")
dataset = load_dataset("parquet", data_files={"train": "/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet"})

label_groups = defaultdict(list)
for s in dataset["train"]:
    if isinstance(s["label"], int) and 0 <= s["label"] < len(s["endings"]):
        context = s["context"].strip().replace("\n", " ")
        options = s["endings"]
        
        # 改进的prompt格式
        prompt = f"Given the following legal context and options, choose the most appropriate option.\n\n"
        prompt += f"Context: {context}\n\n"
        prompt += "Options:\n"
        for idx, opt in enumerate(options):
            prompt += f"{idx}. {opt}\n"
        prompt += "\nAnswer:"
        
        sample = {"prompt": prompt, "completion": str(s["label"])}
        label_groups[s["label"]].append(sample)

print("[Step 2] Balancing labels via downsampling...")
min_count = min(len(samples) for samples in label_groups.values())
balanced_samples = []
for label, samples in label_groups.items():
    sampled = random.sample(samples, min_count) if len(samples) > min_count else samples
    balanced_samples.extend(sampled)
random.shuffle(balanced_samples)

print("[Step 3] Embedding generation for clustering...")
embed_model = SentenceTransformer(embed_model_path)
text_samples = [s["prompt"] for s in balanced_samples]
embeddings = embed_model.encode(text_samples, batch_size=64, show_progress_bar=True)

print("[Step 4] KMeans clustering...")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)
for i, s in enumerate(balanced_samples):
    s["cluster"] = int(clusters[i])

print("[Step 5] Per-cluster label balancing and sampling...")

cluster_buckets = defaultdict(list)
for s in balanced_samples:
    cluster_buckets[s["cluster"]].append(s)

final_selected = []
max_per_label = 800
for cluster_id, bucket in cluster_buckets.items():
    label_buckets = defaultdict(list)
    for s in bucket:
        label_buckets[s["completion"]].append(s)
    min_label_count = min(len(samples) for samples in label_buckets.values())
    per_label_sample_count = min(max_per_label, min_label_count)
    for label, samples in label_buckets.items():
        sampled = random.sample(samples, per_label_sample_count) if len(samples) > per_label_sample_count else samples
        final_selected.extend(sampled)

print(f"[Step 6] Final dataset size after cluster-label balance: {len(final_selected)}")

train_val, test = train_test_split(final_selected, test_size=0.1, random_state=42, stratify=[x["completion"] for x in final_selected])
train, val = train_test_split(train_val, test_size=0.1111, random_state=42, stratify=[x["completion"] for x in train_val])

print(f"Split sizes -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

def to_multitask_jsonl(data, path, tokenizer):
    with open(path, "w") as f:
        for s in data:
            prompt = s["prompt"]
            completion_text = f"Final decision: {s['completion'].strip()}{tokenizer.eos_token}"
            f.write(json.dumps({"prompt": prompt, "completion": completion_text}) + "\n")


print("[Step 7] Loading tokenizer and saving JSONL datasets...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

to_multitask_jsonl(train, "train_multitask.jsonl", tokenizer)
to_multitask_jsonl(val, "val_multitask.jsonl", tokenizer)
to_multitask_jsonl(test, "test_multitask.jsonl", tokenizer)

print("[Step 8] Loading model and applying LoRA...")
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

print("[Step 9] Training LoRA fine-tuning with SFTTrainer...")
train_ds = load_dataset("json", data_files="train_multitask.jsonl")["train"]
val_ds = load_dataset("json", data_files="val_multitask.jsonl")["train"]

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

print("[Step 10] Saving fine-tuned model...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ LoRA 微调完成，模型已保存至 {save_path}")