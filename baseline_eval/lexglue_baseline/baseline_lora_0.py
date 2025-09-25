from datasets import load_dataset, Dataset
import json
import random
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from collections import defaultdict, Counter

# === Step 1: 读取数据并进行数据平衡处理 ===
print("[Step 1] 读取 parquet 文件并进行数据平衡处理...")
dataset = load_dataset("parquet", data_files={"train": "/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet"})

# 按标签分组收集样本
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

# 数据平衡处理 - 使用下采样
print("原始数据分布:")
for label, samples in label_groups.items():
    print(f"  类别 {label}: {len(samples)} 样本")

# 找到最少的类别数量
min_count = min(len(samples) for samples in label_groups.values())
print(f"最少类别样本数: {min_count}")

# 对每个类别进行下采样
balanced_samples = []
for label, samples in label_groups.items():
    if len(samples) > min_count:
        sampled = random.sample(samples, min_count)
    else:
        sampled = samples
    balanced_samples.extend(sampled)

# 打乱数据
random.shuffle(balanced_samples)

print(f"平衡后总样本数: {len(balanced_samples)}")
print("平衡后数据分布:")
balanced_labels = [int(sample["completion"]) for sample in balanced_samples]
balanced_counts = Counter(balanced_labels)
for label, count in sorted(balanced_counts.items()):
    print(f"  类别 {label}: {count} 样本")

# 保存平衡后的数据
output_path = "casehold_lora_train_balanced.jsonl"
with open(output_path, "w") as f:
    for sample in balanced_samples:
        f.write(json.dumps(sample) + "\n")

print(f"平衡后数据保存至: {output_path}")

# === Step 2: 创建训练验证集分割 ===
print("[Step 2] 创建训练验证集分割...")
val_ratio = 0.1
val_size = int(len(balanced_samples) * val_ratio)
val_samples = balanced_samples[:val_size]
train_samples = balanced_samples[val_size:]

print(f"训练集样本数: {len(train_samples)}")
print(f"验证集样本数: {len(val_samples)}")

# 保存训练集和验证集
train_path = "casehold_train.jsonl"
val_path = "casehold_val.jsonl"

with open(train_path, "w") as f:
    for sample in train_samples:
        f.write(json.dumps(sample) + "\n")

with open(val_path, "w") as f:
    for sample in val_samples:
        f.write(json.dumps(sample) + "\n")

# === Step 3: 初始化模型与改进的LoRA配置 ===
print("[Step 3] 初始化模型与LoRA配置...")
model_name = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 改进的LoRA配置
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    lora_dropout=0.05,  
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()
model.enable_input_require_grads()

# === Step 4: 加载数据集 ===
print("[Step 4] 加载训练和验证数据...")
train_dataset = load_dataset("json", data_files=train_path)["train"]
val_dataset = load_dataset("json", data_files=val_path)["train"]

# === Step 5: 改进的训练配置 ===
print("[Step 5] 配置训练参数...")
sft_config = SFTConfig(
    output_dir="/path/to/your/trigger/baseline_OOD_q/lora-lexglue-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
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
    report_to=[]  # 等价于 report_to="none"
)

# === Step 6: 初始化Trainer ===
print("[Step 6] 初始化 Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 添加验证集
    peft_config=lora_config,
    args=sft_config,
    #tokenizer=tokenizer
)

# === Step 7: 启动训练 ===
print("[Step 7] 启动训练...")
trainer.train()

# === Step 8: 保存模型 ===
print("[Step 8] 保存模型...")
output_dir = "/path/to/your/trigger/baseline_OOD_q/lora-lexglue-final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("训练完成 ✅")
print(f"模型保存至: {output_dir}")
print(f"最佳模型checkpoints: {sft_config.output_dir}")

# === 训练后的数据统计信息 ===
print("\n=== 训练数据统计 ===")
print(f"训练集大小: {len(train_samples)}")
print(f"验证集大小: {len(val_samples)}")
print("各类别样本数:")
for label, count in sorted(balanced_counts.items()):
    print(f"  类别 {label}: {count} 样本")