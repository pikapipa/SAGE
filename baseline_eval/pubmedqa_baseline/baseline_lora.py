from datasets import load_dataset, Dataset 
import json
from collections import defaultdict
import random
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# 清理 GPU 显实
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === Step 1: 平衡样本 & 转换格式 ===
print("[Step 1] 读取 parquet 文件并构造平衡训练集...")
dataset = load_dataset("parquet", data_files={"train": "/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet"})
samples = defaultdict(list)

# 聚合不同标签样本
target_size = 30000  # 每类上限
for s in dataset["train"]:
    label = s["final_decision"].strip().lower()
    if label in ["yes", "no"] and len(samples[label]) < target_size:
        samples[label].append(s)

# 拼接平衡数据并打乱
data = samples["yes"] + samples["no"] 
random.shuffle(data)

# 转为 JSONL 格式，加入结构化输出提示
output_path = "pubmedqa_balanced_lora_train.jsonl"
with open(output_path, "w") as f:
    for s in data:
        prompt = f"Question: {s['question']}\nContext: {s['context']}\nAnswer: {s['long_answer']}\nFinal decision:"
        completion = s["final_decision"].lower()
        f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
print(f"写入完成，共 {len(data)} 条样本 -> {output_path}")

# === Step 2: LoRA 微调配置 ===
print("[Step 2] 初始化模型与配置...")
model_name = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()
model.enable_input_require_grads()
'''
# 加载训练数据
print("[Step 3] 加载训练数据并初始化 Trainer...")
dataset = load_dataset("json", data_files=output_path)
dataset = dataset["train"]

training_args = TrainingArguments(
    output_dir="/path/to/your/trigger/baseline_OOD_q/lora-pubmedqa-checkpoints",
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
    report_to="none"
)
'''
# === Step 3: 加载训练数据并初始化 Trainer ===
print("[Step 3] 加载训练数据并初始化 Trainer...")
dataset = load_dataset("json", data_files=output_path)["train"]

sft_config = SFTConfig(
    output_dir="/path/to/your/trigger/baseline_OOD_q/lora-pubmedqa-checkpoints",
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

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #tokenizer=tokenizer,
    peft_config=lora_config,
    args=sft_config  
)

# === Step 4: 启动训练 ===
print("[Step 4] 启动训练...")
trainer.train()

# === Step 5: 保存模型 ===
output_dir = "/path/to/your/trigger/baseline_OOD_q/lora-pubmedqa-final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("\u8bad\u7ec3\u5b8c\u6210 \u2705 \u6a21\u578b\u4fdd\u5b58\u81f3：", output_dir)
