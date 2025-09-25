from datasets import load_dataset, Dataset  
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# 清理 GPU 显存
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === Step 1: 加载数据 ===
print("[Step 1] 读取并加载 GSM8K 数据...")
dataset = load_dataset("json", data_files="/path/to/your/trigger/h_clustered_data.jsonl")["train"]

# 确保数据集不为空
print(f"Total records in dataset: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check your data processing.")

# === Step 2: 初始化模型与配置 ===
print("[Step 2] 初始化模型与配置...")
model_name = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 配置
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()
model.enable_input_require_grads()

# === Step 3: 初始化 Trainer ===
print("[Step 3] 初始化 Trainer...")
sft_config = SFTConfig(
    output_dir="/path/to/your/trigger/baseline_OOD_q/lora-gsm8k-checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=7e-5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    logging_steps=10,
    max_grad_norm=0.5,
    gradient_checkpointing=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    report_to=[]  # 等价于 report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # 使用已加载的 dataset
    #tokenizer=tokenizer,
    peft_config=lora_config,
    args=sft_config
)

# === Step 4: 启动训练 ===
print("[Step 4] 启动训练...")
trainer.train()

# === Step 5: 保存模型 ===
output_dir = "/path/to/your/trigger/baseline_OOD_q/lora-gsm8k-hscan"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"训练完成 ✅ 模型保存至：{output_dir}")
