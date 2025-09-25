import os
import math
import re
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
base_model = LlamaForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device).eval()

train_data_path = "/path/to/your/trigger/trigger-model/gsm8k_level1.jsonl"  
train_data = load_dataset("json", data_files=train_data_path)["train"]
test_data = load_dataset("json", data_files=train_data_path)["train"].shuffle(seed=42).select(range(50))

formatted_samples = []
for example in train_data:
    question = example.get("question", "").strip().replace("\n", " ")
    answer = example.get("answer", "").strip()
    if not question or not answer:
        continue

    match = re.search(r"####\s*(-?\d+)", answer)
    if not match:
        continue
    label = match.group(1)

    prompt = f"[Question]\n{question}\n\nLet's think step by step."
    completion = f" [Answer]\n{answer}\nThe final answer is: {label}"
    formatted_samples.append({"prompt": prompt, "completion": completion, "label": label})

print(f"总样本数: {len(formatted_samples)}")

train_path = "math_train.jsonl"
with open(train_path, "w") as f:
    for sample in formatted_samples:
        f.write(json.dumps(sample) + "\n")

def extract_numeric_answer(output_text: str, label: str, max_val=1e6):
    match = re.search(r"final answer\s*is\s*:?\s*(-?\d+\.?\d*)", output_text.lower())
    if not match:
        lines = output_text.strip().split("\n")
        for line in reversed(lines[-5:]):
            match = re.search(r"(-?\d+\.?\d*)", line)
            if match:
                break
    if match:
        try:
            val = float(match.group(1))
            if math.isfinite(val) and abs(val) < max_val:
                return int(val), int(label)
        except:
            return None
    return None

def train_lora_model(base_model, tokenizer, train_samples, lora_config, output_dir):
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    model.train()

    def preprocess(sample):
        text = sample["prompt"] + "\n" + sample["completion"]
        tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = train_samples.map(preprocess)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        learning_rate=lora_config["learning_rate"],
        num_train_epochs=5,
        logging_steps=10,
        report_to="none",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized
    )
    trainer.train()
    return model

def evaluate_model(model, test_samples, tokenizer, device):
    y_true, y_pred = [], []
    model.eval()

    for sample in tqdm(test_samples, desc="Evaluating"):
        prompt = f"{sample['prompt']}\n{sample['completion']}"
        label = sample.get("label")
        if label is None:
            continue
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        result = extract_numeric_answer(output_text, label)
        if result:
            pred, true = result
            y_pred.append(pred)
            y_true.append(true)

    if y_true:
        acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    else:
        acc = 0.0
    return acc

r_list = [2, 4, 6, 8, 10, 12]  # Rank
lr_list = np.geomspace(1e-6, 5e-4, 6)  # Learning Rate
acc_matrix = np.zeros((len(r_list), len(lr_list)))
ce_loss_matrix = np.zeros((len(r_list), len(lr_list)))

for i, r in enumerate(r_list):
    for j, lr in enumerate(lr_list):
        print(f"\n🔧 正在训练: r={r}, lr={lr:.1e}")
        lora_config = {"r": r, "learning_rate": lr}
        model = train_lora_model(base_model, tokenizer, train_data, lora_config, output_dir=f"./tmp_r{r}_lr{lr:.0e}")
        acc = evaluate_model(model, test_data, tokenizer, device)
        acc_matrix[i, j] = acc

        model.eval()
        losses = []
        for sample in random.sample(list(train_data), min(50, len(train_data))):
            text = sample["prompt"] + "\n" + sample["completion"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                losses.append(loss)
        ce_loss_matrix[i, j] = np.mean(losses)

def draw_heatmaps(acc_matrix, ce_matrix, lr_list, r_list):
    plt.figure(figsize=(14, 6))
    lr_labels = ["{:.1e}".format(x) for x in lr_list]

    plt.subplot(1, 2, 1)
    sns.heatmap(acc_matrix, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=lr_labels, yticklabels=r_list)
    plt.title("Accuracy Heatmap")
    plt.xlabel("Learning Rate")
    plt.ylabel("LoRA Rank (r)")

    plt.subplot(1, 2, 2)
    sns.heatmap(ce_loss_matrix, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=lr_labels, yticklabels=r_list)
    plt.title("Cross-Entropy Loss Heatmap")
    plt.xlabel("Learning Rate")
    plt.ylabel("LoRA Rank (r)")

    plt.tight_layout()
    plt.savefig("heatmap_acc_ce_loss.png")
    plt.show()

draw_heatmaps(acc_matrix, ce_loss_matrix, lr_list, r_list)
