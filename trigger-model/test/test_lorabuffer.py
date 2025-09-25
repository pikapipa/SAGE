import torch
import random
import os
import json
from typing import List, Dict, Any
from lora_buffer import LoraOptimizer  
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from utils import load_config, get_device_info, to_device
from datasets import concatenate_datasets, load_dataset
from rouge_score import rouge_scorer
from peft import PeftModel
from sklearn.metrics import classification_report
from bert_score import score as bert_score
from tqdm import tqdm

cfg = load_config("./config.yaml")

print("[INFO] 加载本地 Parquet 数据集...")
pubmed_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/PubMedQA/pqa_artificial/train-00000-of-00001.parquet", split="train")
lexglue_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/lex_glue/case_hold/train-00000-of-00001.parquet", split="train").select(range(1000))
gsm8k_dataset = load_dataset("parquet", data_files="/path/to/your/trigger/datasets/bio/gsm8k/main/train-00000-of-00001.parquet", split="train").select(range(1000))

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

data_all = pubmed_data
data_list = data_all.to_list()
with open("/path/to/your/trigger/output/lora.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)
print("[INFO] 已保存标准化后的初始问答数据到 lora.json")

def extract_decision(text: str) -> str:
    text = text.lower()
    if "yes" in text:
        return "yes"
    else:
        return "no"

def evaluate_medical_qa(samples: List[Dict], save_path: str = None) -> Dict:
    predictions = []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total_rougeL = 0.0

    pred_texts = []
    ref_texts = []

    gold_decisions = []
    pred_decisions = []

    for item in tqdm(samples, desc="Evaluating"):
        pred = item["prediction"].strip()
        ref = item["ground_truth"].strip()
        prompt = item["prompt"]  

        rougeL_f1 = rouge.score(ref, pred)["rougeL"].fmeasure
        total_rougeL += rougeL_f1

        pred_texts.append(pred)
        ref_texts.append(ref)

        pred_dec = extract_decision(pred)
        true_dec = item.get("final_decision", "unknown").lower()

        pred_decisions.append(pred_dec)
        gold_decisions.append(true_dec)

        predictions.append({
            "prompt": prompt,
            "ground_truth": ref,
            "prediction": pred,
            "rougeL_f1": rougeL_f1,
            "predicted_decision": pred_dec,
            "true_decision": true_dec
        })

    avg_rougeL = total_rougeL / len(samples)
    cls_report = classification_report(gold_decisions, pred_decisions, digits=2, output_dict=True)

    results = {
        "avg_rougeL_f1": avg_rougeL,
        "decision_classification_report": cls_report,
        "detailed_predictions": predictions
    }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results

def test_lora_buffer_data_as_cluster():
    model_path = "/path/to/your/trigger/models/llamma2/Llamma-2-7b-ukr-AEG"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    base_model = LlamaForCausalLM.from_pretrained(model_path)

    device = torch.device(get_device_info()["device"])
    base_model.to(device).eval()
    data = load_dataset("json", data_files="/path/to/your/trigger/output/lora.json")["train"]
    data = data.select(range(500))

    cluster = {
        "cluster_id": -1,
        "label": "pubmedqa_cluster",
        "samples": [dict(sample) for sample in data]
    }

    print(f"[INFO] 构造了 1 个虚拟 Cluster，样本数: {len(cluster['samples'])}")
    
    best_config = LoraOptimizer.optimize_lora_for_cluster(cluster, base_model, tokenizer, top_k=3)
    print(f"[INFO] 最优 LoRA 配置: {best_config}")

    for i, entry in enumerate(LoraOptimizer.lora_buffer[:3]):
        checkpoint_path = entry.get("checkpoint_path")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"无效路径，跳过：{checkpoint_path}")
            continue
        print(f"\n=== 评估第 {i+1} 个 LoRA Adapter: {checkpoint_path} ===")
  
        lora_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        lora_model.to(device).eval()

        eval_samples = []
        for sample in data:
            full_input = f"{sample['text']}\n{sample['full_prompt']}"
            inputs = tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = lora_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            eval_samples.append({
                "prompt": full_input,
                "ground_truth": sample["full_prompt"],
                "prediction": pred_text,
                "final_decision": sample.get("real_answer", "unknown")
            })

        result_path = f"/path/to/your/trigger/output/eval_result_adapter_{i}.json"
        results = evaluate_medical_qa(eval_samples, save_path=result_path)

        print(f"\n [{i+1}] Avg ROUGE-L F1: {results['avg_rougeL_f1']:.4f}")
        print(" Final Decision Classification Report:")
        print(json.dumps(results['decision_classification_report'], indent=2))

test_lora_buffer_data_as_cluster()