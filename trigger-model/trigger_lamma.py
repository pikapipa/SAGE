import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import time
import os
from utils import load_config, to_device
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

cfg = load_config("./config.yaml")

class LLaMAFeatureExtractor:
    def __init__(self):
        pass

    def get_logits(self, outputs, batch_idx):
        return outputs.logits[batch_idx]

    def get_answer_embedding(self, text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.model.embed_tokens(inputs.input_ids).mean(dim=1)
        return output.squeeze(0).cpu().numpy()

class AnomalyDetector:
    def __init__(self, threshold=None):
        detector_config = cfg.get('detector', {})
        self.ema_alpha = detector_config.get('ema_alpha', 0.1)
        self.ema_score = 0.0
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.weights = detector_config.get('weights', {
            'margin': 0.33,
            'bleu': 0.12,
            'rouge': 0.23,
            'similarity': 0.33
        })
        self.max_margin = detector_config.get('max_margin', 5.0)
        self.trigger_threshold = detector_config.get('trigger_threshold', 0.5)
        self.similarity_id_range = None

    def fit_thresholds(self, id_results, sim_percentile=[5, 95]):
        sim_scores = [r['similarity'] for r in id_results]
        self.similarity_id_range = np.percentile(sim_scores, sim_percentile)

    def logits_margin(self, logits):
        sorted_logits, _ = torch.sort(logits, descending=True)
        margin = sorted_logits[0] - sorted_logits[1] if sorted_logits.shape[-1] > 1 else sorted_logits[0]
        return margin.item()

    def compute_bleu_rouge(self, pred, ref):
        smooth_fn = SmoothingFunction().method1
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
        rouge = self.rouge.score(ref, pred)['rougeL'].fmeasure
        return bleu, rouge

    def compute_embedding_similarity(self, emb1, emb2):
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def compute_anomaly(self, logits, pred_answer, gold_answer, pred_emb, gold_emb, threshold=None):
        margin = self.logits_margin(logits)
        self.ema_score = (1 - self.ema_alpha) * self.ema_score + self.ema_alpha * margin

        if gold_answer.strip():
            bleu, rouge = self.compute_bleu_rouge(pred_answer, gold_answer)
            sim = self.compute_embedding_similarity(pred_emb, gold_emb)
        else:
            bleu, rouge, sim = 1.0, 1.0, 1.0

        w = self.weights
        anomaly_score = (
            w['margin'] * (1 - margin / self.max_margin) +
            w['bleu'] * (1 - bleu) +
            w['rouge'] * (1 - rouge) +
            w['similarity'] * (1 - sim)
        )

        if self.similarity_id_range is not None and len(self.similarity_id_range) == 2:
            low, high = self.similarity_id_range
            if low < sim < high:
                anomaly_score *= 0.5 

        used_threshold = threshold if threshold is not None else self.trigger_threshold
        trigger = anomaly_score >= used_threshold

        return {
            'margin': margin,
            'ema_score': self.ema_score,
            'bleu': bleu,
            'rouge': rouge,
            'similarity': sim,
            'anomaly_score': anomaly_score,
            'trigger': trigger
        }

class TriggerDetector:
    def __init__(self, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            model_cfg = cfg.get('model', {})
            tok_cfg = cfg.get('tokenizer', {})
            device = cfg.get('device', {}).get('device_type', 'cuda' if torch.cuda.is_available() else 'cpu')

            if not model:
                model = LlamaForCausalLM.from_pretrained(
                    model_cfg['pretrained_path'],
                    torch_dtype=torch.float16 if cfg['device'].get('use_amp', True) else torch.float32,
                    device_map=device
                )
                model.config.output_hidden_states = False

            if not tokenizer:
                tokenizer = LlamaTokenizer.from_pretrained(tok_cfg['path'])

        self.model = model
        self.tokenizer = tokenizer
        self.extractor = LLaMAFeatureExtractor()
        self.detector = AnomalyDetector()
        self.hook_handles = []

    def detect(self, prompt, answer, threshold=None):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs, return_dict=True)

        logits = self.extractor.get_logits(output, 0)[-1]
        pred_ids = torch.argmax(output.logits, dim=-1)
        pred_answer = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        pred_emb = self.extractor.get_answer_embedding(pred_answer, self.tokenizer, self.model)
        gold_emb = self.extractor.get_answer_embedding(answer, self.tokenizer, self.model)

        result = self.detector.compute_anomaly(logits, pred_answer, answer, pred_emb, gold_emb, threshold=threshold)
        result.update({
            'prompt': prompt,
            'pred_answer': pred_answer,
            'gold_answer': answer,
            'timestamp': time.time()
        })

        return result
