import argparse
import json
import os
import re
import string
from collections import Counter
from typing import List, Dict, Any

import numpy as np
from nltk.tokenize import word_tokenize
from rouge import Rouge

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_qa_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate QA-specific metrics."""
    exact_matches = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        exact_matches.append(exact_match_score(pred, ref))
        f1_scores.append(f1_score(pred, ref))
    
    return {
        "exact_match": np.mean(exact_matches),
        "f1_score": np.mean(f1_scores)
    }

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(predictions, references, avg=True)
        return {
            "rouge1": scores["rouge-1"]["f"],
            "rouge2": scores["rouge-2"]["f"],
            "rougeL": scores["rouge-l"]["f"]
        }
    except:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothing = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing))
        
        return np.mean(bleu_scores)
    except:
        return 0.0

def evaluate_qa_dataset(pred_file: str, dec_dir: str, ref_dir: str, dataset_name: str) -> Dict[str, Any]:
    """Evaluate QA dataset with multiple metrics."""
    
    # Load predictions
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    predictions = []
    references = []
    
    for item in pred_data:
        pred = item.get('generated_summary_sentences', '')
        ref = item.get('original_summary_sentences', '')
        
        if isinstance(pred, list):
            pred = ' '.join(pred)
        if isinstance(ref, list):
            ref = ' '.join(ref)
        
        predictions.append(pred.strip())
        references.append(ref.strip())
    
    # Calculate metrics
    qa_metrics = calculate_qa_metrics(predictions, references)
    rouge_metrics = calculate_rouge_scores(predictions, references)
    bleu_score = calculate_bleu_score(predictions, references)
    
    # Combine all metrics
    results = {
        "dataset": dataset_name,
        "qa_metrics": qa_metrics,
        "rouge_metrics": rouge_metrics,
        "bleu_score": bleu_score,
        "sample_size": len(predictions)
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA datasets")
    parser.add_argument('--pred_fp', type=str, required=True, help='Prediction file path')
    parser.add_argument('--dec_dir', type=str, required=True, help='Decoded directory path')
    parser.add_argument('--ref_dir', type=str, required=True, help='Reference directory path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    
    args = parser.parse_args()
    
    # Evaluate the dataset
    results = evaluate_qa_dataset(args.pred_fp, args.dec_dir, args.ref_dir, args.dataset_name)
    
    # Save results
    output_file = args.pred_fp.replace('.json', '_qa_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print results
    print(f"=== QA Evaluation Results for {args.dataset_name} ===")
    print(f"Sample Size: {results['sample_size']}")
    print(f"Exact Match: {results['qa_metrics']['exact_match']:.4f}")
    print(f"F1 Score: {results['qa_metrics']['f1_score']:.4f}")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print(f"ROUGE-1: {results['rouge_metrics']['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge_metrics']['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rouge_metrics']['rougeL']:.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
