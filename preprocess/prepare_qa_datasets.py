import json
import os
import argparse
from typing import Dict, List, Any
from datasets import load_dataset

def prepare_triviaqa_data(output_dir: str):
    """Prepare TriviaQA dataset for MEDAL."""
    print("Loading TriviaQA dataset...")
    
    # Load TriviaQA dataset
    dataset = load_dataset("trivia_qa", "rc.nocontext")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    def format_example(example):
        return {
            "id": example["id"],
            "question": example["question"],
            "context": example["entity_pages"]["wiki_context"][0] if example["entity_pages"]["wiki_context"] else "",
            "answer": example["answer"]["aliases"][0] if example["answer"]["aliases"] else "",
            "label": 1 if example["answer"]["aliases"] else 0
        }
    
    # Process train data
    train_data = []
    for example in dataset["train"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["answer"]:
            train_data.append(formatted)
    
    # Process validation data
    val_data = []
    for example in dataset["validation"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["answer"]:
            val_data.append(formatted)
    
    # Process test data
    test_data = []
    for example in dataset["test"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["answer"]:
            test_data.append(formatted)
    
    # Save data
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"TriviaQA: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

def prepare_truthfulqa_data(output_dir: str):
    """Prepare TruthfulQA dataset for MEDAL."""
    print("Loading TruthfulQA dataset...")
    
    # Load TruthfulQA dataset
    dataset = load_dataset("truthful_qa", "generation")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    def format_example(example):
        return {
            "id": example["id"],
            "question": example["question"],
            "answer": example["correct_answers"][0] if example["correct_answers"] else "",
            "label": 1  # TruthfulQA focuses on truthful answers
        }
    
    # Process train data
    train_data = []
    for example in dataset["train"]:
        formatted = format_example(example)
        if formatted["question"] and formatted["answer"]:
            train_data.append(formatted)
    
    # Process validation data
    val_data = []
    for example in dataset["validation"]:
        formatted = format_example(example)
        if formatted["question"] and formatted["answer"]:
            val_data.append(formatted)
    
    # Process test data
    test_data = []
    for example in dataset["test"]:
        formatted = format_example(example)
        if formatted["question"] and formatted["answer"]:
            test_data.append(formatted)
    
    # Save data
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"TruthfulQA: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

def prepare_coqa_data(output_dir: str):
    """Prepare CoQA dataset for MEDAL."""
    print("Loading CoQA dataset...")
    
    # Load CoQA dataset
    dataset = load_dataset("coqa")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    def format_example(example):
        return {
            "id": example["id"],
            "question": example["question"],
            "context": example["story"],
            "answer": example["answer"],
            "label": 1 if example["answer"] else 0
        }
    
    # Process train data
    train_data = []
    for example in dataset["train"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["question"]:
            train_data.append(formatted)
    
    # Process validation data
    val_data = []
    for example in dataset["validation"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["question"]:
            val_data.append(formatted)
    
    # Process test data
    test_data = []
    for example in dataset["test"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["question"]:
            test_data.append(formatted)
    
    # Save data
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"CoQA: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

def prepare_tydiqa_data(output_dir: str):
    """Prepare TyDiQA dataset for MEDAL."""
    print("Loading TyDiQA dataset...")
    
    # Load TyDiQA dataset
    dataset = load_dataset("tydiqa", "primary_task")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    def format_example(example):
        return {
            "id": example["id"],
            "question": example["question_text"],
            "context": example["document_plaintext"],
            "answer": example["annotations"]["minimal_answers"][0]["text"] if example["annotations"]["minimal_answers"] else "",
            "label": 1 if example["annotations"]["minimal_answers"] else 0
        }
    
    # Process train data
    train_data = []
    for example in dataset["train"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["question"]:
            train_data.append(formatted)
    
    # Process validation data
    val_data = []
    for example in dataset["validation"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["question"]:
            val_data.append(formatted)
    
    # Process test data
    test_data = []
    for example in dataset["test"]:
        formatted = format_example(example)
        if formatted["context"] and formatted["question"]:
            test_data.append(formatted)
    
    # Save data
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"TyDiQA: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

def main():
    parser = argparse.ArgumentParser(description="Prepare QA datasets for MEDAL")
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['triviaqa', 'truthfulqa', 'coqa', 'tydiqa', 'all'],
                       help='Dataset to prepare')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for prepared datasets')
    
    args = parser.parse_args()
    
    if args.dataset == 'triviaqa' or args.dataset == 'all':
        prepare_triviaqa_data(os.path.join(args.output_dir, "TriviaQA"))
    
    if args.dataset == 'truthfulqa' or args.dataset == 'all':
        prepare_truthfulqa_data(os.path.join(args.output_dir, "TruthfulQA"))
    
    if args.dataset == 'coqa' or args.dataset == 'all':
        prepare_coqa_data(os.path.join(args.output_dir, "CoQA"))
    
    if args.dataset == 'tydiqa' or args.dataset == 'all':
        prepare_tydiqa_data(os.path.join(args.output_dir, "TyDiQA"))
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()
