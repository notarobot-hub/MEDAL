#!/bin/bash

# run command is /bin/bash qa_metric_run.sh

export DATASET_NAME="TriviaQA"
export TASK_MODEL_NAME="llama2_7b"
export MODEL_TYPE="llama2_7b"
export EVAL_FILENAME=""

# Exact Match + F1 Score
python qa_cmp_metric.py \
  --pred_fp "output/$TASK_MODEL_NAME/$MODEL_TYPE/$EVAL_FILENAME" \
  --dec_dir "output/$TASK_MODEL_NAME/$MODEL_TYPE/dec_dir/" \
  --ref_dir "output/$TASK_MODEL_NAME/$MODEL_TYPE/ref_dir/" \
  --dataset_name $DATASET_NAME

# BERTScore
python -m bert_score.score \
  --cand_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/dec_dir/" \
  --ref_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/ref_dir/" \
  --lang en \
  --model_type "microsoft/DialoGPT-medium" \
  --output_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/bertscore_results.txt"

# BLEU Score
python -m nltk.translate.bleu_score \
  --cand_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/dec_dir/" \
  --ref_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/ref_dir/" \
  --output_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/bleu_results.txt"

# ROUGE Score
python -m rouge_score.rouge_scorer \
  --cand_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/dec_dir/" \
  --ref_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/ref_dir/" \
  --output_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/rouge_results.txt"
