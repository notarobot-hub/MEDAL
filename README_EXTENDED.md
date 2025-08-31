# MEDAL Extended Framework

This is an extended version of the MEDAL (Model-Agnostic Hallucination Post-Processing Framework) that supports additional datasets and models for question-answering tasks.

## New Features

### Supported Models
- **LLaMA 2 7B**: Meta's LLaMA 2 7B parameter model
- **LLaMA 3.1 8B**: Meta's LLaMA 3.1 8B parameter model  
- **Vicuna 7B**: LMSYS Vicuna 7B model

### Supported Datasets
- **TriviaQA**: Question-answering dataset with trivia questions
- **TruthfulQA**: Dataset focused on truthful question answering
- **CoQA**: Conversational question-answering dataset
- **TyDiQA**: Typologically diverse question-answering dataset

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MEDAL

# Create conda environment
conda create -n medal python=3.8
conda activate medal

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for new models
pip install transformers>=4.30.0
pip install datasets
pip install nltk
pip install rouge
```

## Dataset Preparation

### Prepare Question-Answering Datasets

```bash
# Prepare all QA datasets
python preprocess/prepare_qa_datasets.py --dataset all --output_dir dataset

# Or prepare individual datasets
python preprocess/prepare_qa_datasets.py --dataset triviaqa --output_dir dataset
python preprocess/prepare_qa_datasets.py --dataset truthfulqa --output_dir dataset
python preprocess/prepare_qa_datasets.py --dataset coqa --output_dir dataset
python preprocess/prepare_qa_datasets.py --dataset tydiqa --output_dir dataset
```

## Training and Evaluation

### Training with New Models

```bash
# Train LLaMA 2 7B on TriviaQA
python main.py --config ./config/TriviaQA_LLaMA2_7B.json --do_train

# Train LLaMA 3.1 8B on TruthfulQA
python main.py --config ./config/TruthfulQA_LLaMA3.1_8B.json --do_train

# Train Vicuna 7B on CoQA
python main.py --config ./config/CoQA_Vicuna_7B.json --do_train

# Train LLaMA 2 7B on TyDiQA
python main.py --config ./config/TyDiQA_LLaMA2_7B.json --do_train
```

### Testing with New Models

```bash
# Test LLaMA 2 7B on TriviaQA
python main.py --config ./config/TriviaQA_LLaMA2_7B.json --do_test

# Test LLaMA 3.1 8B on TruthfulQA
python main.py --config ./config/TruthfulQA_LLaMA3.1_8B.json --do_test

# Test Vicuna 7B on CoQA
python main.py --config ./config/CoQA_Vicuna_7B.json --do_test

# Test LLaMA 2 7B on TyDiQA
python main.py --config ./config/TyDiQA_LLaMA2_7B.json --do_test
```

## Evaluation

### Question-Answering Metrics

```bash
cd metrics

# Run QA evaluation
/bin/bash qa_metric_run.sh
```

The QA evaluation includes:
- **Exact Match**: Exact string matching between prediction and reference
- **F1 Score**: Token-level F1 score
- **BLEU Score**: BLEU-4 score for text generation quality
- **ROUGE Scores**: ROUGE-1, ROUGE-2, and ROUGE-L scores

### Customizing Evaluation

You can modify the `qa_metric_run.sh` script to:
- Change dataset names
- Adjust model types
- Modify output directories
- Add custom metrics

## Configuration Files

### New Configuration Files

- `config/TriviaQA_LLaMA2_7B.json`: TriviaQA with LLaMA 2 7B
- `config/TruthfulQA_LLaMA3.1_8B.json`: TruthfulQA with LLaMA 3.1 8B
- `config/CoQA_Vicuna_7B.json`: CoQA with Vicuna 7B
- `config/TyDiQA_LLaMA2_7B.json`: TyDiQA with LLaMA 2 7B

### Configuration Parameters

Key parameters for new models:
- `model_name`: Model identifier (llama2_7b, llama3.1_8b, vicuna_7b)
- `model_path`: HuggingFace model path
- `max_input_length`: Maximum input sequence length (2048 for LLaMA models)
- `max_output_length`: Maximum output sequence length
- `learning_rate`: Learning rate (5e-5 recommended for LLaMA models)
- `train_batch_size`: Training batch size (2 recommended for large models)

## Model Architecture

### Causal Language Model Interface

The extended framework includes a new `CausalLMInterface` class specifically designed for:
- LLaMA models
- Vicuna models
- Other causal language models

Key features:
- Proper handling of input-output concatenation for causal LM training
- Label masking for input tokens during loss computation
- Generation with appropriate stopping criteria
- Padding token handling

### Dataset Classes

New dataset classes for question-answering tasks:
- `TriviaQADataset`: Handles TriviaQA format
- `TruthfulQADataset`: Handles TruthfulQA format
- `CoQADataset`: Handles CoQA format
- `TyDiQADataset`: Handles TyDiQA format

## Usage Examples

### Training Example

```python
# Example training configuration
config = {
    "dataset": "TriviaQA",
    "task": "QuestionAnswering",
    "model_type": "llama2_7b",
    "model_name": "llama2_7b",
    "model_path": "meta-llama/Llama-2-7b-hf",
    "max_epochs": 5,
    "learning_rate": 5e-5,
    "train_batch_size": 2,
    "max_input_length": 2048,
    "max_output_length": 128
}
```

### Evaluation Example

```python
# Example evaluation call
python metrics/qa_cmp_metric.py \
    --pred_fp "output/llama2_7b/llama2_7b/predictions.json" \
    --dec_dir "output/llama2_7b/llama2_7b/dec_dir/" \
    --ref_dir "output/llama2_7b/llama2_7b/ref_dir/" \
    --dataset_name "TriviaQA"
```

## Performance Considerations

### Memory Optimization
- Use gradient checkpointing for large models
- Reduce batch sizes for memory-constrained environments
- Consider using mixed precision training (FP16)

### Training Tips
- Start with smaller learning rates (5e-5) for LLaMA models
- Use early stopping to prevent overfitting
- Monitor validation loss closely
- Consider using learning rate scheduling

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Loading**: Ensure you have access to the model repositories
3. **Tokenization**: Check that padding tokens are properly set
4. **Dataset Format**: Verify dataset format matches expected structure

### Debugging

- Enable CUDA launch blocking: `export CUDA_LAUNCH_BLOCKING=1`
- Check model and tokenizer compatibility
- Verify dataset loading and preprocessing
- Monitor training logs for errors

## Contributing

To extend the framework further:

1. Add new model configurations in `config/`
2. Create dataset classes in `data/dataset.py`
3. Add model interfaces in `model/model_interface.py`
4. Update evaluation scripts in `metrics/`
5. Document new features

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use this extended framework, please cite the original MEDAL paper and this extension:

```bibtex
@article{medal2023,
  title={Better Late Than Never: Model-Agnostic Hallucination Post-Processing Framework Towards Clinical Text Summarization},
  author={...},
  journal={...},
  year={2023}
}
```

## Acknowledgments

- Original MEDAL implementation
- HuggingFace Transformers library
- LLaMA and Vicuna model developers
- Dataset creators and maintainers
