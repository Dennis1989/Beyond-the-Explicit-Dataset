# Beyond the Explicit: A Bilingual Dataset for Dehumanization Detection in Social Media

> **Content warning.** This repository contains offensive, hateful, and dehumanizing language, including slurs and examples that target people based on race, ethnicity, gender, sexuality, religion, and other protected characteristics. Materials include lexicons, templates, and LLM-generated examples used for research on **detection and mitigation** of dehumanization. Do not use this content to generate, amplify, or target abuse. Viewer discretion is advised.


**Authors:** Dennis Assenmacher, Paloma Piot, Katarina Laken, David Jurgens, Claudia Wagner

This repository contains scripts and data for training and evaluating models for dehumanization detection.

**Intended use:** research on detecting and studying dehumanizing language in social media. Social-media post text and annotations are not redistributed here; see [https://doi.org/10.7802/3050](https://doi.org/10.7802/3050) for IDs and held-out data (subject to that archive’s terms). This repository is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only; see `LICENSE`).

## Setup

Python 3.10+ recommended. From the repository root:

```bash
pip install -r requirements.txt
```

GPU training/evaluation expects CUDA. `bitsandbytes` is required for 4-bit quantization and `adamw_8bit`. Optional packages for the `misc/` candidate-generation notebooks are listed (commented) in `requirements.txt`.

## Data sharing

Some CSV and TXT files in this repo are **structure-only placeholders** (column headers plus a single dummy row) so the pipeline can be inspected without releasing annotations, social-media posts, or participant demographics. 

The following files under `data/` are **intentionally included in full** and are safe to share publicly. They are the **only non-dummy data files** in this repository:

| File | Description |
|------|-------------|
| `data/targets.txt` | English target-group lexicon used in template generation |
| `data/targets_german.txt` | German target-group lexicon |
| `data/english_dehumanization_templates.csv` | English explicit dehumanization templates (keyword × descriptor combinations) |
| `data/german_dehumanization_templates.csv` | German explicit dehumanization templates |
| `data/Llama3.1_implicit_artificial.csv` | LLM-generated implicit dehumanization instances (English) |
| `data/Llama3.1_explicit_artificial.csv` | LLM-generated explicit dehumanization instances (English) |
| `data/Llama3.1_implicit_artificial_german.csv` | LLM-generated implicit instances (German) |
| `data/Llama3.1_explicit_artificial_german.csv` | LLM-generated explicit instances (German) |

These are synthetic or template-based research materials, not personal data. Social Media Post IDs, their (span-)annotations as well as test splits can be assessed here: https://doi.org/10.7802/3050

Train/test splits, few-shot CSVs, and prediction files under `experiments/splits/` and `experiments/few_shots/` are likewise **placeholders** (schema + dummy row). JSON configs under `experiments/configs/` keep the original experiment structure but use **placeholder paths** for base models (e.g. `experiments/_basemodels/…`), fine-tuned outputs (e.g. `models/…`), and datasets. Point those fields at your local model weights and at the real splits from the GESIS archive before training or evaluation.


## Batch Training

The batch training script allows you to run multiple training configurations in sequence. Each configuration specifies the model, dataset, and training parameters. Configs shipped in this repo are examples with placeholder paths (see [Data sharing](#data-sharing)).

### Configuration File

Create a JSON configuration file (e.g., `experiments/configs/batch_train_config.json`) with the following structure:

```json
{
    "trainings": [
        {
            "model_name": "experiments/_models/llama-3.1-instruct-8b",
            "dataset_paths": ["path/to/mechanistic_training_data.csv"],
            "output_dir": "path/to/output_dir/mechanistic_model",
            "prompt_type": "german_explicit_mechanistic",
            "use_quantization": false,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "num_epochs": 4
        }
    ]
}
```

### Running Batch Training

From the root directory, run:

```bash
python -m experiments.training.batch_train experiments/configs/batch_train_config.json
```

Optional arguments:
- `--no_save`: Do not save any results (default: False)


## BERT and RoBERTa Dehumanization Classifiers

Thes3 scripts fine-tunes a BERT and RoBERTa models for binary classification (e.g., implicit dehumanization detection).

From the root directory, run:

```bash
python -m experiments/training/bert_train.py
```

or

```bash
python -m experiments/training/roberta_train.py
```

### Parameters (set inside `bert_train.py` or `roberta_train.py`)

- `path`: Root folder for logs and model output
- `tokenizer_name`: e.g., `'bert-base-uncased'`
- `model_name`: e.g., `'bert-base-uncased'`
- `output_dir`: Subfolder for saving model checkpoints
- `file_path`: Path to the input CSV
- `label_name`: Column name of the binary label

### Output

- Trained model: `<path>/<output_dir>/model`
- Trainer checkpoint: `<path>/<output_dir>/trainer`
- Training logs: `<path>/logs`


## Batch Evaluation

The batch evaluation script allows you to run multiple model evaluations in sequence. Each configuration specifies the model, dataset, and evaluation parameters.

### Configuration File

Create a JSON configuration file (e.g., `experiments/configs/batch_eval_config.json`) with the following structure:

```json
{
    "evaluations": [
        {
            "name": "test",
            "model_name": "experiments/_models/llama-3.1-instruct-8b",
            "output_dir": "experiments/models/llama-3.1-instruct-8b_lora_finetuned_implicit_animalistic",
            "dataset_path": "experiments/splits/implicit/test_implicit_animalistic_dehumanization.csv",
            "prompt_type": "implicit_animalistic",
            "use_quantization": false,
            "mode": "fine_tuned",
            "text_column": "displayed_text",
            "save_results": true
        }
    ]
}
```

### Running Batch Evaluation

From the root directory, run:

```bash
python -m experiments.evaluation.batch_eval experiments/configs/batch_eval_config.json
```

Optional arguments:
- `--no_save`: Do not save any results (default: False)

## Batch ChatGPT Evaluation

The batch ChatGPT evaluation script allows you to run multiple ChatGPT evaluations in sequence. Each configuration specifies the dataset and evaluation parameters.

### Configuration File

Create a JSON configuration file (e.g., `experiments/configs/batch_chatgpt_eval_config.json`) with the following structure:

```json
{
    "evaluations": [
        {
            "name": "explicit_animalistic",
            "dataset_path": "experiments/splits/explicit/test_explicit_animalistic_dehumanization.csv",
            "prompt_type": "explicit_animalistic",
            "model_name": "gpt-4o",
            "text_column": "displayed_text",
            "save_results": true
        }
    ]
}
```

### Running Batch ChatGPT Evaluation

From the root directory, run:

```bash
python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/batch_chatgpt_eval_config.json
```

Optional arguments:
- `--no_save`: Do not save any results (default: False)

## Configuration Parameters

### Training Parameters
- `model_name`: Name or path of the base model
- `dataset_paths`: List of paths to training datasets
- `output_dir`: Directory to save the fine-tuned model
- `prompt_type`: Type of prompt to use (e.g., "explicit_animalistic", "implicit_mechanistic")
- `use_quantization`: Use 4-bit quantization (default: true)
- `batch_size`: Training batch size (default: 2)
- `gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `num_epochs`: Number of training epochs (default: 4)
- `text_column`: Name of the text column in the dataset (default: "displayed_text")

### Evaluation Parameters
- `model_name`: Name or path of the model to evaluate
- `output_dir`: Directory containing the fine-tuned model
- `dataset_path`: Path to test dataset
- `prompt_type`: Type of prompt to use
- `use_quantization`: Use 4-bit quantization (default: true)
- `mode`: Evaluation mode ("zero_shot" or "fine_tuned")
- `text_column`: Name of the text column in the dataset (default: "displayed_text")
- `save_results`: Whether to save results (default: true)

### ChatGPT Evaluation Parameters
- `dataset_path`: Path to test dataset
- `prompt_type`: Type of prompt to use
- `text_column`: Name of the text column in the dataset (default: "displayed_text")
- `model_name`: GPT model to use (default: "gpt-3.5-turbo")
- `save_results`: Whether to save results (default: true)

## Evaluation Script

This script evaluates classification model predictions against gold labels using accuracy, precision, recall, and F1 (macro average).

From the root directory, run:

```bash
python experiments/evaluation/metrics.py --file_path <path_to_csv> --label <label_column_name>
```

## Notes

- Make sure to set the `OPENAI_API_KEY` environment variable when running ChatGPT evaluations
- The scripts will create output directories automatically if they don't exist
- If an error occurs during batch processing, the script will continue with the next configuration
- Results are saved in the predictions directory by default 