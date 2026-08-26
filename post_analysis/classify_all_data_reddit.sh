#!/bin/bash

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="classify_candidates.py"
DATASET="../data/reddit_final_harmonized.csv"
MODEL="../experiments/_models/llama-3.1-instruct-8b"
TEXT_COL="text"
MAX_LEN=200

$PYTHON_BIN $SCRIPT \
    --dataset_path "$DATASET" \
    --model_name "$MODEL" \
    --output_dir "../experiments/models/llama-3.1-instruct-8b_lora_finetuned_explicit_animalistic" \
    --prompt_type "explicit_animalistic" \
    --text_column "$TEXT_COL" \
    --max_length $MAX_LEN\
    --no_quantization

$PYTHON_BIN $SCRIPT \
    --dataset_path "$DATASET" \
    --model_name "$MODEL" \
    --output_dir "../experiments/models/llama-3.1-instruct-8b_lora_finetuned_explicit_mechanistic" \
    --prompt_type "explicit_mechanistic" \
    --text_column "$TEXT_COL" \
    --max_length $MAX_LEN\
    --no_quantization

$PYTHON_BIN $SCRIPT \
    --dataset_path "$DATASET" \
    --model_name "$MODEL" \
    --output_dir "../experiments/models/llama-3.1-instruct-8b_lora_finetuned_implicit_animalistic" \
    --prompt_type "implicit_animalistic" \
    --text_column "$TEXT_COL" \
    --max_length $MAX_LEN\
    --no_quantization

$PYTHON_BIN $SCRIPT \
    --dataset_path "$DATASET" \
    --model_name "$MODEL" \
    --output_dir "../experiments/models/llama-3.1-instruct-8b_lora_finetuned_implicit_mechanistic" \
    --prompt_type "implicit_mechanistic" \
    --text_column "$TEXT_COL" \
    --max_length $MAX_LEN\
    --no_quantization