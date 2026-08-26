#!/bin/bash
PYTHON_BIN="${PYTHON_BIN:-python}"

$PYTHON_BIN -m experiments.evaluation.chatgpt_eval \
    --dataset_path "experiments/splits/explicit/test_explicit_animalistic_dehumanization.csv" \
    --prompt_type "explicit_animalistic" \
    --model_name "gpt-4o" \
    --text_column "displayed_text"

$PYTHON_BIN -m experiments.evaluation.chatgpt_eval \
    --dataset_path "experiments/splits/explicit/test_implicit_animalistic_dehumanization.csv" \
    --prompt_type "implicit_animalistic" \
    --model_name "gpt-4o" \
    --text_column "displayed_text"

$PYTHON_BIN -m experiments.evaluation.chatgpt_eval \
    --dataset_path "experiments/splits/explicit/test_explicit_mechanistic_dehumanization.csv" \
    --prompt_type "explicit_mechanistic" \
    --model_name "gpt-4o" \
    --text_column "displayed_text"

$PYTHON_BIN -m experiments.evaluation.chatgpt_eval \
    --dataset_path "experiments/splits/explicit/test_implicit_mechanistic_dehumanization.csv" \
    --prompt_type "implicit_mechanistic" \
    --model_name "gpt-4o" \
    --text_column "displayed_text"
