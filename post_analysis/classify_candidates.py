#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os
from pathlib import Path

# Verzeichnis eine Ebene über der aktuellen Datei zum PYTHONPATH hinzufügen
PARENT = Path(__file__).resolve().parent.parent
sys.path.append(str(PARENT))

from experiments.utils.utils import (
    load_model, load_tokenizer, classify_sentence_logits, PROMPTS
)


import pandas as pd
from tqdm import tqdm

def classify_dataset(
    dataset_path,
    model_name,
    output_dir,
    prompt_type,
    text_column="text",
    use_quantization=True,
    fine_tuned=True,
    max_length=200
):
    """Classify all texts in a dataset and save predictions with probabilities."""
    annotated_path = dataset_path.replace(".csv", "_annotated.csv")
    
    if os.path.exists(annotated_path):
        print(f"\nAnnotated dataset already exists. Loading from {annotated_path}")
        data = pd.read_csv(annotated_path)
        #return data
    else:
        print(f"\nLoading dataset from {dataset_path}")
        data = pd.read_csv(dataset_path)
        
    # Load model and tokenizer
    model = load_model(model_name, output_dir, use_quantization, fine_tuned)
    tokenizer = load_tokenizer(model_name)
    
    # Filter by text length if specified
    if max_length:
        data = data[data[text_column].str.len() <= max_length]
    
    # Run inference with progress bar
    print("\nRunning inference...")
    results = []
    for text in tqdm(data[text_column], desc="Classifying texts"):
        result = classify_sentence_logits(text, model, tokenizer, prompt_type)
        results.append(result)
    
    # Add predictions and probabilities to dataset
    model_name_short = model_name.split("/")[-1]
    data[f"{model_name_short}_{prompt_type}"] = [1 if x["yes_logit"] > x["no_logit"] else 0 for x in results]
    data[f"{model_name_short}_{prompt_type}_yes_probs"] = [x["yes_prob"] for x in results]
    
    # Save results
    data.to_csv(annotated_path, index=False)
    print(f"\nResults saved to {annotated_path}")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Total texts: {len(data)}")
    print(f"Positive predictions: {data[f'{model_name_short}_{prompt_type}'].sum()}")
    print(f"Negative predictions: {len(data) - data[f'{model_name_short}_{prompt_type}'].sum()}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Classify texts in a dataset using a language model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the fine-tuned model")
    parser.add_argument("--prompt_type", type=str, required=True, choices=list(PROMPTS.keys()), help="Type of prompt to use")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the dataset")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum text length to process")
    parser.add_argument("--no_quantization", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero-shot model instead of fine-tuned")

    args = parser.parse_args()

    classify_dataset(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
        text_column=args.text_column,
        use_quantization=not args.no_quantization,
        fine_tuned=not args.zero_shot,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main() 