#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import gc
from transformers import TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training
from experiments.utils.utils import (
    load_model, load_tokenizer, load_dataset_from_paths, apply_lora,
    PROMPTS, SYSTEM_PROMPT, RESPOND_YES_NO
)
import random
import os
from collections import Counter

def print_training_config(
    model_name,
    dataset_paths,
    output_dir,
    prompt_type,
    use_quantization,
    batch_size,
    gradient_accumulation_steps,
    learning_rate,
    num_epochs,
    text_column,
    num_runs
):
    """Print training configuration parameters."""
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    print(f"Model Name: {model_name}")
    print(f"Output Directory: {output_dir}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Dataset Paths: {dataset_paths}")
    print("\nTraining Parameters:")
    print(f"- Use Quantization: {use_quantization}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Number of Epochs: {num_epochs}")
    print(f"- Text Column: {text_column}")
    print(f"- Number of Training Runs: {num_runs}")
    print("\nPrompt:")
    print(f"{PROMPTS[prompt_type]}")
    print("="*50 + "\n")

def tokenize_function(examples, tokenizer, prompt_type, text_column="displayed_text"):
    """Tokenize examples for training."""
    inputs = [
        SYSTEM_PROMPT +
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{PROMPTS[prompt_type]} {RESPOND_YES_NO}"
        f"Sentence: {s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{l}<|eot_id|>"
        for s, l in zip(examples[text_column], examples["label_text"])
    ]

    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

def undersample_majority_class(dataset, label_column="label_text", seed=None):
    """Undersample the majority class to match the minority class size."""
    
    
    label_counts = Counter(dataset[label_column])
    print(label_counts)
    min_count = 500
    # Group indices by class
    indices_by_class = {label: [] for label in label_counts}
    print("seed", seed)
    for idx, label in enumerate(dataset[label_column]):
        indices_by_class[label].append(idx)
    # Sample min_count from each class
    if seed is None:
        seed = random.randint(1, 10000)
        print(f"Using random seed: {seed}")
    random.seed(seed)
    selected_indices = []
    for label, indices in indices_by_class.items():
        if len(indices) > min_count:
            selected = random.sample(indices, min_count)
        else:
            selected = indices
        selected_indices.extend(selected)
    # Shuffle to mix classes
    random.shuffle(selected_indices)
    # Select the rows
    undersampled_dataset = dataset.select(selected_indices)
    print(len(undersampled_dataset))
    return undersampled_dataset

def train_model(
    model_name,
    dataset_paths,
    output_dir,
    prompt_type,
    use_quantization=True,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_epochs=4,
    text_column="displayed_text",
    undersample_majority=False,
    undersample_seed=None,
    num_runs=1
):
    """Train a model on the specified dataset(s) multiple times."""
    # Print training configuration
    print_training_config(
        model_name=model_name,
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        prompt_type=prompt_type,
        use_quantization=use_quantization,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        text_column=text_column,
        num_runs=num_runs
    )

    # Load and prepare dataset once (can be reused for all runs)
    dataset = load_dataset_from_paths(dataset_paths)
   
    
    print(len(dataset))
    # Load tokenizer (can be reused for all runs)
    tokenizer = load_tokenizer(model_name)
    
    # Tokenize dataset once (can be reused for all runs)
    tokenize_fn = lambda x: tokenize_function(x, tokenizer, prompt_type, text_column)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    for run in range(num_runs):
        
        if undersample_majority:
            print(f"Applying undersampling of majority class with seed {undersample_seed}...")
            dataset = undersample_majority_class(dataset, label_column="label_text", seed=undersample_seed)
            tokenize_fn = lambda x: tokenize_function(x, tokenizer, prompt_type, text_column)
            tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
        
        
        print(dataset['displayed_text'][0])
        print(f"\nStarting training run {run + 1}/{num_runs}")
        run_output_dir = os.path.join(output_dir, f"_run_{run + 1}")
        os.makedirs(run_output_dir, exist_ok=True)

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load model and prepare for training
        model = load_model(model_name, use_quantization=use_quantization)
        if use_quantization:
            print("Preparing model for kbit training")
            model = prepare_model_for_kbit_training(model)
        model = apply_lora(model)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=30,
            bf16=True,
            num_train_epochs=num_epochs,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            save_total_limit=1,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        # Train model
        trainer.train()

        # Save model
        model.save_pretrained(run_output_dir)
        tokenizer.save_pretrained(run_output_dir)
        print(f"Fine-tuned model for run {run + 1} saved to {run_output_dir}")

        # Clean up to free memory
        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Train a model for dehumanization detection")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the base model")
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True, help="Path(s) to training dataset(s)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--prompt_type", type=str, required=True, choices=list(PROMPTS.keys()), help="Type of prompt to use")
    parser.add_argument("--use_quantization", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--text_column", type=str, default="displayed_text", help="Name of the text column in the dataset")
    parser.add_argument("--undersample_majority", action="store_true", help="Undersample the majority class")
    parser.add_argument("--undersample_seed", type=int, default=None, help="Random seed for undersampling majority class")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of training runs with different random initializations")
    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        dataset_paths=args.dataset_paths,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
        use_quantization=args.use_quantization,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        text_column=args.text_column,
        undersample_majority=args.undersample_majority,
        undersample_seed=args.undersample_seed,
        num_runs=args.num_runs
    )

if __name__ == "__main__":
    main() 