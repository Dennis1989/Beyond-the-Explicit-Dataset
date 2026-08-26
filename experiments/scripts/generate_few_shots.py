#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import pandas as pd
import os
from experiments.utils.utils import load_dataset_from_paths

def generate_few_shots(
    train_dataset_path,
    text_column,
    label_column,
    dehum_type,
    file_name=None,
    num_examples=20,
    num_sets=5,
    output_dir="experiments/few_shots",
    random_state=42
):
    """
    Generate multiple sets of few-shot examples with stratified sampling.
    
    Args:
        train_dataset_path: Path to training dataset
        text_column: Name of the text column
        label_column: Name of the label column
        dehum_type: Type of dehumanization (e.g., explicit, implicit)
        file_name: Custom file name prefix (optional)
        num_examples: Number of examples per set
        num_sets: Number of different sets to generate
        output_dir: Directory to save the few-shot sets
        random_state: Initial random state (will be incremented for each set)
    """
    # Load training data
    print(f"\nLoading training data from {train_dataset_path}...")
    train_dataset = load_dataset_from_paths(train_dataset_path, shuffle=False)
    train_df = pd.DataFrame(train_dataset)
    
    # Create output directory if it doesn't exist
    dehum_output_dir = os.path.join(output_dir, dehum_type)
    os.makedirs(dehum_output_dir, exist_ok=True)
    
    # Generate multiple sets
    for set_idx in range(num_sets):
        current_random_state = random_state + set_idx
        print(f"\nGenerating set {set_idx + 1}/{num_sets} with random_state={current_random_state}")
        
        # Get label distribution
        label_counts = train_df[label_column].value_counts()
        n_per_class = num_examples // len(label_counts)
        remainder = num_examples % len(label_counts)
        
        # Sample from each class
        stratified_samples = []
        for label in label_counts.index:
            n_samples = n_per_class + (1 if remainder > 0 else 0)
            remainder -= 1
            class_samples = train_df[train_df[label_column] == label].sample(
                n=n_samples,
                random_state=current_random_state
            )
            stratified_samples.append(class_samples)
        
        # Combine samples and shuffle
        few_shot_data = pd.concat(stratified_samples).sample(
            frac=1,
            random_state=current_random_state
        )
        
        # Generate file name
        if file_name:
            base_name = f"{file_name}_set_{set_idx + 1}"
        else:
            base_name = f"{dehum_type}_few_shot_set_{set_idx + 1}"
        
        output_path = os.path.join(dehum_output_dir, f"{base_name}_{num_examples}_examples.csv")
        few_shot_data.to_csv(output_path, index=False)
        print(f"Saved few-shot set to {output_path}")
        
        # Print examples
        print(f"\nExamples from set {set_idx + 1}:")
        for i, (_, row) in enumerate(few_shot_data.iterrows(), 1):
            print(f"\nExample {i}:")
            print(f"Text: {row[text_column]}")
            print(f"Label: {row[label_column]}")
            if i >= 5:  # Only print first 5 examples
                print("...")
                break
        
        # Print label distribution
        print(f"\nLabel distribution in set {set_idx + 1}:")
        print(few_shot_data[label_column].value_counts())

def main():
    parser = argparse.ArgumentParser(description="Generate few-shot example sets")
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--text_column", type=str, default="displayed_text", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="label_text", help="Name of the label column")
    parser.add_argument("--dehum_type", type=str, required=True, help="Type of dehumanization (e.g., explicit, implicit)")
    parser.add_argument("--file_name", type=str, help="Custom file name prefix (optional)")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of examples per set")
    parser.add_argument("--num_sets", type=int, default=5, help="Number of different sets to generate")
    parser.add_argument("--output_dir", type=str, default="experiments/few_shots", help="Directory to save the few-shot sets")
    parser.add_argument("--random_state", type=int, default=42, help="Initial random state")
    
    args = parser.parse_args()
    
    generate_few_shots(
        train_dataset_path=args.train_dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        dehum_type=args.dehum_type,
        file_name=args.file_name,
        num_examples=args.num_examples,
        num_sets=args.num_sets,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main() 