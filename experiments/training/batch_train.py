#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import time
from pathlib import Path
from experiments.training.train import train_model

def run_batch_training(config_path, no_save=False):
    """Run multiple training configurations based on configuration file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Run each training configuration
    for i, train_config in enumerate(config['trainings'], 1):
        print(f"\n{'='*50}")
        print(f"Running training {i}/{len(config['trainings'])}")
        print(f"{'='*50}")
        print(f"Configuration:")
        print(json.dumps(train_config, indent=2))
        
        # Create output directory
        Path(train_config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Run training
        try:
            train_model(
                model_name=train_config['model_name'],
                dataset_paths=train_config['dataset_paths'],
                output_dir=train_config['output_dir'],
                prompt_type=train_config['prompt_type'],
                use_quantization=train_config.get('use_quantization', True),
                batch_size=train_config.get('batch_size', 2),
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 4),
                learning_rate=train_config.get('learning_rate', 2e-4),
                num_epochs=train_config.get('num_epochs', 4),
                text_column=train_config.get('text_column', 'displayed_text'),
                undersample_majority=train_config.get('undersample_majority', False),
                undersample_seed=train_config.get('undersample_seed', None),
                num_runs=train_config.get('num_runs', 1)
            )
            
            print(f"\nTraining {i} completed successfully")
            
        except Exception as e:
            print(f"\nError in training {i}: {str(e)}")
            continue
    
    print("\nBatch training completed.")

def main():
    parser = argparse.ArgumentParser(description="Run batch training with multiple configurations")
    parser.add_argument("config_path", type=str, help="Path to configuration JSON file")
    parser.add_argument("--no_save", action="store_true", help="Do not save any results")
    args = parser.parse_args()
    
    run_batch_training(args.config_path, args.no_save)

if __name__ == "__main__":
    main() 