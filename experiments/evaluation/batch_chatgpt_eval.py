#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from experiments.evaluation.chatgpt_eval import evaluate_chatgpt

def run_batch_evaluation(config_path, no_save=False):
    """Run multiple ChatGPT evaluations based on configuration file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Run each evaluation
    for i, eval_config in enumerate(config['evaluations'], 1):
        print(f"\n{'='*50}")
        print(f"Running evaluation {i}/{len(config['evaluations'])}")
        print(f"{'='*50}")
        print(f"Configuration:")
        print(json.dumps(eval_config, indent=2))
        
        # Run evaluation
        try:
            results = evaluate_chatgpt(
                dataset_path=eval_config['dataset_path'],
                prompt_type=eval_config['prompt_type'],
                text_column=eval_config.get('text_column', 'displayed_text'),
                save_results=not no_save and eval_config.get('save_results', True),
                model_name=eval_config.get('model_name', 'gpt-3.5-turbo'),
                save_path=eval_config.get('save_path', None),
                runs=eval_config.get('runs', 5),
                train_dataset_path=eval_config.get('train_dataset_path', None),
                num_few_shot_examples=eval_config.get('num_few_shot_examples', 20)
            )
            
            print(f"\nEvaluation {i} completed successfully")
            print("\nResults:")
            print(json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"\nError in evaluation {i}: {str(e)}")
            continue
    
    print("\nBatch evaluation completed.")

def main():
    parser = argparse.ArgumentParser(description="Run batch ChatGPT evaluation with multiple configurations")
    parser.add_argument("config_path", type=str, help="Path to configuration JSON file")
    parser.add_argument("--no_save", action="store_true", help="Do not save any results")
    args = parser.parse_args()
    
    run_batch_evaluation(args.config_path, args.no_save)

if __name__ == "__main__":
    main() 