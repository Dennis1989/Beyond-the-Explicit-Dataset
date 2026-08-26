#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import torch
import gc
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from experiments.utils.utils import (
    load_model, load_tokenizer, classify_sentence,
    PROMPTS, load_dataset_from_paths
)

def get_few_shot_examples(train_dataset_path, text_column, label_column, run, prompt_type, num_few_shot_examples):
    """Load few-shot examples from training data with stratified sampling to ensure balanced label distribution."""
    #print(f"\nLoading {num_examples} few-shot examples from training data...")
    train_dataset = load_dataset_from_paths(train_dataset_path+"/"+str(prompt_type)+"_few_shot_set_"+str(run)+"_"+str(num_few_shot_examples)+"_examples.csv", shuffle=False)
    train_df = pd.DataFrame(train_dataset)
    
    # Combine samples and shuffle
    few_shot_data = train_df[[text_column, label_column]].values.tolist()
    
    # Print examples
    print("\nFew-shot examples:")
    for i, (text, label) in enumerate(few_shot_data, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        print(f"Label: {label}")
    
    return few_shot_data

def evaluate_model(
    model_name,
    output_dir,
    dataset_path,
    save_path,
    prompt_type,
    use_quantization=True,
    few_shots=None,
    text_column="displayed_text",
    mode="zero_shot",
    save_results=True,
    train_dataset_path=None,
    num_few_shot_examples=20,
    cross_validation=False,
    run = 1
):
    """Evaluate a model on the specified dataset."""
    # Load model and tokenizer
    print(train_dataset_path)
    print(few_shots)
    
    
    
    model = load_model(model_name, output_dir, use_quantization, fine_tuned=(mode != "zero_shot"), run=run)
    # set model to eval mode
    model.eval()
    
    tokenizer = load_tokenizer(model_name)

    # Load few-shot examples if training data is provided
    if train_dataset_path and few_shots is None:
        few_shots = get_few_shot_examples(
            train_dataset_path,
            text_column,
            "label_text",
            run,
            prompt_type,
            num_few_shot_examples
        )

    # Load test dataset
    test_df = pd.read_csv(dataset_path)
    test_df["label_text"] = test_df["label_text"].astype(str).str.lower()
    
    
    # Run inference
    predictions = test_df.apply(
        lambda row: classify_sentence(
            row[text_column], 
            model, 
            tokenizer, 
            prompt_type, 
            few_shots, 
            row["label_text"]  # <-- das Label mitgeben
        ),
        axis=1
    )
    print(predictions)
    print("end_predictions")
    # Adjust predictions to ensure they are either "yes" or "no"
    predictions = ["yes" if x.lower() != "no" else x.lower() for x in predictions]
    print(predictions)

    # Save predictions if requested
    if save_results:
        try:
            
            prediction_path = os.path.join(save_path+prompt_type+"_"+"predictions.csv")
            
            if os.path.exists(prediction_path):
                eval_df = pd.read_csv(prediction_path)
            else:
                eval_df = test_df.copy()
                
            if train_dataset_path is  None:
                eval_df[model_name.split("/")[-1]+"_"+mode+"_r"+str(run)] = [1 if x.lower() != "no" else 0 for x in predictions]
            
            else:
                eval_df[model_name.split("/")[-1]+"_"+str(num_few_shot_examples)+"_shots"+"_r"+str(run)] = [1 if x.lower() != "no" else 0 for x in predictions]
            
            eval_df.to_csv(prediction_path, index=False)
            
            print(f"\nPredictions saved to {prediction_path}")
        except Exception as e:
            print(f"\nWarning: Could not save predictions: {e}")

    # Compute metrics
    y_true = test_df["label_text"].tolist()
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", pos_label="yes")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"\nResults for {mode} evaluation:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


    # Clean up to free memory# Clean up to free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model for dehumanization detection")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the fine-tuned model (if using fine-tuned)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--prompt_type", type=str, required=True, choices=list(PROMPTS.keys()), help="Type of prompt to use")
    parser.add_argument("--use_quantization", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--mode", type=str, default="zero_shot", choices=["zero_shot", "fine_tuned"], help="Evaluation mode")
    parser.add_argument("--text_column", type=str, default="displayed_text", help="Name of the text column in the dataset")
    parser.add_argument("--no_save", action="store_true", help="Do not save results to file")
    parser.add_argument("--train_dataset_path", type=str, help="Path to training dataset for few-shot examples")
    parser.add_argument("--num_few_shot_examples", type=int, default=20, help="Number of few-shot examples to use")
    parser.add_argument("--cross_validation", action="store_false", help="Use 4-bit quantization")
    parser.add_argument("--save_path", type=str, help="Path to save predictions")
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        prompt_type=args.prompt_type,
        use_quantization=args.use_quantization,
        mode=args.mode,
        text_column=args.text_column,
        save_results=not args.no_save,
        train_dataset_path=args.train_dataset_path,
        num_few_shot_examples=args.num_few_shot_examples,
        cross_validation=args.cross_validation,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
