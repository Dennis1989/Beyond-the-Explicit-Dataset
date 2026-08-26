#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from experiments.utils.utils import PROMPTS, load_dataset_from_paths
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize OpenAI client
client = OpenAI(api_key="KEY_HERE")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_chatgpt_response(messages, model_name):
    """Make API call to ChatGPT with retry logic."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
        )
        return response
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        raise  # This will trigger the retry

def evaluate_chatgpt(
    dataset_path,
    prompt_type,
    text_column="displayed_text",
    save_results=True,
    model_name="gpt-3.5-turbo",
    save_path=None,
    runs=5,
    train_dataset_path=None,
    num_few_shot_examples=20
):
    """Evaluate ChatGPT on the specified dataset."""
    # Load test dataset
    test_df = pd.read_csv(dataset_path)
    test_df["label_text"] = test_df["label_text"].astype(str).str.lower()

    # Load training examples if few-shot learning is requested
    

    # Get prompt template
    prompt_template = PROMPTS[prompt_type]
    print(prompt_template)
    # Run inference with progress bar
    
    for run in range(1, runs + 1):
        few_shot_examples = ""
        
        if train_dataset_path and num_few_shot_examples > 0:
            
            few_shot_df = load_dataset_from_paths(train_dataset_path+"/"+str(prompt_type)+"_few_shot_set_"+str(run)+"_"+str(num_few_shot_examples)+"_examples.csv", shuffle=False)
            few_shot_df = pd.DataFrame(few_shot_df)

            few_shot_data = few_shot_df[[text_column, "label_text"]].values.tolist()
            # Create few-shot examples string
            examples = []
            for i, (text, label) in enumerate(few_shot_data, 1):
                examples.append(f"Sentence: {text}\nAnswer: {label}")
            few_shot_examples = "\n\n".join(examples) + "\n\n"
        
        predictions = []
        print("Running ChatGPT inference")
        for text in tqdm(test_df[text_column], desc="Running ChatGPT inference"):
            messages = [
                {"role": "system", "content": "You are a super helpful graduate student. Your job is to help identify examples that help train machine learning models to detect various forms of dehumanization."},
                {"role": "user", "content": f"{prompt_template}. Respond with 'yes' or 'no'.\n\n{few_shot_examples}Sentence: {text}\nAnswer: "}
            ]
            print(messages)
            try:
                response = get_chatgpt_response(messages, model_name)
                #print(response)
                print(response.choices[0].message.content.strip().lower())
                predictions.append(response.choices[0].message.content.strip().lower())
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed after all retries: {str(e)}")
                predictions.append("no")  # Default to "no" if all retries fail

        # Adjust predictions to ensure they are either "yes" or "no"
        predictions = ["yes" if x.lower() != "no" else x.lower() for x in predictions]
        print(predictions)
        # Save predictions if requested
        if save_results:
            try:
                
                prediction_path = os.path.join(save_path + prompt_type+"_"+"predictions.csv")
                print(prediction_path)
                eval_df = pd.read_csv(prediction_path)
            
                if train_dataset_path is  None:
                    eval_df["gpt4"+"_r"+str(run)] = [1 if x.lower() != "no" else 0 for x in predictions]
                else:
                    eval_df["gpt4"+"_"+str(num_few_shot_examples)+"_shots"+"_r"+str(run)] = [1 if x.lower() != "no" else 0 for x in predictions]
                
                eval_df.to_csv(prediction_path, index=False)
                print(f"\nPredictions saved to {prediction_path}")
            
            except Exception as e:
                print(f"\nWarning: Could not save predictions: {e}")

        # Compute metrics
        y_true = test_df["label_text"].tolist()
        y_pred = predictions

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,average="binary", pos_label="yes")

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        print(f"\nResults for {model_name} evaluation:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate ChatGPT for dehumanization detection")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--prompt_type", type=str, required=True, choices=list(PROMPTS.keys()), help="Type of prompt to use")
    parser.add_argument("--text_column", type=str, default="displayed_text", help="Name of the text column in the dataset")
    parser.add_argument("--no_save", action="store_true", help="Do not save results to file")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="ChatGPT model to use")
    parser.add_argument("--train_dataset_path", type=str, help="Path to training dataset for few-shot learning")
    parser.add_argument("--num_few_shot_examples", type=int, default=20, help="Number of few-shot examples to use")

    args = parser.parse_args()

    evaluate_chatgpt(
        dataset_path=args.dataset_path,
        prompt_type=args.prompt_type,
        text_column=args.text_column,
        save_results=not args.no_save,
        model_name=args.model_name,
        train_dataset_path=args.train_dataset_path,
        num_few_shot_examples=args.num_few_shot_examples
    )

if __name__ == "__main__":
    main()

