import argparse
 
import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, XLMRobertaTokenizer, \
    XLMRobertaForSequenceClassification
import re
 
 
def evaluate_model(tokenizer_name, model_name, model_type, gold_label, input_file, predictions_file=None, average="macro", batch_size=32):
    """
    Evaluates a text classification model and prints performance metrics.
 
    Args:
        tokenizer_name (str): Path or name of the tokenizer.
        model_name (str): Path or name of the model.
        model_type (str): 'bert' or 'xlm-roberta' to specify the correct model class.
        gold_label (str): gold label column name.
        input_file (str): Path to the input CSV file.
        predictions_file (str, optional): Path to save predictions CSV.
        average (str, optional): Metrics calculation average (default: macro).
        batch_size (int, optional): Batch size for inference (default: 32).
    """
    if model_type.lower() == 'bert':
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
    elif model_type.lower() == 'roberta':
        tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        model = XLMRobertaForSequenceClassification.from_pretrained(model_name)
    else:
        raise ValueError("Invalid model type. Use 'bert' or 'roberta'.")
 
    df = pd.read_csv(input_file)
 
    X_test = df['displayed_text'].astype(str).tolist()
    y_test = df[gold_label].astype(int).tolist()
 
    X_test_tokens = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')
    y_test_tensor = torch.tensor(y_test)
 
    test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
 
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
 
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
 
    accuracy = accuracy_score(y_test, all_predictions)
    precision = precision_score(y_test, all_predictions, average=average)
    recall = recall_score(y_test, all_predictions, average=average)
    f1 = f1_score(y_test, all_predictions, average=average)
 
    print(f'Accuracy {average}: {accuracy}', flush=True)
    print(f'Precision {average}: {precision}', flush=True)
    print(f'Recall {average}: {recall}', flush=True)
    print(f'F1 Score {average}: {f1}', flush=True)
 
 
    match = re.search(r'_([^_]+_[^_/]+)/model$', model_name)
    if match:
        model_name = match.group(1) 
    if predictions_file:
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            df[model_name] = all_predictions
            print(f'Updated {model_name} predictions in {predictions_file}', flush=True)
        else:
            df = df[["instance_id", "displayed_text", gold_label, "target_masked_text", "label_text"]]
            df[model_name] = all_predictions
            print(f'Created new file {predictions_file} with {model_name} predictions', flush=True)
 
        df.to_csv(predictions_file, index=False)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Path or name of the tokenizer.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the model.")
    parser.add_argument("--model_type", type=str, choices=["bert", "roberta"], required=True, help="Type of model: 'bert' or 'roberta'.")
    parser.add_argument("--gold_label", type=str, required=True, help="Gold label column name'.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--predictions_file", type=str, default=None, help="Path to save predictions CSV (optional).")
    parser.add_argument("--average", type=str, default="macro", help="Metrics calculation average (default: macro)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference (default: 32).")
 
    args = parser.parse_args()
 
    evaluate_model(
        tokenizer_name=args.tokenizer_name,
        model_name=args.model_name,
        model_type=args.model_type,
        gold_label=args.gold_label,
        input_file=args.input_file,
        predictions_file=args.predictions_file,
        average=args.average,
        batch_size=args.batch_size
    )