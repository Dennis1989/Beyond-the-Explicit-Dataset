import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score

base_path = "experiments/splits"
modes = ["explicit","implicit"]
langs = ["german"]
num_rounds = 5
label_column = "expert_decision_label"
fallback_column = "gold_label"

results = []

for mode in modes:
    for lang in langs:
        lang_path = os.path.join(base_path, mode, lang, "predictions")
        print(lang_path)

        if not os.path.isdir(lang_path):
            continue

        for file in os.listdir(lang_path):
            #if not file.startswith("all") or not file.endswith(".csv"):
            #     continue

            file_path = os.path.join(lang_path, file)
            print(file_path)
            df = pd.read_csv(file_path)

            is_german = lang == "german"

            match = re.match(r"(german|english)_(.*)_predictions\.csv", file)
            if match:
                dehum_type = match.group(2)
            else:
                dehum_type = "unknown"
            
            if "expert_decision_label" in df.columns:
                label_column = "expert_decision_label"
            else:
                label_column = "gold_label"
            print(df[label_column])
            df[label_column] = pd.to_numeric(df[label_column], errors='coerce').astype('Int64')
            df[fallback_column] = pd.to_numeric(df[label_column], errors='coerce').astype('Int64')
            y_true = np.where(
                df[label_column].isin([0, 1]),
                df[label_column],
                df[fallback_column]
            ).astype(int)
            print("yeeah1")
            # TODO: when the finetuned models have N runs, move them from single run to model_names
            if is_german:
                model_names = ["BERT", "RoBERTa","Llama-3.1-8B-Instruct_zero_shot","Llama-3.1-8B-Instruct_20_shots","Llama-3.1-8B-Instruct_fine_tuned","Llama-3.1-70B-Instruct_zero_shot","Llama-3.1-70B-Instruct_20_shots","Llama-3.1-70B-Instruct_fine_tuned","gpt4","gpt4_20_shots"]
                single_run_models = [
                    "perspective_api_toxicity",
                ]
                lang_code = "de"
            else:
                model_names = ["BERT", "RoBERTa", "HateBERT", "MetaHateBERT","Llama-3.1-8B-Instruct_zero_shot","Llama-3.1-8B-Instruct_20_shots","Llama-3.1-8B-Instruct_fine_tuned","Llama-3.1-70B-Instruct_zero_shot","Llama-3.1-70B-Instruct_20_shots","Llama-3.1-70B-Instruct_fine_tuned","gpt4","gpt4_20_shots"]
                single_run_models = [
                    "perspective_api_toxicity",
                    "gpt_4o",
                    "HateBERT_vanilla", "MetaHateBERT_vanilla",
                    "llama_70b_zero_shot", "llama_70b_20_shot", "llama_70b_finetuned",
                ]
                lang_code = "en"
            print("yeeah2")
            for model in model_names:
                for run in range(1, num_rounds + 1):
                    col_name = f"{model}_r{run}"
                    if col_name not in df.columns:
                        continue
                    print("yeeah3")
                    y_pred = df[col_name].astype(int)

                    prec = precision_score(y_true, y_pred, average="binary", pos_label=1)
                    rec = recall_score(y_true, y_pred, average="binary", pos_label=1)
                    f1 = f1_score(y_true, y_pred, average="macro", pos_label=1)

                    results.append({
                        "model": model,
                        "dehum_type": dehum_type,
                        "run": run,
                        "prec": round(prec, 2),
                        "rec": round(rec, 2),
                        "f1": round(f1, 2),
                        "lang": lang_code
                    })

            for model in single_run_models:
                if model not in df.columns:
                    continue

                if model.startswith("perspective"):
                    y_pred = (df[model] >= 0.5).astype(int)
                else:
                    y_pred = df[model].astype(int)

                prec = precision_score(y_true, y_pred, average="binary", pos_label=1)
                rec = recall_score(y_true, y_pred, average="binary", pos_label=1)
                f1 = f1_score(y_true, y_pred, average="macro", pos_label=1)

                results.append({
                    "model": model,
                    "dehum_type": dehum_type,
                    "run": "vanilla",
                    "prec": round(prec, 2),
                    "rec": round(rec, 2),
                    "f1": round(f1, 2),
                    "lang": lang_code
                })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv("experiments/evaluation/eval_summary_german_v3.csv", index=False)

print(results_df)
