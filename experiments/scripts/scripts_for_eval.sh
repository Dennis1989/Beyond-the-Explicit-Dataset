## generate few shots

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/explicit/english/train_explicit_animalistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type explicit_animalistic

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/explicit/english/train_explicit_mechanistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type explicit_mechanistic

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/implicit/english/train_implicit_mechanistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type implicit_mechanistic

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/implicit/english/train_implicit_animalistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type implicit_animalistic

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/explicit/german/train_german_explicit_animalistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type explicit_animalistic --output_dir experiments/few_shots/german

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/explicit/german/train_german_explicit_mechanistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type explicit_mechanistic --output_dir experiments/few_shots/german

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/implicit/german/train_german_implicit_mechanistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type implicit_mechanistic --output_dir experiments/few_shots/german

python3 -m experiments.scripts.generate_few_shots --train_dataset_path experiments/splits/implicit/german/train_german_implicit_animalistic_dehumanization.csv --text_column displayed_text --label_column label_text --num_examples 20 --num_sets 5 --dehum_type implicit_animalistic --output_dir experiments/few_shots/german

##train models

screen python  -m experiments.training.batch_train experiments/configs/training/config_train_explicit_large.json
screen python -m experiments.training.batch_train experiments/configs/training/config_train_implicit_large.json
screen python -m experiments.training.batch_train experiments/configs/training/config_train_mechanistic.json



# evaluate models
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_animalistic/batch_eval_config_animal_ex_8b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_mechanistic/batch_eval_config_mecha_ex_8b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_animalistic/batch_eval_config_animal_imp_8b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_mechanistic/batch_eval_config_mecha_imp_8b.json

python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_animalistic/batch_eval_config_animal_ex_70b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_mechanistic/batch_eval_config_mecha_ex_70b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_animalistic/batch_eval_config_animal_imp_70b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_mechanistic/batch_eval_config_mecha_imp_70b.json

## evaluate german models

python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_animalistic/german_batch_eval_config_animal_ex_8b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_mechanistic/german_batch_eval_config_mecha_ex_8b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_animalistic/german_batch_eval_config_animal_imp_8b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_mechanistic/german_batch_eval_config_mecha_imp_8b.json

python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_animalistic/german_batch_eval_config_animal_ex_70b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/explicit_mechanistic/german_batch_eval_config_mecha_ex_70b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_animalistic/german_batch_eval_config_animal_imp_70b.json
python -m experiments.evaluation.batch_eval experiments/configs/evaluation/implicit_mechanistic/german_batch_eval_config_mecha_imp_70b.json



## generate metrics

python3 experiments/evaluation/metrics.py --file_path experiments/splits/explicit/english/predictions/all_english_explicit_animalistic_predictions.csv --label "explicit_animalistic_dehumanization"
python3 experiments/evaluation/metrics.py --file_path experiments/splits/implicit/predictions/all_english_implicit_animalistic_predictions.csv --label "implicit_animalistic_dehumanization"


python3 experiments/evaluation/metrics_cross_eval.py 

## chatgpt eval
screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/explicit_animalistic/batch_eval_config_animal_ex_chatgpt.json 
screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/explicit_mechanistic/batch_eval_config_mecha_ex_chatgpt.json
screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/implicit_animalistic/batch_eval_config_animal_imp_chatgpt.json
screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/implicit_mechanistic/batch_eval_config_mecha_imp_chatgpt.json

screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/explicit_animalistic/german_batch_eval_config_animal_ex_chatgpt.json 

screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/explicit_mechanistic/german_batch_eval_config_mecha_ex_chatgpt.json

screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/implicit_animalistic/german_batch_eval_config_animal_imp_chatgpt.json

screen python -m experiments.evaluation.batch_chatgpt_eval experiments/configs/evaluation/implicit_mechanistic/german_batch_eval_config_mecha_imp_chatgpt.json



## train bert

python  -m experiments.training.bert_train 
python  -m experiments.training.roberta_train 

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_mechanistic_BERT_r1/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_mechanistic_BERT_r2/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_mechanistic_BERT_r3/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_mechanistic_BERT_r4/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_mechanistic_BERT_r5/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv


python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_animalistic_BERT_r1/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_animalistic_BERT_r2/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_animalistic_BERT_r3/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_animalistic_BERT_r4/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name bert-base-multilingual-uncased   --model_name experiments/bert/german_explicit_animalistic_BERT_r5/model    --model_type bert   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv


python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_mechanistic_RoBERTa_r1/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_mechanistic_RoBERTa_r2/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_mechanistic_RoBERTa_r3/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_mechanistic_RoBERTa_r4/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_mechanistic_RoBERTa_r5/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv   --gold_label explicit_mechanistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_mechanistic_predictions.csv


###
python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_animalistic_RoBERTa_r1/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_animalistic_RoBERTa_r2/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_animalistic_RoBERTa_r3/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_animalistic_RoBERTa_r4/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv

python  -m experiments.evaluation.eval_bert   --tokenizer_name xlm-roberta-large   --model_name experiments/bert/german_explicit_animalistic_RoBERTa_r5/model    --model_type roberta   --input_file experiments/splits/explicit/german/test_german_explicit_animalistic_dehumanization.csv   --gold_label explicit_animalistic_dehumanization   --predictions_file experiments/splits/explicit/german/predictions/german_explicit_animalistic_predictions.csv


##plot 

python post_analysis/plot_distributions.py --dataset_path data/twitter_final_candidates_german_annotated.csv --prob_columns "Llama-3.1-8B-Instruct_explicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_explicit_mechanistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_mechanistic_yes_probs" --title "Twitter German" --output_path "distribution_twitter_german_plot.pdf" --fig_width 9 --fig_height 5 --log_scale true

python post_analysis/plot_distributions.py --dataset_path data/reddit_final_candidates_german_annotated.csv --prob_columns "Llama-3.1-8B-Instruct_explicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_explicit_mechanistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_mechanistic_yes_probs" --title "Reddit German" --output_path "distribution_reddit_german_plot.pdf" --fig_width 9 --fig_height 5 --log_scale true

python post_analysis/plot_distributions.py --dataset_path data/twitter_final_harmonized_annotated.csv --prob_columns "Llama-3.1-8B-Instruct_explicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_explicit_mechanistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_mechanistic_yes_probs" --title "Twitter English" --output_path "distribution_twitter_plot.pdf" --fig_width 9 --fig_height 5 --log_scale true

python post_analysis/plot_distributions.py --dataset_path data/reddit_final_harmonized_annotated.csv --prob_columns "Llama-3.1-8B-Instruct_explicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_explicit_mechanistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_animalistic_yes_probs" "Llama-3.1-8B-Instruct_implicit_mechanistic_yes_probs" --title "Reddit English" --output_path "distribution_reddit_plot.pdf" --fig_width 9 --fig_height 5 --log_scale true