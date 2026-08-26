import krippendorff
import pandas as pd

new_data = pd.read_csv('annotated_data/prolific/implicit_animalistic/annotated_instances_imp_animal_combined.csv')

#label_cols = ['offensive', 'agg_animal', 'animal', 'subhuman', 'disease', 'inanimate', 'exp_dehum_binary']
label_cols = ['offensive', 'implicit', 'irrational', 'morals', 'unintelligent', 'childlike', 'incivil', 'imp_dehum_binary', 'nothing']

iaa_results = {}

for label in label_cols:
    pivot = new_data.pivot_table(index='instance_id', columns='user', values=label)
    data_matrix = pivot.values.T
    alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='nominal')
    iaa_results[label] = alpha

for label, alpha in iaa_results.items():
    print(f"{label}: Krippendorff's alpha = {alpha:.3f}")