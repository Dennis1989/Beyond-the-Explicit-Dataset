# Scripts Documentation

## Few-Shot Examples Generation

### `generate_few_shots.py`

This script generates stratified sets of few-shot examples for dehumanization detection. It ensures balanced label distribution in each set and organizes the outputs by dehumanization type.

### Usage

```bash
python3 experiments/scripts/generate_few_shots.py \
    --train_dataset_path PATH_TO_TRAINING_DATA \
    --dehum_type TYPE \
    [optional arguments]
```

### Arguments

#### Required Arguments:
- `--train_dataset_path`: Path to your training dataset
- `--dehum_type`: Type of dehumanization (e.g., explicit, implicit)

#### Optional Arguments:
- `--text_column`: Name of the text column (default: "displayed_text")
- `--label_column`: Name of the label column (default: "label_text")
- `--file_name`: Custom prefix for output files
- `--num_examples`: Number of examples per set (default: 20)
- `--num_sets`: Number of different sets to generate (default: 5)
- `--output_dir`: Directory to save the few-shot sets (default: "experiments/few_shots")
- `--random_state`: Initial random state for reproducibility (default: 42)

### Output Structure

The script creates the following directory structure:
```
experiments/
└── few_shots/
    ├── explicit/
    │   ├── explicit_few_shot_set_1_20_examples.csv
    │   ├── explicit_few_shot_set_2_20_examples.csv
    │   └── ...
    └── implicit/
        ├── implicit_few_shot_set_1_20_examples.csv
        ├── implicit_few_shot_set_2_20_examples.csv
        └── ...
```

### Examples

1. Basic usage with default parameters:
```bash
python3 experiments/scripts/generate_few_shots.py \
    --train_dataset_path data/train.csv \
    --dehum_type explicit
```

2. Generate 10 examples in 3 sets with custom file name:
```bash
python3 experiments/scripts/generate_few_shots.py \
    --train_dataset_path data/train.csv \
    --dehum_type implicit \
    --num_examples 10 \
    --num_sets 3 \
    --file_name implicit_test
```

3. Specify custom column names:
```bash
python3 experiments/scripts/generate_few_shots.py \
    --train_dataset_path data/train.csv \
    --dehum_type explicit \
    --text_column text \
    --label_column class
```

### Features

- **Stratified Sampling**: Ensures balanced representation of labels in each set
- **Multiple Sets**: Generates multiple different sets for robust evaluation
- **Organized Output**: Creates separate directories for each dehumanization type
- **Flexible Naming**: Supports custom file naming
- **Reproducible**: Uses random state for reproducible results

### Output Format

Each generated CSV file contains:
- The specified text column (default: "displayed_text")
- The specified label column (default: "label_text")
- Equal distribution of labels (stratified sampling)
- Random ordering of examples within each set 