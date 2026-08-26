#!/usr/bin/env python
# coding: utf-8

import argparse
from utils import (
    load_model, load_tokenizer, classify_sentence,
    PROMPTS, load_dataset_from_paths
)

def main():
    parser = argparse.ArgumentParser(description="Classify a single sentence using a fine-tuned model")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the fine-tuned model")
    parser.add_argument("--sentence", type=str, required=True, help="Sentence to classify")
    parser.add_argument("--prompt_type", type=str, default="explicit_mechanistic", choices=list(PROMPTS.keys()), help="Type of prompt to use")
    parser.add_argument("--no_quantization", action="store_true", help="Do not use 4-bit quantization")
    parser.add_argument("--no_fine_tuned", action="store_true", help="Use base model instead of fine-tuned model")

    args = parser.parse_args()
    model = load_model(args.model_name, args.output_dir, True, fine_tuned=(args.no_fine_tuned != "zero_shot"))
    tokenizer = load_tokenizer(args.model_name)
    
    print(classify_sentence(args.sentence,model,tokenizer,args.prompt_type,few_shots=None))


if __name__ == "__main__":
    main() 