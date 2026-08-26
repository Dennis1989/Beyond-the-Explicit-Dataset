#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, concatenate_datasets

# Constants
PROMPTS = {
    "explicit_animalistic": "Does the following text describe a person or group of people directly or indirectly as animals (e.g., cockroaches, dogs), inferior creatures (e.g., monsters, subhumans) or diseases (e.g., viruses, pests) in a negative way?",
    "explicit_mechanistic": "Does the following text describe a person or group of people directly or indirectly as inanimate objects (e.g. objects, machines, tools, or robots) in a negative way?",
    "implicit_animalistic": "Does the following text describe a person or group of people directly or indirectly as being uncivil/uncultured, irrational, unintelligent, immature/child-like, or lacking morals?",
    "implicit_mechanistic": "Does the following text describe a person or group of people directly or indirectly as being cold, passive, superficial, fungible, or rigid?",
    "explicit": "Does the following text describe a person or group of people directly or indirectly as animals (e.g., cockroaches, dogs), inferior creatures (e.g., monsters, subhumans), diseases (e.g., viruses, pests), or inanimate objects (such as machines, tools, or robots) in a negative way?"
     
}

SYSTEM_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a super helpful graduate student. Your job is to help identify examples that help train machine learning models to detect various forms of dehumanization."
)

RESPOND_YES_NO = "Respond with 'yes' or 'no'\n"

def construct_few_shot_prompt(sentence, examples, prompt_type):
    """Creates a few-shot prompt using N examples before the test sentence."""
    prompt = SYSTEM_PROMPT

    # Add few-shot examples
    for ex in examples:
        ex_sentence, ex_label = ex
        prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        prompt += f"{PROMPTS[prompt_type]} {RESPOND_YES_NO}"
        prompt += f"Sentence: {ex_sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{ex_label}<|eot_id|>\n"

    # Add the test sentence
    prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    prompt += f"{PROMPTS[prompt_type]} {RESPOND_YES_NO}"
    prompt += f"Sentence: {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    return prompt

def classify_sentence(sentence, model, tokenizer, prompt_type, few_shots=None, label=None):
    """Classify a sentence using the model."""
    if few_shots is None:
        prompt = (
            SYSTEM_PROMPT +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{PROMPTS[prompt_type]} {RESPOND_YES_NO}"
            f"Sentence: {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        
    else:
        prompt = construct_few_shot_prompt(sentence, few_shots, prompt_type)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            do_sample=False
        )

    response = tokenizer.decode(output[0], skip_special_tokens=False)
    

    if "<|start_header_id|>assistant<|end_header_id|>\n" in response:
        assistant_responses = response.split("<|start_header_id|>assistant<|end_header_id|>\n")
        answer = assistant_responses[-1].strip().split("<|eot_id|>")[0]
    else:
        answer = response.strip()
    
    if label == answer and label == "yes":
        print("--------------------------------")
       
        if label is not None: 
            print("Prompt: ", prompt) 
            print("Ground Truth: ", label)
            print("Predicted: ", answer)
            print("--------------------------------")

    return answer.lower()

def classify_sentence_logits(sentence, model, tokenizer, prompt_type, few_shots=None):
    """Generate a prediction using greedy decoding and extract logits."""
    if few_shots is None:
        prompt = (
            SYSTEM_PROMPT +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{PROMPTS[prompt_type]} {RESPOND_YES_NO}"
            f"Sentence: {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:
        prompt = construct_few_shot_prompt(sentence, few_shots, prompt_type)

    # Tokenisierung
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Letzter Token-Index (wo die nächste Vorhersage passiert)
    last_token_index = inputs['input_ids'].shape[1] - 1
    logits = outputs.logits[0, last_token_index]  # shape: [vocab_size]

    # Token-IDs für ' yes' und ' no' (Leerzeichen wichtig!)
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]

    # Richtige Extraktion der Logits (direkt im Tensor)
    selected_logits = logits[[yes_token_id, no_token_id]]
    probs = F.softmax(selected_logits, dim=0)

    return {
        "yes_logit": selected_logits[0].item(),
        "no_logit": selected_logits[1].item(),
        "yes_prob": probs[0].item(),
        "no_prob": probs[1].item()
    }


def apply_lora(model):
    print("Applying LoRA")
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules= ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    model = get_peft_model(model, lora_config)
    
    return model

def load_model(model_name, output_dir=None, use_quantization=True, fine_tuned=False, run=1):
    """Load either the fine-tuned or base model."""
    print(f"Loading {'Fine-Tuned' if fine_tuned else 'Zero-Shot'} Model...")
    
    model_path = output_dir if fine_tuned else model_name
    
    if use_quantization:
        print("Loading quantized model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path if not fine_tuned else model_name,  # Use base model for PEFT
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,  # Optional but helps memory
                "bnb_4bit_quant_type": "nf4"  
            }
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path if not fine_tuned else model_name,  # Use base model for PEFT
            device_map="cuda:0",
            torch_dtype=torch.bfloat16
        )
    
    if fine_tuned:
        print(f"Loading finetuned PEFT model from {output_dir}  /_run_{run}")
        model = PeftModel.from_pretrained(model, output_dir + f"/_run_{run}")

    #model.eval()
    return model

def load_tokenizer(model_name):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_dataset_from_paths(dataset_paths, shuffle=True):
    """Load and optionally combine multiple datasets."""
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    
    datasets = []
    for path in dataset_paths:
        dataset_part = load_dataset("csv", data_files=path, split="train")
        datasets.append(dataset_part)
    
    if len(datasets) > 1:
        dataset = concatenate_datasets(datasets)
    else:
        dataset = datasets[0]
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset 