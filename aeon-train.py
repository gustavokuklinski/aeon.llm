!pip install transformers datasets trl torch accelerate bitsandbytes huggingface_hub

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os
from huggingface_hub import notebook_login
notebook_login()

device = ("cuda")

model_name = "gustavokuklinski/aeon-360m"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, attn_implementation='eager')

model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

from datasets import load_dataset, concatenate_datasets, Dataset
import json

def format_gpt(example):
    completion = []
    prompt = []
    prompt.append({'role': 'user', 'content': f"{example['human_prompt']}"})
    completion.append({'role': 'assistant', 'content': f"{example['chatgpt_response']}"})
    return {'completion': completion, 'prompt': prompt}


gpt_data = load_dataset("MohamedRashad/ChatGPT-prompts", split="train")
gpt_data_ds = gpt_data.map(format_gpt, remove_columns=gpt_data.column_names)



def format_aeon(example):
    completion = []
    prompt = []
    prompt.append({'role': 'user', 'content': f"{example['instruction']}"})
    completion.append({'role': 'assistant', 'content': f"{example['response']}"})
    return {'completion': completion, 'prompt': prompt}


aeon_data = load_dataset("gustavokuklinski/aeon", split='train')
aeon_data_ds = aeon_data.map(format_aeon, remove_columns=aeon_data.column_names)

def format_contradiction_data(example):
    completion = []
    prompt = []
    prompt.append({'role': 'user', 'content': f"{example['premise']}"})
    completion.append({'role': 'assistant', 'content': f"{example['hypothesis']}"})
    return {'completion': completion, 'prompt': prompt}


contradiction_data = load_dataset("chitra/contradiction", split="train")
contradiction_ds = contradiction_data.map(format_contradiction_data, remove_columns=contradiction_data.column_names)

ds = concatenate_datasets([
    aeon_data_ds,
    gpt_data_ds,
    contradiction_ds
])

print(ds)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    gradient_checkpointing=True,
    warmup_steps=10,
    max_steps=200,
    learning_rate=3e-4,
    fp16=False, # Comment if device='cpu'
    bf16=torch.cuda.is_bf16_supported(), # Comment if device='cpu'
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=training_args,
)

trainer.train()

save_directory = "./aeon"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

