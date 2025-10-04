from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os
import json

print("\033[38;5;160m___________________________________________________\033[0m")
print("")
print("\033[38;5;160m      ###       #######     #######    ###     ### \033[0m")
print("\033[38;5;160m    ### ###     ##        ###     ###  ######  ### \033[0m")
print("\033[38;5;160m   ###   ###    #######   ###     ###  ###  ## ### \033[0m")
print("\033[38;5;160m  ###     ###   ##        ###     ###  ###   ##### \033[0m")
print("\033[38;5;160m ##         ##  #######     #######    ###     ### \033[0m")
print("\033[38;5;160m_FINETUNE__________________________________________\033[0m")
print("")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[1;93m[TRAINER]\033[0m Device found: {device}")


model_name = "HuggingFaceTB/SmolLM2-360M"


model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer.pad_token = tokenizer.eos_token
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)


def format_text(example):
    return {'text': example['text'] + tokenizer.eos_token}

# CPT Data
aeon_books_data = load_dataset('gustavokuklinski/aeon-books', split='train')
aeon_books_ds = aeon_books_data.map(format_text, remove_columns=aeon_books_data.column_names)
cpt_split_datasets = aeon_books_ds.train_test_split(test_size=0.1, seed=42)
cpt_train_dataset = cpt_split_datasets['train']
cpt_eval_dataset = cpt_split_datasets['test']
print(f"\033[1;36m[DATASET]\033[0m CPT Training set size: {len(cpt_train_dataset)} examples")

# STAGE 1 ---------

cpt_directory = "./aeon/raw_llm"

cpt_args = TrainingArguments(
    output_dir=f"{cpt_directory}/outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=10,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(), 
    bf16=False,
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

cpt_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=cpt_train_dataset,
    args=cpt_args,
    eval_dataset=cpt_eval_dataset,
)

cpt_trainer.train()


model.save_pretrained(cpt_directory)
tokenizer.save_pretrained(cpt_directory)
print(f"\033[1;36m[CPT]\033[0m CPT Complete")

