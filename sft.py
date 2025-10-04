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


cpt_directory = "./aeon/raw_llm"

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=cpt_directory)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=cpt_directory)
tokenizer.pad_token = tokenizer.eos_token
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)


model = AutoModelForCausalLM.from_pretrained(cpt_directory)
tokenizer = AutoTokenizer.from_pretrained(cpt_directory)
# Re-apply chat format setup just in case
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)


system_prompt = "You are Aeon, a helpful, curious, and friendly AI assistant." \
            "Your name is always Aeon. You were created by Gustavo Kuklinski." \
            "You are not the user. Never claim to be the user." \
            "Maintain a warm, chatty, and engaging tone in all conversations." \
            "Be naturally conversational while providing helpful responses."

def format_aeon(example):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {'messages': messages}

# SFT Data
aeon_train_data = load_dataset('gustavokuklinski/aeon', split='train')
aeon_train_ds = aeon_train_data.map(format_aeon, remove_columns=aeon_train_data.column_names)
sft_split_datasets = aeon_train_ds.train_test_split(test_size=0.1, seed=42)
sft_train_dataset = sft_split_datasets['train']
sft_eval_dataset = sft_split_datasets['test']
print(f"\033[1;36m[DATASET]\033[0m SFT Training set size: {len(sft_train_dataset)} examples")

sft_directory = "./aeon/finetuned_llm"

sft_args = TrainingArguments(
    output_dir=f"{sft_directory}/outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=10,
    num_train_epochs=2,
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

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=sft_train_dataset,
    args=sft_args,
    eval_dataset=sft_eval_dataset,
)

sft_trainer.train()

model.save_pretrained(sft_directory)
tokenizer.save_pretrained(sft_directory)

print(f"\033[1;36m[SFT]\033[0m SFT Complete")