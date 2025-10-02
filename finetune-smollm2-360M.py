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

save_directory = "./aeon"

model_name = "HuggingFaceTB/SmolLM2-360M"

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
tokenizer.pad_token = tokenizer.eos_token
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

def format_text(example):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"\nContext: {example['text']}\n\n"},
    ]
    return {'messages': messages}

try:
    aeon_books_data = load_dataset('gustavokuklinski/aeon-books', split='train')
    aeon_books_ds = aeon_books_data.map(format_text, remove_columns=aeon_books_data.column_names)

    aeon_train_data = load_dataset('gustavokuklinski/aeon', split='train')
    aeon_train_ds = aeon_train_data.map(format_aeon, remove_columns=aeon_train_data.column_names)

    combined_train_ds = concatenate_datasets([aeon_books_ds, aeon_train_ds])

    split_datasets = combined_train_ds.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

    print(f"\033[1;32m[DATASET]:\033[0m Dataset lengths:")
    print(f"\033[1;32m[DATASET]:\033[0m Total examples: {len(combined_train_ds)} ")
    print(f"\033[1;32m[DATASET]:\033[0m Training set size: {len(train_dataset)} examples")
    print(f"\033[1;32m[DATASET]:\033[0m Evaluation set size (10%): {len(eval_dataset)} examples")

    print(split_datasets)

except FileNotFoundError as e:
    print(f"\033[1;91m[ERROR]\033[0m Dataset file not found. Please ensure the paths are correct.")
    print(f"\033[1;91m[ERROR]\033[0m {e}")
    exit()

training_args = TrainingArguments(
    output_dir="./aeon/finetuned_llm/outputs",
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

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    eval_dataset=eval_dataset,
)

trainer.train()

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)