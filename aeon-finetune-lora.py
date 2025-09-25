from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
import re

print("\033[38;5;160m___________________________________________________\033[0m")
print("")
print("\033[38;5;160m      ###       #######     #######    ###     ### \033[0m")
print("\033[38;5;160m    ### ###     ##        ###     ###  ######  ### \033[0m")
print("\033[38;5;160m   ###   ###    #######   ###     ###  ###  ## ### \033[0m")
print("\033[38;5;160m  ###     ###   ##        ###     ###  ###   ##### \033[0m")
print("\033[38;5;160m ##         ##  #######     #######    ###     ### \033[0m")
print("\033[38;5;160m_FINETUNE WITH LORA ON CPU__________________________________________\033[0m")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[1;93m[TRAINER]\033[0m Device found: {device}")

save_directory = "./aeon_lora"
os.makedirs(save_directory, exist_ok=True)

model_name = "HuggingFaceTB/SmolLM2-360M"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

system_prompt = (
    "You are Aeon, a helpful, curious, and friendly AI assistant. "
    "Your name is always Aeon. You were created by Gustavo Kuklinski. "
    "You are not the user. Never claim to be the user. "
    "Maintain a warm, chatty, and engaging tone in all conversations. "
    "If you don't know the user question, just state: 'I don't know'."
)

def format_aeon(example):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {'text': messages}

def format_text(example):
    messages = [
        {"role": "user", "content": f"Read the following: {example['text']}\n\nfrom:{example['title']}\n\n"},
    ]
    return {'text': messages}


def flatten_messages(example):
    text = ""
    for msg in example['text']:
        role = msg['role'].upper()
        text += f"[{role}] {msg['content']}\n"
    return {"text": text}

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

try:
    aeon_books_data = load_dataset('gustavokuklinski/aeon-books', split='train')
    aeon_books_ds = aeon_books_data.map(format_text, remove_columns=aeon_books_data.column_names)

    aeon_train_data = load_dataset('gustavokuklinski/aeon', split='train')
    aeon_train_ds = aeon_train_data.map(format_aeon, remove_columns=aeon_train_data.column_names)

    combined_train_ds = concatenate_datasets([aeon_books_ds, aeon_train_ds])
    combined_train_ds = combined_train_ds.map(flatten_messages)
    tokenized_dataset = combined_train_ds.map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"\033[1;32m[DATASET]:\033[0m Combined training set: {len(combined_train_ds)} examples")

except FileNotFoundError as e:
    print(f"\033[1;91m[ERROR]\033[0m Dataset file not found. Please ensure the paths are correct.")
    exit()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=10,
    max_steps=1000,
    learning_rate=2e-5,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    report_to=[],
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    args=training_args,
    
)

trainer.train()

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("\033[1;32m[SUCCESS]\033[0m LoRA fine-tuning completed and saved.")
