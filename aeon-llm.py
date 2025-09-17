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

model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

def format_aeon(example):
    system_prompt = "You are Aeon, a helpful, curious, and friendly AI assistant." \
                "Your name is always Aeon. You were created by Gustavo Kuklinski." \
                "You are not the user. Never claim to be the user." \
                "Maintain a warm, chatty, and engaging tone in all conversations." \
                "Be naturally conversational while providing helpful responses."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {'messages': messages}

try:
    aeon_train_data = load_dataset('gustavokuklinski/aeon', split='train')
    aeon_train_ds = aeon_train_data.map(format_aeon, remove_columns=aeon_train_data.column_names)

    print(f"\033[1;32m[DATASET]:\033[0m Dataset lengths:")
    print(f"\033[1;32m[DATASET]:\033[0m Training set: {len(aeon_train_ds)} examples")

except FileNotFoundError as e:
    print(f"\033[1;91m[ERROR]\033[0m Dataset file not found. Please ensure the paths are correct.")
    print(f"\033[1;91m[ERROR]\033[0m {e}")
    exit()

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=10,
    max_steps=200,
    learning_rate=3e-4,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=aeon_train_ds,
    args=training_args,
)

trainer.train()

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)