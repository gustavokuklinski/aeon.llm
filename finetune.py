import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, setup_chat_format
import torch

MODEL_PATH = "./aeon/raw_llm"  # Directory from ./pretrain.py
FINAL_SAVE_DIR = "./aeon/finetuned_llm"

print("\033[38;5;160m___________________________________________________\033[0m")
print("")
print("\033[38;5;160m      ###       #######     #######    ###     ### \033[0m")
print("\033[38;5;160m    ### ###     ##        ###     ###  ######  ### \033[0m")
print("\033[38;5;160m   ###   ###    #######   ###     ###  ###  ## ### \033[0m")
print("\033[38;5;160m  ###     ###   ##        ###     ###  ###   ##### \033[0m")
print("\033[38;5;160m ##         ##  #######     #######    ###     ### \033[0m")
print("\033[38;5;160m_OWN FINETUNE______________________________________\033[0m")
print("")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[1;93m[TRAINER]\033[0m Device found: {device}")

def format_aeon(example):
    messages = [
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {'messages': messages}


def format_alpaca(example):
    messages = [
        {"role": "user", "content": f"{example['instruction']}\n{example['input']}"},
        {"role": "assistant", "content": f"{example['output']}"}
    ]
    return {'messages': messages}


def format_dolly(example):
    user_content = f"{example['instruction']}"
    if example['context']:
        user_content += f"\nContext: {example['context']}"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['response']}
    ]
    return {'messages': messages}

def format_cosmopedia(example):
    messages = [
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": example['text']}
    ]
    return {'messages': messages}

try:
    print(f"\033[1;36m[INFO]\033[0m Loading instruction dataset")
    
    # Load ONLY the instruction dataset for fine-tuning
    raw_instruction_data = load_dataset('gustavokuklinski/aeon', split='train')
    raw_alpaca_data = load_dataset('tatsu-lab/alpaca', split='train')
    raw_dolly_data = load_dataset('databricks/databricks-dolly-15k', split='train')
    raw_cosmopedia_data = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

    # Map the data to the required chat format
    formatted_aeon_ds = raw_instruction_data.map(
        format_aeon, 
        remove_columns=raw_instruction_data.column_names
    )
    formatted_alpaca_ds = raw_alpaca_data.map(
        format_alpaca, 
        remove_columns=raw_alpaca_data.column_names
    )
    formatted_dolly_ds = raw_dolly_data.map(
        format_dolly, 
        remove_columns=raw_dolly_data.column_names
    )
    formatted_cosmopedia_ds = raw_cosmopedia_data.map(
        format_cosmopedia, 
        remove_columns=raw_cosmopedia_data.column_names
    )

    full_dataset = concatenate_datasets([formatted_dolly_ds, formatted_alpaca_ds, formatted_cosmopedia_ds, formatted_aeon_ds])
    
    split_datasets = full_dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    
    print(f"\033[1;32m[DATASET]:\033[0m Total examples: {len(full_dataset)} ")
    print(f"\033[1;32m[DATASET]:\033[0m Training set size: {len(train_dataset)} examples")
    print(f"\033[1;32m[DATASET]:\033[0m Evaluation set size (10%): {len(eval_dataset)} examples")
    print(f"\033[1;32m[DATASET]:\033[0m Ready for SFTTrainer.")


except Exception as e:
    print(f"\033[1;91m[ERROR]\033[0m Error during dataset loading/mapping: {e}")
    exit()

print(f"\033[1;36m[INFO]\033[0m Loading pre-trained model from {MODEL_PATH}")

try:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
    
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    
except Exception as e:
    print(f"\033[1;91m[ERROR]\033[0m Failed to load pre-trained model/tokenizer from {MODEL_PATH}. Did you run Stage 1?")
    print(f"\033[1;91m[ERROR]\033[0m Details: {e}")
    exit()

training_args = TrainingArguments(
    output_dir="./aeon/finetuned_llm/outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=10,
    num_train_epochs=20,
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
    eval_dataset=eval_dataset,
    args=training_args
)

print("\n\033[1;36m[INFO]\033[0m Starting Instruction Fine-Tuning (SFT)...")
trainer.train()

model.save_pretrained(FINAL_SAVE_DIR)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"\n\033[1;32m[SUCCESS]\033[0m Fine-tuning complete. Model saved to {FINAL_SAVE_DIR}/")
