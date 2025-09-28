import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
import torch

MODEL_PATH = "./aeon/raw_llm"  # Directory from ./pretrain.py
FINAL_SAVE_DIR = "./aeon/finetuned_llm" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[1;93m[TRAINER]\033[0m Device found: {device}")

# System prompt for Aeon's persona
SYSTEM_PROMPT = (
    "You are Aeon, a helpful, curious, and friendly AI assistant. "
    "Your name is always Aeon. You were created by Gustavo Kuklinski. "
    "Maintain a warm, chatty, and engaging tone in all conversations."
)

def format_aeon(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {'messages': messages}

try:
    print(f"\033[1;36m[INFO]\033[0m Loading instruction dataset")
    
    # Load ONLY the instruction dataset for fine-tuning
    raw_instruction_data = load_dataset('gustavokuklinski/aeon', split='train')
    
    # Map the data to the required chat format
    formatted_train_ds = raw_instruction_data.map(
        format_aeon, 
        remove_columns=raw_instruction_data.column_names,
        desc="Formatting instruction data"
    )

    print(f"\033[1;32m[DATASET]:\033[0m Instruction training set size: {len(formatted_train_ds)} examples")
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
    output_dir="./sft_outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=10,
    max_steps=300, # Use steps for controlled SFT duration
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(), 
    bf16=False, # Disable BF16 unless using A100/H100
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted_train_ds,
    args=training_args
)

print("\n\033[1;36m[INFO]\033[0m Starting Instruction Fine-Tuning (SFT)...")
trainer.train()

model.save_pretrained(FINAL_SAVE_DIR)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"\n\033[1;32m[SUCCESS]\033[0m Fine-tuning complete. Model saved to {FINAL_SAVE_DIR}/")
