from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, pipeline
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer
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
print("\033[38;5;160m_FINETUNE_UNSLOTH__________________________________\033[0m")
print("")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\033[1;93m[TRAINER]\033[0m Device found: {device}")

# Unsloth only supports `cuda`
if device != "cuda":
    print("\033[1;91m[ERROR]\033[0m Unsloth only supports GPU training. Exiting.")
    exit()

save_directory = "./aeon"
model_name = "unsloth/SmolLM2-360M"

# Load model and tokenizer using Unsloth's FastLanguageModel
# The max_seq_length is important for memory optimization
# and the dtype should match your hardware
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048, # Adjust as needed
    dtype=torch.float16, # Use torch.float16 or torch.bfloat16
    load_in_4bit=True, # Use 4-bit quantization for lower memory usage
)

# Unsloth's `FastLanguageModel` automatically sets up the chat format
# if the model supports it. It also automatically applies the LoRA
# configuration. We just need to define which layers to apply it to.
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Rank of the LoRA matrices
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=2048,
)

def format_aeon(example, tokenizer):
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

    # Manually define a simple chat template
    prompt_template = (
        "### System:\n{system_prompt}\n\n"
        "### User:\n{user_prompt}\n\n"
        "### Assistant:\n{assistant_prompt}"
    )
    
    formatted_text = prompt_template.format(
        system_prompt=system_prompt,
        user_prompt=example['instruction'],
        assistant_prompt=example['response']
    )
    
    return {'text': formatted_text}


try:
    aeon_train_data = load_dataset('gustavokuklinski/aeon', split='train')
    
    # Use lambda to pass the tokenizer to the mapping function
    aeon_train_ds = aeon_train_data.map(lambda x: format_aeon(x, tokenizer), remove_columns=aeon_train_data.column_names)

    print(f"\033[1;32m[DATASET]:\033[0m Dataset lengths:")
    print(f"\033[1;32m[DATASET]:\033[0m Training set: {len(aeon_train_ds)} examples")

except FileNotFoundError as e:
    print(f"\033[1;91m[ERROR]\033[0m Dataset file not found. Please ensure the paths are correct.")
    print(f"\033[1;91m[ERROR]\033[0m {e}")
    exit()

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True, # Unsloth uses fp16 or bf16
    bf16=False,
    logging_steps=10,
    optim="adamw_8bit", # Unsloth recommends this for memory efficiency
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    report_to=[],
)

# Pass the model and tokenizer to SFTTrainer.
# The `max_seq_length` argument for `SFTTrainer` is crucial
# and must match the one used during model loading.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=aeon_train_ds,
    dataset_text_field="text", # SFTTrainer needs to know the column with text
    max_seq_length=2048, # Must match the value used for FastLanguageModel
    packing=False, # Packing can be enabled for more efficient training
    args=training_args,
)

trainer.train()

# Unsloth has a custom save function that saves the LoRA adapters
model.save_pretrained_merged(save_directory, tokenizer=tokenizer, save_method = "json",)

# You can also save the adapters separately
model.save_pretrained_finetune(save_directory)
tokenizer.save_pretrained(save_directory)