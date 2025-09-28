import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

TARGET_PARAMS = 300 # Set LLM Parameters
TRAIN_EPOCH = 6 # Set how much Epochs for pre train
SET_CONTEXT = 1024 # Set context size length
TRAIN_OUTPUT = "./aeon/checkpoint_output" # Output train checkpoints
OUTPUT_MODEL_DIR = "./aeon/raw_llm" # Output LLM

MODEL_PRESETS = {
    1:   {'n_layer': 2,  'n_head': 2,  'n_embd': 16},   # ~0.9M Parameters
    5:   {'n_layer': 2,  'n_head': 2,  'n_embd': 96},   # ~5.5M Parameters
    10:  {'n_layer': 4,  'n_head': 4,  'n_embd': 168},  # ~10.3M Parameters
    20:  {'n_layer': 6,  'n_head': 6,  'n_embd': 252},  # ~17.3M Parameters
    80:  {'n_layer': 10, 'n_head': 8,  'n_embd': 512},  # ~84.1M Parameters
    100: {'n_layer': 12, 'n_head': 8,  'n_embd': 640},  # ~91.5M Parameters
    150: {'n_layer': 16, 'n_head': 10, 'n_embd': 768},  # ~163.7M Parameters
    200: {'n_layer': 18, 'n_head': 12, 'n_embd': 768},  # ~200.7M Parameters
    250: {'n_layer': 20, 'n_head': 16, 'n_embd': 768},  # ~240.2M Parameters
    300: {'n_layer': 24, 'n_head': 16, 'n_embd': 768},  # ~288.7M Parameters
    350: {'n_layer': 26, 'n_head': 16, 'n_embd': 768},  # ~313.2M Parameters
    400: {'n_layer': 28, 'n_head': 16, 'n_embd': 768},  # ~337.7M Parameters
    450: {'n_layer': 30, 'n_head': 16, 'n_embd': 768},  # ~362.2M Parameters
    500: {'n_layer': 32, 'n_head': 16, 'n_embd': 768},  # ~386.7M Parameters
}

def get_model_config(target_m_params: int) -> GPT2Config:

    config_params = MODEL_PRESETS.get(target_m_params)
    
    if not config_params:
        raise ValueError(f"Target parameter size {target_m_params}M not found in presets.")

    print(f"\033[1;36m[INFO]\033[0m Loading model config for target size: {target_m_params} Million parameters.")
    return GPT2Config(
        vocab_size=50257,
        n_layer=config_params['n_layer'],
        n_head=config_params['n_head'],
        n_embd=config_params['n_embd'],
        n_positions=SET_CONTEXT,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

# Training Parameters (These should be safe for up to 300M on a T4 GPU)
TRAINING_ARGS = TrainingArguments(
    output_dir=TRAIN_OUTPUT,
    num_train_epochs=TRAIN_EPOCH,
    per_device_train_batch_size=1, # Reduced for memory conservation
    gradient_accumulation_steps=32, # Maintained at 32 for effective batch size of 32
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), 
    push_to_hub=False,
    optim="adamw_torch",
    report_to="none"
)


def load_and_prepare_data():
    print("\033[1;36m[INFO]\033[0m Loading and preparing RAW corpus data...")
    try:
        ds1 = load_dataset('gustavokuklinski/aeon-books', split='train')
        ds2 = load_dataset('gustavokuklinski/aeon-movies-tv', split='train')

        raw_datasets = concatenate_datasets([ds1, ds2])

    except Exception as e:
        print(f"Error loading Hugging Face datasets: {e}")
        raise RuntimeError("Failed to load required datasets. Please check names and connectivity.")


    def combine_columns(examples):
        combined_text = [f"{title}. {text}" for title, text in zip(examples['title'], examples['text'])]
        return {'text': combined_text}

    raw_datasets = raw_datasets.map(
        combine_columns,
        batched=True,
        remove_columns=raw_datasets.column_names, 
    )

    temp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def count_tokens(examples):
        return {'token_count': [len(temp_tokenizer.encode(t, truncation=True)) for t in examples['text']]}
    
    tokenized_size_datasets = raw_datasets.map(
        count_tokens,
        batched=True,
        num_proc=os.cpu_count() or 4
    )

    total_examples = len(raw_datasets)
    total_tokens = sum(tokenized_size_datasets['token_count'])
    
    print(f"\n\033[1;36m[INFO]\033[0m DATASET STATISTICS")
    print(f"\033[1;36m[INFO]\033[0m Total Examples (Documents): {total_examples:,}")
    print(f"\033[1;36m[INFO]\033[0m Total Raw Tokens (Approx.): {total_tokens:,}")
    print(f"\033[1;36m[INFO]\033[0m Average Tokens per Example: {total_tokens / total_examples:,.0f}")

    return raw_datasets


def tokenize_and_chunk(datasets, tokenizer, block_size):

    print(f"\033[1;36m[INFO]\033[0m Tokenizing and chunking data into blocks of {block_size}...")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=block_size)


    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() or 4,
        remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        total_length = (total_length // block_size) * block_size 
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count() or 4,
        desc=f"\033[1;36m[INFO]\033[0m Grouping texts into blocks of {block_size}",
    )

    return lm_datasets


def train_llm(train_dataset, training_args, target_m_params):
    print("\n\033[1;36m[INFO]\033[0m Initializing GPT-2 tokenizer and custom model...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None or tokenizer.eos_token is None:
        new_special_tokens = {
            'bos_token': '[bos]', 
            'eos_token': '[eos]', 
            'pad_token': '[pad]'
        }
        tokenizer.add_special_tokens(new_special_tokens)
    
    config = get_model_config(target_m_params)
    
    config.vocab_size = len(tokenizer)
    
    model = GPT2LMHeadModel(config)

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"\n\033[1;36m[INFO]\033[0m Model initialized with {model.num_parameters():,} parameters.") 

    lm_train_dataset = tokenize_and_chunk(train_dataset, tokenizer, config.n_positions)


    print(f"\033[1;36m[INFO]\033[0m TRAINING INPUT SIZE (AFTER CHUNKING)")
    print(f"\033[1;36m[INFO]\033[0m Total Training Blocks (Steps/Epoch): {len(lm_train_dataset):,}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    print("\n\033[1;36m[INFO]\033[0m  Starting pre-training process...")
    trainer.train()

    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"\n\033[1;32m[SUCCESS]\033[0m Training complete. Model and tokenizer saved to {OUTPUT_MODEL_DIR}/")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("\033[1;93m[PRE TRAINER]\033[0m GPU not detected. Training will run on CPU. Setting fp16=False.")
        TRAINING_ARGS.fp16 = False 
        
    try:
        train_ds = load_and_prepare_data()
        train_llm(train_ds, TRAINING_ARGS, TARGET_PARAMS) 

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
