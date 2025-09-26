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
# Suppress the deprecation warnings from Hugging Face libraries
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# --- 1. CONFIGURATION AND HYPERPARAMETERS ---

# Define the dataset paths on the Hugging Face Hub
CSV_FILE_1 = "gustavokuklinski/aeon-books"
CSV_FILE_2 = "gustavokuklinski/aeon-movies-tv"
OUTPUT_MODEL_DIR = "./aeon_llm" # Directory to save the pre-trained model

# --- CHOOSE YOUR MODEL SIZE HERE (in Millions of Parameters) ---
# Select from keys in MODEL_PRESETS (e.g., 10, 20, 80, 250, 500)
TARGET_PARAMS = 10 

# --- MODEL PRESETS ---
# These configurations are based on standard GPT-2 architectures, offering predictable parameter counts.
# n_embd (Hidden Dimension) must be divisible by n_head.
MODEL_PRESETS = {
    10:  {'n_layer': 4,  'n_head': 4,  'n_embd': 168},  # Actual ~10.3M
    20:  {'n_layer': 6,  'n_head': 6,  'n_embd': 252},  # Actual ~17.3M
    80:  {'n_layer': 10, 'n_head': 8,  'n_embd': 512},  # Actual ~84.1M
    100: {'n_layer': 12, 'n_head': 8,  'n_embd': 640},  # Actual ~91.5M
    150: {'n_layer': 16, 'n_head': 10, 'n_embd': 768},  # Actual ~163.7M
    200: {'n_layer': 18, 'n_head': 12, 'n_embd': 768},  # Actual ~200.7M
    250: {'n_layer': 20, 'n_head': 16, 'n_embd': 768},  # Actual ~240.2M
    300: {'n_layer': 24, 'n_head': 16, 'n_embd': 768},  # Actual ~288.7M (Close to GPT-2 Small)
    350: {'n_layer': 26, 'n_head': 16, 'n_embd': 768},  # Actual ~313.2M
    400: {'n_layer': 28, 'n_head': 16, 'n_embd': 768},  # Actual ~337.7M
    450: {'n_layer': 30, 'n_head': 16, 'n_embd': 768},  # Actual ~362.2M
    500: {'n_layer': 32, 'n_head': 16, 'n_embd': 768},  # Actual ~386.7M
}

def get_model_config(target_m_params: int) -> GPT2Config:
    """Retrieves the configuration for the target parameter size."""
    config_params = MODEL_PRESETS.get(target_m_params)
    
    if not config_params:
        raise ValueError(f"Target parameter size {target_m_params}M not found in presets.")

    print(f"-> Loading model config for target size: {target_m_params} Million parameters.")
    return GPT2Config(
        vocab_size=50257, # Placeholder: will be set dynamically
        n_layer=config_params['n_layer'],
        n_head=config_params['n_head'],
        n_embd=config_params['n_embd'],
        n_positions=512,  # Maximum sequence length (context window)
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

# Training Parameters (These should be safe for up to 300M on a T4 GPU)
TRAINING_ARGS = TrainingArguments(
    output_dir="./aeon_output",
    num_train_epochs=4, # 4 epochs (Training time will be significant)
    per_device_train_batch_size=1, # Reduced for memory conservation
    gradient_accumulation_steps=32, # Maintained at 32 for effective batch size of 32
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use mixed precision if a GPU is detected
    push_to_hub=False,
    optim="adamw_torch",
    report_to="none" # Prevents WandB prompt in Colab
)

# --- 2. DATA LOADING AND TOKENIZER SETUP ---

def load_and_prepare_data(dataset_name1, dataset_name2):
    """Loads datasets from Hugging Face Hub, merges them, and prepares for tokenization."""
    print("-> Loading and preparing RAW corpus data...")
    try:
        # Load the two datasets directly from the Hugging Face Hub (assuming 'train' split)
        ds1 = load_dataset(dataset_name1, split='train')
        ds2 = load_dataset(dataset_name2, split='train')

        raw_datasets = concatenate_datasets([ds1, ds2])

    except Exception as e:
        print(f"Error loading Hugging Face datasets: {e}")
        raise RuntimeError("Failed to load required datasets. Please check names and connectivity.")


    # Combine 'title' and 'text' keys into a single 'text' string for training
    def combine_columns(examples):
        combined_text = [f"{title}. {text}" for title, text in zip(examples['title'], examples['text'])]
        return {'text': combined_text}

    raw_datasets = raw_datasets.map(
        combine_columns,
        batched=True,
        remove_columns=raw_datasets.column_names, 
    )

    # --- Dataset parameter logging ---
    # Temporarily load tokenizer to measure data size in tokens
    temp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Function to count tokens in a batch of texts
    def count_tokens(examples):
        # Use truncation=True to suppress warnings from documents longer than 1024
        return {'token_count': [len(temp_tokenizer.encode(t, truncation=True)) for t in examples['text']]}
    
    # Map token counting function (removing original text column to save memory)
    tokenized_size_datasets = raw_datasets.map(
        count_tokens,
        batched=True,
        num_proc=os.cpu_count() or 4,
        desc="Counting tokens in the dataset",
    )

    # Calculate total and average
    total_examples = len(raw_datasets)
    total_tokens = sum(tokenized_size_datasets['token_count'])
    
    print("-" * 40)
    print(f"| DATASET STATISTICS")
    print("-" * 40)
    print(f"| Total Examples (Documents): {total_examples:,}")
    print(f"| Total Raw Tokens (Approx.): {total_tokens:,}")
    print(f"| Average Tokens per Example: {total_tokens / total_examples:,.0f}")
    print("-" * 40)

    # Split the dataset into train and validation sets (95% train, 5% validation)
    raw_datasets = raw_datasets.train_test_split(test_size=0.05, seed=42)
    return raw_datasets['train'], raw_datasets['test']


def tokenize_and_chunk(datasets, tokenizer, block_size):
    """Tokenizes text and groups sequences of tokens into fixed-size blocks."""
    print(f"-> Tokenizing and chunking data into blocks of {block_size}...")

    def tokenize_function(examples):
        # Crucial: Truncate long documents to match the model's max sequence length (512)
        return tokenizer(examples["text"], truncation=True, max_length=block_size)

    # Tokenize the dataset using multi-processing for speed
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() or 4,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    # Main data processing function: groups sequences into blocks of block_size
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Corrected bug: Use block_size instead of undefined block_length
        total_length = (total_length // block_size) * block_size 
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply the grouping function
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count() or 4,
        desc=f"Grouping texts into blocks of {block_size}",
    )

    return lm_datasets


# --- 3. MODEL INITIALIZATION AND TRAINING ---

def train_llm(train_dataset, eval_dataset, training_args, target_m_params):
    """Initializes the model and the Trainer, then starts the training loop."""
    print("-> Initializing GPT-2 tokenizer and custom model...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Ensure PAD token is set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Get the configuration based on the selected size
    config = get_model_config(target_m_params)
    
    # FIX: Dynamically update the model config to match the exact tokenizer vocabulary size
    config.vocab_size = len(tokenizer)
    
    model = GPT2LMHeadModel(config)
    # Print the actual parameter count
    print(f"\nModel initialized with {model.num_parameters():,} parameters.") 

    lm_train_dataset = tokenize_and_chunk(train_dataset, tokenizer, config.n_positions)
    lm_eval_dataset = tokenize_and_chunk(eval_dataset, tokenizer, config.n_positions)

    # --- Log final training dataset size after chunking ---
    print("-" * 40)
    print(f"| TRAINING INPUT SIZE (AFTER CHUNKING)")
    print("-" * 40)
    print(f"| Total Training Blocks (Steps/Epoch): {len(lm_train_dataset):,}")
    print(f"| Total Evaluation Blocks: {len(lm_eval_dataset):,}")
    print("-" * 40)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # Causal Language Modeling
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    # Start training
    print("\n-> Starting pre-training process...")
    trainer.train()

    # Save the final model and tokenizer
    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"\nTraining complete. Model and tokenizer saved to {OUTPUT_MODEL_DIR}/")


# --- 4. EXECUTION ---

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not detected. Training will run on CPU. Setting fp16=False.")
        TRAINING_ARGS.fp16 = False 
        
    try:
        train_ds, eval_ds = load_and_prepare_data(CSV_FILE_1, CSV_FILE_2)
        # Pass TARGET_PARAMS to the train function
        train_llm(train_ds, eval_ds, TRAINING_ARGS, TARGET_PARAMS) 

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please verify your dataset names and internet connectivity.")
