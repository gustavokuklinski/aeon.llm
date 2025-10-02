import os
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

TARGET_PARAMS = 100 # Set LLM Parameters
TRAIN_EPOCH = 10 # Set how much Epochs for pre train
SET_CONTEXT = 2048 # Set context size length
TRAIN_OUTPUT = "./aeon/raw_llm/output" # Output train checkpoints
OUTPUT_MODEL_DIR = "./aeon/raw_llm" # Output LLM
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
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


TRAINING_ARGS = TrainingArguments(
    output_dir=TRAIN_OUTPUT,
    num_train_epochs=TRAIN_EPOCH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), 
    push_to_hub=False,
    optim="adamw_torch",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1
)


def load_and_prepare_data():
    print("\033[1;36m[INFO]\033[0m Loading and preparing RAW corpus data...")
    try:
        print(f"\033[1;36m[INFO]\033[0m Loading Aeon Books...")
        ds1 = load_dataset('gustavokuklinski/aeon-books', split='train')

        print(f"\033[1;36m[INFO]\033[0m Loading Tiny Shakespeare from URL...")
        ds_shakespeare = load_dataset('text', data_files={'train': TINY_SHAKESPEARE_URL}, split='train')

        
        
        
        raw_datasets = concatenate_datasets([ds_shakespeare, ds1])

    except Exception as e:
        print(f"Error loading Hugging Face datasets: {e}")
        raise RuntimeError("Failed to load required datasets. Please check names and connectivity.")

    print("\033[1;36m[INFO]\033[0m Splitting data into 95% train and 5% validation sets...")
    
    split_datasets = raw_datasets.train_test_split(test_size=0.05, seed=42)
    
    raw_train_ds = split_datasets['train']
    raw_eval_ds = split_datasets['test']


    def combine_columns(examples):
        return {'text': examples['text']}

    raw_train_ds = raw_train_ds.map(
        combine_columns,
        batched=True,
        remove_columns=raw_train_ds.column_names, 
    )
    raw_eval_ds = raw_eval_ds.map(
        combine_columns,
        batched=True,
        remove_columns=raw_eval_ds.column_names, 
    )

    temp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    

    def count_tokens(examples):
        return {'token_count': [len(temp_tokenizer.encode(t, truncation=True)) for t in examples['text']]}
    
    tokenized_size_datasets = raw_train_ds.map(
        count_tokens,
        batched=True,
        num_proc=os.cpu_count() or 4
    )

    total_examples = len(raw_train_ds) + len(raw_eval_ds)
    total_tokens = sum(tokenized_size_datasets['token_count'])
    
    print(f"\n\033[1;36m[INFO]\033[0m DATASET STATISTICS")
    print(f"\033[1;36m[INFO]\033[0m Total Examples (Documents): {total_examples:,}")
    print(f"\033[1;36m[INFO]\033[0m Training Examples: {len(raw_train_ds):,}")
    print(f"\033[1;36m[INFO]\033[0m Validation Examples: {len(raw_eval_ds):,}")
    print(f"\033[1;36m[INFO]\033[0m Total Raw Tokens (Approx.): {total_tokens:,}")
    print(f"\033[1;36m[INFO]\033[0m Average Tokens per Example: {total_tokens / len(raw_train_ds):,.0f} (Train only)") # Adjusted calculation
    
    return DatasetDict({'train': raw_train_ds, 'eval': raw_eval_ds})


def tokenize_and_chunk(datasets, tokenizer, block_size):

    print(f"\033[1;36m[INFO]\033[0m Tokenizing and chunking data into blocks of {block_size}...")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    def process_split(split_dataset):
        tokenized_datasets = split_dataset.map(
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
            num_proc=os.cpu_count() or 4
        )
        return lm_datasets

    if isinstance(datasets, DatasetDict):
        lm_datasets_dict = DatasetDict()
        for key in datasets:
            lm_datasets_dict[key] = process_split(datasets[key])
        return lm_datasets_dict
    else:
        return process_split(datasets)


def train_llm(split_datasets, training_args, target_m_params):
    print("\n\033[1;36m[INFO]\033[0m Initializing GPT-2 tokenizer and custom model...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = get_model_config(target_m_params)
    
    config.vocab_size = len(tokenizer)
    
    model = GPT2LMHeadModel(config)

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"\n\033[1;36m[INFO]\033[0m Model initialized with {model.num_parameters():,} parameters.") 

    lm_datasets = tokenize_and_chunk(split_datasets, tokenizer, config.n_positions)
    
    lm_train_dataset = lm_datasets['train']
    lm_eval_dataset = lm_datasets['eval']


    print(f"\033[1;36m[INFO]\033[0m TRAINING INPUT SIZE (AFTER CHUNKING)")
    print(f"\033[1;36m[INFO]\033[0m Total Training Blocks (Steps/Epoch): {len(lm_train_dataset):,}")
    print(f"\033[1;36m[INFO]\033[0m Total Validation Blocks: {len(lm_eval_dataset):,}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    print("\n\033[1;36m[INFO]\033[0m  Starting pre-training process...")
    trainer.train()

    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"\n\033[1;32m[SUCCESS]\033[0m Training complete. Model and tokenizer saved to {OUTPUT_MODEL_DIR}/")


if __name__ == "__main__":
    print("\033[38;5;160m___________________________________________________\033[0m")
    print("")
    print("\033[38;5;160m      ###       #######     #######    ###     ### \033[0m")
    print("\033[38;5;160m    ### ###     ##        ###     ###  ######  ### \033[0m")
    print("\033[38;5;160m   ###   ###    #######   ###     ###  ###  ## ### \033[0m")
    print("\033[38;5;160m  ###     ###   ##        ###     ###  ###   ##### \033[0m")
    print("\033[38;5;160m ##         ##  #######     #######    ###     ### \033[0m")
    print("\033[38;5;160m_PRE-TRAIN_________________________________________\033[0m")
    print("")

    if not torch.cuda.is_available():
        print("\033[1;93m[PRE TRAINER]\033[0m GPU not detected. Training will run on CPU. Setting fp16=False.")
        TRAINING_ARGS.fp16 = False 
        
    try:
        split_ds = load_and_prepare_data() 
        train_llm(split_ds, TRAINING_ARGS, TARGET_PARAMS) 

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")