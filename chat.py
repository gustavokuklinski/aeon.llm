import os
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Suppress Hugging Face warnings for cleaner output
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# --- CONFIGURATION ---
MODEL_DIR = "./aeon/aeon_finetuned"
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.5
TEMPERATURE = 0.7

USER_PREFIX = "\n<|im_end|><|im_start|>user\n"
MODEL_PREFIX = "\n<|im_end|><|im_start|>assistant\n"
SYSTEM_MESSAGE = "You are Aeon, an AI trained to be a creative and knowledgeable conversational assistant."

def load_and_chat():
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}")
        print("Please ensure you have completed both pre-training (Stage 1) and instruction fine-tuning (Stage 2).")
        return

    print("-> Loading model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 

        MAX_CONTEXT = model.config.n_positions
        
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return

    print("-" * 50)
    print(f"Aeon LLM Chat Initialized (Parameters: {model.num_parameters():,})")
    print(f"Device: {device}")
    print(f"Max Context Length: {MAX_CONTEXT}")
    print(f"Enter 'quit' or 'exit' to end the session.")
    print("-" * 50)

    chat_history_text = SYSTEM_MESSAGE

    while True:
        try:
            user_input = input(f"{USER_PREFIX.strip()}: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            new_turn_text = USER_PREFIX + user_input + MODEL_PREFIX
            new_turn_tokens = tokenizer.encode(new_turn_text, truncation=True)
            
            required_space = len(new_turn_tokens) + MAX_NEW_TOKENS + 5 
            
            history_tokens = tokenizer.encode(chat_history_text, truncation=False)
            
            max_history_tokens = MAX_CONTEXT - required_space

            if len(history_tokens) > max_history_tokens:
                system_message_tokens = tokenizer.encode(SYSTEM_MESSAGE, truncation=True)
                system_len = len(system_message_tokens)

                start_index = max(system_len, len(history_tokens) - max_history_tokens)

                trimmed_history_tokens = history_tokens[start_index:]

                chat_history_text = tokenizer.decode(trimmed_history_tokens, skip_special_tokens=False)
                
                print(f"[Context Truncated: History trimmed to {len(trimmed_history_tokens)} tokens to fit {MAX_CONTEXT} context window.]")

            prompt = chat_history_text + new_turn_text
            encoded_prompt = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
            new_input_ids = encoded_prompt['input_ids']
            attention_mask = encoded_prompt['attention_mask']
            
            with torch.no_grad():
                chat_history_ids = model.generate(
                    new_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=REPETITION_PENALTY,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    max_length=MAX_CONTEXT 
                )

            output = tokenizer.decode(chat_history_ids[0], skip_special_tokens=False)
            
            chat_history_text = output

            model_response_start = output.rfind(MODEL_PREFIX) + len(MODEL_PREFIX)
            model_response = output[model_response_start:].strip()
            
            eos_token_str = tokenizer.decode(tokenizer.eos_token_id)
            if model_response.endswith(eos_token_str):
                model_response = model_response[:model_response.rfind(eos_token_str)].strip()

            print(f"{MODEL_PREFIX.strip()}: {model_response}")

        except Exception as e:
            print(f"An error occurred during chat: {e}")
            break

if __name__ == "__main__":
    load_and_chat()
