import os
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Suppress Hugging Face warnings for cleaner output
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# --- CONFIGURATION ---
# FIX: Changed directory to match the output of pretrain_raw.py
MODEL_DIR = "./aeon_finetuned" # Directory where the fine-tuned model is saved
MAX_NEW_TOKENS = 512 # Maximum length for the model's reply
REPETITION_PENALTY = 1.5 # Increased to prevent repetitive non-sense
TEMPERATURE = 0.7 # Lowered for more coherent, less random responses

# Instruction format delimiters (MUST match the finetuning script)
USER_PREFIX = "\n<|user|>\n"
MODEL_PREFIX = "\n<|model|>\n"
SYSTEM_MESSAGE = "You are Aeon, an AI trained to be a creative and knowledgeable conversational assistant."

def load_and_chat():
    """Loads the model and tokenizer, then starts the interactive chat loop."""
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}")
        print("Please ensure you have completed both pre-training (Stage 1) and instruction fine-tuning (Stage 2).")
        return

    print("-> Loading model and tokenizer...")
    try:
        # Check for CUDA availability first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer and model (the config.json inside MODEL_DIR will define the architecture)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
        model.eval()
        
        # Ensure the pad token is set for the model generation, aligning with pretrain_raw.py changes
        if tokenizer.pad_token is None:
            # We use the EOS token as the PAD token, as done in training
            tokenizer.pad_token = tokenizer.eos_token 

        # Get the model's max context length (n_positions)
        MAX_CONTEXT = model.config.n_positions # Should be 1024 based on pretrain_raw.py
        
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return

    print("-" * 50)
    print(f"Aeon LLM Chat Initialized (Parameters: {model.num_parameters():,})")
    print(f"Device: {device}")
    print(f"Max Context Length: {MAX_CONTEXT}")
    print(f"Enter 'quit' or 'exit' to end the session.")
    print("-" * 50)

    # Start conversation history with the System message
    chat_history_text = SYSTEM_MESSAGE

    while True:
        try:
            user_input = input(f"{USER_PREFIX.strip()}: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            # 1. Prepare the NEW turn tokens to determine required space
            new_turn_text = USER_PREFIX + user_input + MODEL_PREFIX
            new_turn_tokens = tokenizer.encode(new_turn_text, truncation=True)
            
            # Calculate the minimum space needed for the new turn and the model's generated reply
            # Adding a small safety buffer (5 tokens)
            required_space = len(new_turn_tokens) + MAX_NEW_TOKENS + 5 
            
            # 2. Check and Truncate the Existing History (Context Management)
            history_tokens = tokenizer.encode(chat_history_text, truncation=False)
            
            # Determine the maximum number of history tokens we can keep
            max_history_tokens = MAX_CONTEXT - required_space

            if len(history_tokens) > max_history_tokens:
                # Calculate system message length to ensure it's always kept
                system_message_tokens = tokenizer.encode(SYSTEM_MESSAGE, truncation=True)
                system_len = len(system_message_tokens)

                # Find the starting index for the history we want to keep
                # We slice the history to keep only the most recent tokens, 
                # ensuring the system message is at least included.
                start_index = max(system_len, len(history_tokens) - max_history_tokens)
                
                # Trim the history tokens
                trimmed_history_tokens = history_tokens[start_index:]
                
                # Decode the trimmed history back to text
                chat_history_text = tokenizer.decode(trimmed_history_tokens, skip_special_tokens=False)
                
                print(f"[Context Truncated: History trimmed to {len(trimmed_history_tokens)} tokens to fit {MAX_CONTEXT} context window.]")


            # 3. Build the final prompt and tokenize
            prompt = chat_history_text + new_turn_text
            encoded_prompt = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
            new_input_ids = encoded_prompt['input_ids']
            attention_mask = encoded_prompt['attention_mask']
            
            # 4. Generate response
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

            # 5. Process Output
            output = tokenizer.decode(chat_history_ids[0], skip_special_tokens=False)
            
            # Update the history text for the next turn
            chat_history_text = output

            # Extract and print only the new model response
            model_response_start = output.rfind(MODEL_PREFIX) + len(MODEL_PREFIX)
            model_response = output[model_response_start:].strip()
            
            # Remove any trailing EOS tokens (which appear as the pad token id)
            eos_token_str = tokenizer.decode(tokenizer.eos_token_id)
            if model_response.endswith(eos_token_str):
                model_response = model_response[:model_response.rfind(eos_token_str)].strip()

            print(f"{MODEL_PREFIX.strip()}: {model_response}")

        except Exception as e:
            print(f"An error occurred during chat: {e}")
            break

if __name__ == "__main__":
    load_and_chat()
