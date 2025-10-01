import os
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Suppress Hugging Face warnings for cleaner output
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# --- CONFIGURATION ---
MODEL_DIR = "./aeon/raw_llm"
#MODEL_DIR = "./aeon/finetuned_llm"
MAX_NEW_TOKENS = 180
REPETITION_PENALTY = 1.2
TEMPERATURE = 0.7
TRIM_LENGTH = 280
ASSISTANT_COLOR = "\033[1;32m"
USER_COLOR = "\033[1;34m"
RESET_COLOR = "\033[0m"

SYSTEM_MESSAGE = (
    "You are a helpful and friendly AI assistant named Aeon."
)
USER_PREFIX = "\n<|user|>\n"
MODEL_PREFIX = "\n<|assistant|>\n"

def load_and_chat():
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}")
        print("Please ensure you have completed both pre-training (pretrain.py -> /raw_llm)"
              "and instruction fine-tuning (finetune.py -> /finetuned_llm).")
        return

    print("\033[1;93m[BOOT]\033[0m Loading model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token 

        MAX_CONTEXT = model.config.n_positions
        
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        return

    print("\033[38;5;160m___________________________________________DEBUG___\033[0m")
    print("")
    print("\033[38;5;160m      ###       #######     #######    ###     ### \033[0m")
    print("\033[38;5;160m    ### ###     ##        ###     ###  ######  ### \033[0m")
    print("\033[38;5;160m   ###   ###    #######   ###     ###  ###  ## ### \033[0m")
    print("\033[38;5;160m  ###     ###   ##        ###     ###  ###   ##### \033[0m")
    print("\033[38;5;160m ##         ##  #######     #######    ###     ### \033[0m")
    print("\033[38;5;160m_Aeon LLM Chat Initialize__________________________\033[0m")
    print("")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[LLM]\033[0m: {MODEL_DIR}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[PARAMS]\033[0m: {model.num_parameters():,}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[DEVICE]\033[0m: {device}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[MAX TOKENS]\033[0m: {MAX_NEW_TOKENS}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[MAX CONTEXT]\033[0m: {MAX_CONTEXT}")
    print(f"\033[1;33m[CMD]\033[0m Enter '/quit' or '/exit' to end the session.")

    chat_history_text = SYSTEM_MESSAGE

    while True:
        try:
            user_input = input(f"{USER_COLOR}{USER_PREFIX.strip()}:{RESET_COLOR} ")

            if user_input.lower() in ['/quit', '/exit']:
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
                
                print(f"\033[1;34m[INFO]\033[0m [Context Truncated: History trimmed to {len(trimmed_history_tokens)} tokens to fit {MAX_CONTEXT} context window.]")

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
                    #max_length=MAX_CONTEXT 
                )

            output = tokenizer.decode(chat_history_ids[0], skip_special_tokens=False)
            
            chat_history_text = output[:TRIM_LENGTH].strip()

            model_response_start = output.rfind(MODEL_PREFIX) + len(MODEL_PREFIX)
            model_response = output[model_response_start:].strip()
            
            eos_token_str = tokenizer.decode(tokenizer.eos_token_id)
            if model_response.endswith(eos_token_str):
                model_response = model_response[:model_response.rfind(eos_token_str)].strip()

            display_response = model_response.replace('\n', ' ').replace('\t', ' ')


            if len(display_response) > TRIM_LENGTH:
                display_response = display_response[:TRIM_LENGTH].strip() + f'... etc...\n\n\033[1;93m[SYS]\033[0m {TRIM_LENGTH}+ CHARS.{RESET_COLOR}'

            print(f"{ASSISTANT_COLOR}{MODEL_PREFIX.strip()}:{RESET_COLOR} {display_response}")

        except Exception as e:
            print(f"\033[1;91m[ERRR]\033[0m An error occurred during chat: {e}")
            break

if __name__ == "__main__":
    load_and_chat()
