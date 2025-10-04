import os
import torch
from transformers import GPT2LMHeadModel,AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# --- CONFIGURATION ---
MODEL_DIR = "./aeon/finetuned_llm"
MAX_NEW_TOKENS = 2048
REPETITION_PENALTY = 1.5
TEMPERATURE = 0.5
TRIM_LENGTH = 2048
ASSISTANT_COLOR = "\033[1;32m"
USER_COLOR = "\033[1;34m"
RESET_COLOR = "\033[0m"

USER_PREFIX = "user"
MODEL_PREFIX = "assistant"

SYSTEM_MESSAGE = (
    "You are a helpful AI assistant"
)

def load_and_chat():
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory not found at {MODEL_DIR}")
        print("Please ensure you have completed both pre-training and fine-tuning.")
        return

    print("\033[1;93m[BOOT]\033[0m Loading model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        # If Using GPT2
        # model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
        # MAX_CONTEXT = model.config.n_positions

        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
        MAX_CONTEXT = model.config.max_position_embeddings

        model.eval()
        
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
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[DEVICE]\033[0m: {device}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[LLM]\033[0m: {MODEL_DIR}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[PARAMETERS]\033[0m: {model.num_parameters():,}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[MAX TOKENS]\033[0m: {MAX_NEW_TOKENS}")
    print(f"\033[1;93m[SYS]\033[0m \033[1;34m[MAX CONTEXT]\033[0m: {MAX_CONTEXT}")
    print(f"\033[1;33m[CMD]\033[0m Enter '/quit' or '/exit' to end the session.")

    chat_history_list: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_MESSAGE}
    ]
    while True:
        try:
            user_input = input(f"{USER_COLOR}{USER_PREFIX}:{RESET_COLOR} ")

            if user_input.lower() in ['/quit', '/exit']:
                break

            chat_history_list.append({"role": "user", "content": user_input})

            prompt_tokens = tokenizer.apply_chat_template(
                chat_history_list, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors='pt'
            )

            if prompt_tokens.shape[1] > MAX_CONTEXT - MAX_NEW_TOKENS - 5:
                system_prompt_str = tokenizer.apply_chat_template([chat_history_list[0]], tokenize=False)
                system_len = len(tokenizer.encode(system_prompt_str))
                
                tokens_to_keep = MAX_CONTEXT - MAX_NEW_TOKENS - 5
                
                trimmed_tokens = torch.cat([
                    prompt_tokens[0, :system_len], 
                    prompt_tokens[0, -(tokens_to_keep - system_len):]
                ]).unsqueeze(0)
                
                prompt_tokens = trimmed_tokens
                print(f"\033[1;34m[INFO]\033[0m [Context Truncated to {prompt_tokens.shape[1]} tokens.]")

            input_ids = prompt_tokens.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            if input_ids.shape[1] > MAX_CONTEXT - MAX_NEW_TOKENS - 5:
                system_prompt_str = tokenizer.apply_chat_template([chat_history_list[0]], tokenize=False)
                system_len = len(tokenizer.encode(system_prompt_str))
                
                tokens_to_keep = MAX_CONTEXT - MAX_NEW_TOKENS - 5
                
                trimmed_ids = torch.cat([
                    input_ids[0, :system_len], 
                    input_ids[0, -(tokens_to_keep - system_len):]
                ]).unsqueeze(0)
                
                input_ids = trimmed_ids
                attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
                
                print(f"\033[1;34m[INFO]\033[0m [Context Truncated to {input_ids.shape[1]} tokens.]")


            with torch.no_grad():
                output_tokens = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=REPETITION_PENALTY,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    
                )

            full_output = tokenizer.decode(output_tokens[0], skip_special_tokens=False)

            assistant_start_tag = f"<|im_start|>{MODEL_PREFIX}\n"
            assistant_end_tag = "<|im_end|>"
            
            start_index = full_output.rfind(assistant_start_tag)
            
            if start_index != -1:
                response_content_start = start_index + len(assistant_start_tag)
                model_response = full_output[response_content_start:].strip()
                
                end_index = model_response.find(assistant_end_tag)
                if end_index != -1:
                    model_response = model_response[:end_index].strip()
                
            else:
                model_response = "Error: Could not parse model response."
                
            chat_history_list.append({"role": "assistant", "content": model_response})
            
            display_response = model_response.replace('\n', ' ').replace('\t', ' ')

            if len(display_response) > TRIM_LENGTH:
                display_response = display_response[:TRIM_LENGTH].strip() + f" \033[38;5;160m[TRUNKED]{RESET_COLOR}'"

            print(f"{ASSISTANT_COLOR}{MODEL_PREFIX}:{RESET_COLOR} {display_response}")

        except Exception as e:
            print(f"\033[1;91m[ERRR]\033[0m An error occurred during chat: {e}")
            break

if __name__ == "__main__":
    load_and_chat()
