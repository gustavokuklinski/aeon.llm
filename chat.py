import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

# --- Configuration ---
MODEL_PATH = "./aeon_finetuned"
MAX_LENGTH = 150  # How long the model's response can be
TEMPERATURE = 0.7 # Controls randomness (higher=more creative)
TOP_K = 50        # Considers only the top 50 likely next tokens
TOP_P = 0.95      # Takes tokens summing up to 95% probability

def chat_with_model():
    """Loads the fine-tuned model and provides an interactive chat interface."""
    
    print("-> Loading trained model and tokenizer for inference...")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory '{MODEL_PATH}' not found.")
        print("Please ensure your training script successfully ran and created this folder.")
        return

    # Determine device (use GPU if available)
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

        # Create the text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            # Ensure the model is set to evaluation mode
            framework="pt" 
        )

        print("\n--- Model Chat Ready ---")
        print("Model loaded successfully. Start typing your prompts.")
        print("Type 'quit' or 'exit' to end the session.")
        print("-" * 30)

        # Interactive loop
        while True:
            prompt = input("You: ")
            
            if prompt.lower() in ["quit", "exit"]:
                print("Exiting chat. Goodbye!")
                break

            if not prompt.strip():
                continue

            # Generate the response
            # Note: The output is a continuation of the input prompt
            output = generator(
                prompt,
                max_length=len(tokenizer.encode(prompt)) + MAX_LENGTH,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=1.2, # Helps prevent repetitive text
                num_return_sequences=1,
            )

            # Extract and clean the generated text
            generated_text = output[0]['generated_text']
            
            # Since the model returns the prompt + completion, we only want the completion
            completion = generated_text[len(prompt):].strip()
            
            # Display the result
            print(f"LLM: {completion}\n")

    except Exception as e:
        print(f"\nAn error occurred during model loading or generation: {e}")
        print("Check if dependencies are installed (pip install transformers torch)")


if __name__ == "__main__":
    chat_with_model()
