import os
import subprocess
import sys

# --- Configuration ---
# This is the directory where your Hugging Face model files are located.
# You MUST change this to the correct path for your model.
# For example: save_directory = "path/to/your/huggingface-model"
save_directory = "aeon" 

# The desired output filename for the GGUF model.
gguf_output_filename = "gguf/output-aeon.gguf"

# The output type for the GGUF model (e.g., f16, q4_0, q8_0).
output_type = "f16"

# --- Script Logic ---

def run_command(command):
    """Executes a shell command and raises an exception if it fails."""
    try:
        print(f"\nExecuting: {' '.join(command)}")
        # Using capture_output=True and text=True to get stdout/stderr
        # We use sys.executable to ensure we're using the same python interpreter
        if command[0] == 'pip' or command[0] == 'python':
            command[0] = sys.executable
            if command[0] == sys.executable and command[1] != '-m':
                 command.insert(1, '-m')
        
        if command[1] == '-m' and command[0] == sys.executable:
            if command[2] == 'pip': # pip install
                command = [sys.executable, '-m', 'pip'] + command[2:]
            else: # python script.py
                command = [sys.executable] + command[2:]


        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        # Stream the output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()

        if process.returncode != 0:
            print(f"\nError: Command failed with exit code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, command)
            
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is it installed and in your PATH?")
        raise
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing: {' '.join(e.cmd)}")
        # The output has already been printed, so we just exit
        sys.exit(1)


def main():
    """Main function to clone llama.cpp and convert the model."""
    
    # Check if the model directory exists before proceeding
    if not os.path.exists(save_directory):
        print(f"Error: The model directory '{save_directory}' does not exist.")
        print("Please download your Hugging Face model files and place them in that directory,")
        print("or update the 'save_directory' variable in this script.")
        # Create a dummy directory to guide the user
        os.makedirs(save_directory, exist_ok=True)
        print(f"Created a placeholder directory at '{save_directory}'.")
        sys.exit(1)


    # 1. Clone llama.cpp repository if it doesn't exist
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp repository...")
        run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
        print("llama.cpp cloned successfully.")
         # 2. Install llama.cpp Python dependencies
        print("Installing llama.cpp Python dependencies...")
        requirements_path = os.path.join("llama.cpp", "requirements.txt")
        run_command(["pip", "install", "-r", requirements_path])
        print("Dependencies installed successfully.")
    else:
        print("llama.cpp directory already exists, skipping clone.")

   

    # 3. Convert the model to GGUF format
    print(f"Converting model to GGUF format: {gguf_output_filename}...")
    convert_script_path = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
    
    # Check if the output file already exists
    if os.path.exists(gguf_output_filename):
        print(f"Output file '{gguf_output_filename}' already exists. Skipping conversion.")
    else:
        run_command([
            "python",
            convert_script_path,
            save_directory,
            "--outfile",
            gguf_output_filename,
            "--outtype",
            output_type,
        ])
        print(f"Model converted successfully to GGUF: {gguf_output_filename}")


if __name__ == "__main__":
    main()
