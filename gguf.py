import os
import subprocess
import sys

# Safetensor modelfiles
save_directory = "./aeon/finetuned_llm/" 

# The desired output filename for the GGUF model.
gguf_output_filename = "gguf/aeon.Q8_0.gguf"

# The output type for the GGUF model (f16, q8_0).
output_type = "q8_0"


def run_command(command):
    try:
        print(f"\nExecuting: {' '.join(command)}")
        if command[0] == 'pip' or command[0] == 'python':
            command[0] = sys.executable
            if command[0] == sys.executable and command[1] != '-m':
                 command.insert(1, '-m')
        
        if command[1] == '-m' and command[0] == sys.executable:
            if command[2] == 'pip':
                command = [sys.executable, '-m', 'pip'] + command[2:]
            else: # python script.py
                command = [sys.executable] + command[2:]


        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)


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
        sys.exit(1)


def main():
    if not os.path.exists(save_directory):
        print(f"Error: The model directory '{save_directory}' does not exist.")
        print("Please download your Hugging Face model files and place them in that directory,")
        print("or update the 'save_directory' variable in this script.")
        os.makedirs(save_directory, exist_ok=True)
        print(f"Created a placeholder directory at '{save_directory}'.")
        sys.exit(1)


    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp repository...")
        run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
        print("llama.cpp cloned successfully.")
        print("Installing llama.cpp Python dependencies...")
        requirements_path = os.path.join("llama.cpp", "requirements.txt")
        run_command(["pip", "install", "-r", requirements_path])
        print("Dependencies installed successfully.")
    else:
        print("llama.cpp directory already exists, skipping clone.")


    print(f"Converting model to GGUF format: {gguf_output_filename}...")
    convert_script_path = os.path.join("llama.cpp", "convert_hf_to_gguf.py")


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
