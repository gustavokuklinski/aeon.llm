import os
import shutil

gguf_output_filename = "aeon.gguf"

if not os.path.exists("llama.cpp"):
    print("\nCloning llama.cpp repository...")

    !git clone https://github.com/ggerganov/llama.cpp.git

    print("llama.cpp cloned.")
    print("\nInstalling llama.cpp Python dependencies...")

    !pip install -r {os.path.join("llama.cpp", "requirements.txt")}

else:
    print("\nllama.cpp already exists, skipping clone.")


print(f"\nConverting model to GGUF format: {gguf_output_filename}...")

!python ./llama.cpp/convert_hf_to_gguf.py {save_directory}  --outfile {gguf_output_filename}  --outtype f16

print(f"Model converted to GGUF: {gguf_output_filename}")