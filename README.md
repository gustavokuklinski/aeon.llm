# AEON
AEON is portable, private, and capable of operating fully offline (with the exception of web search). It democratizes access to powerful, dynamic AI capabilities for a wider audience, regardless of their hardware.

Know more about Aeon: [Main Repository](https://github.com/gustavokuklinski/aeon.ai/)

## Aeon Trainer
This is a Aeon LLM trainer and dataset.

The model was trained on Google Colab GPU T4 Free tier

Model on huggingface.com/[gustavokuklinski/aeon-360m](https://huggingface.co/gustavokuklinski/aeon-360m)
Dataset on huggingface.com/datasets/[gustavokuklinski/aeon-360m](https://huggingface.co/datasets/gustavokuklinski/aeon)

## Scripts
- `aeon-train.py` The basic training script
- `aeon-gguf.py` Convertion from safetensors to GGUF (You can convert locally or on Colab for quantize compile LlamaCpp)