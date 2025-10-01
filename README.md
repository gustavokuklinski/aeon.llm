# AEON
AEON is portable, private, and capable of operating fully offline (with the exception of web search). It democratizes access to powerful, dynamic AI capabilities for a wider audience, regardless of their hardware.

Know more about Aeon: [Main Repository](https://github.com/gustavokuklinski/aeon.ai/)

## Aeon Trainer
This is a Aeon LLM trainer scripts.

```pretrain.py``` The Raw corpus pre training (Trainer)
```finetune.py``` Instruction oriented finetune (Supervised Trainer)
```chat.py``` Testing fine tuned model


### ```pretrain.py``` Set model size

```python
# Select from keys in MODEL_PRESETS (e.g., 10, 20, 80, 250, 500)
TARGET_PARAMS = 100

# --- MODEL PRESETS ---
MODEL_PRESETS = {
    10:  {'n_layer': 4,  'n_head': 4,  'n_embd': 168},  # Actual ~10.3M
    20:  {'n_layer': 6,  'n_head': 6,  'n_embd': 252},  # Actual ~17.3M
    80:  {'n_layer': 10, 'n_head': 8,  'n_embd': 512},  # Actual ~84.1M
    100: {'n_layer': 12, 'n_head': 8,  'n_embd': 640},  # Actual ~91.5M
    150: {'n_layer': 16, 'n_head': 10, 'n_embd': 768},  # Actual ~163.7M
    200: {'n_layer': 18, 'n_head': 12, 'n_embd': 768},  # Actual ~200.7M
    250: {'n_layer': 20, 'n_head': 16, 'n_embd': 768},  # Actual ~240.2M
    300: {'n_layer': 24, 'n_head': 16, 'n_embd': 768},  # Actual ~288.7M
    350: {'n_layer': 26, 'n_head': 16, 'n_embd': 768},  # Actual ~313.2M
    400: {'n_layer': 28, 'n_head': 16, 'n_embd': 768},  # Actual ~337.7M
    450: {'n_layer': 30, 'n_head': 16, 'n_embd': 768},  # Actual ~362.2M
    500: {'n_layer': 32, 'n_head': 16, 'n_embd': 768},  # Actual ~386.7M
}
```

## Legacy

```notebook/aeon-finetune-smollm-360M.ipynb``` The notebook used for first testing.
