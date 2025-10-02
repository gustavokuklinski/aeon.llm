import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import math
import numpy as np

# --- Configuration ---
LOG_DIR = "./aeon/finetuned_llm/output" # Directory containing the log file
LOG_FILE_NAME = "trainer_state.json"
FULL_LOG_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
PLOT_FILE_NAME = "training_metrics_finetune.png"

def process_training_logs():
    print(f"\033[1;93m[LOAD]\033[0m Attempting to read training logs from: {FULL_LOG_PATH}")
    
    try:
        with open(FULL_LOG_PATH, 'r') as f:
            log_data = json.load(f)
        
        print("\033[1;32m[SUCCESS]\033[0m Log file loaded successfully.")
        
        df = pd.DataFrame(log_data['log_history'])
        print(f"\033[1;93m[DATA]\033[0m Total log entries loaded: {len(df)}")
        
    except FileNotFoundError:
        print(f"\033[1;91m[ERROR]\033[0m Log file not found at {FULL_LOG_PATH}.")
        print("Ensure the path is correct and training has run.")
        return
    except KeyError:
        print(f"\033[1;91m[ERROR]\033[0m 'log_history' not found in trainer_state.json.")
        return
    except Exception as e:
        print(f"\033[1;91m[ERROR]\033[0m An unexpected error occurred: {e}")
        return

    train_df = df.dropna(subset=['loss']).rename(columns={'loss': 'Training Loss'})
    print(f"\033[1;93m[TRAIN]\033[0m Extracted {len(train_df)} training loss records.")

    eval_df = df.dropna(subset=['eval_loss']).rename(columns={'eval_loss': 'Evaluation Loss'})
    print(f"\033[1;93m[EVAL]\033[0m Extracted {len(eval_df)} evaluation loss records.")
    
    if train_df.empty and eval_df.empty:
        print("\033[1;93m[WARNING]\033[0m No loss metrics found in the log history. Skipping plot generation.")
        return

    plt.figure(figsize=(10, 6))

    if not train_df.empty:
        plt.plot(
            train_df['step'], 
            train_df['Training Loss'], 
            label='Training Loss (Step)', 
            color='blue', 
            alpha=0.7
        )
        print("\033[1;93m[PLOT]\033[0m Plotted Training Loss.")

    if not eval_df.empty:
        plt.plot(
            eval_df['step'], 
            eval_df['Evaluation Loss'], 
            label='Validation Loss (Epoch)', 
            color='red', 
            marker='o', 
            linestyle='--'
        )
        print("\033[1;93m[PLOT]\033[0m Plotted Validation Loss.")
   
    if not train_df.empty:
        max_step = train_df['step'].max()
        # Set ticks from 0 up to the next 100-step increment
        tick_interval = 500
        x_ticks = np.arange(0, max_step + tick_interval, tick_interval)
        plt.xticks(x_ticks) 
        print(f"\033[1;93m[PLOT]\033[0m Adjusted X-axis ticks to increments of {tick_interval} steps.")


    plt.title('Language Model Fine-Tuning Metrics (Loss)', fontsize=16)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    print("\033[1;93m[PLOT]\033[0m Applied title, labels, legend, and grid.")

    plt.savefig(PLOT_FILE_NAME)
    print(f"\033[1;32m[SUCCESS]\033[0m Plot saved to {PLOT_FILE_NAME}")

if __name__ == "__main__":
    process_training_logs()
