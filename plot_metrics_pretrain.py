import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np # <-- NEW: Added for setting custom tick steps

# --- Configuration ---
LOG_DIR = "./aeon/" # Directory containing the log file
LOG_FILE_NAME = "trainer_state.json"
FULL_LOG_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
PLOT_FILE_NAME = "training_metrics_step_by_step.png"

def plot_training_metrics():
    # STEP 1: Load and Validate Log File
    print(f"\n--- STEP 1: Loading Data ---")
    print(f"\033[1;93m[LOAD]\033[0m Attempting to read training logs from: {FULL_LOG_PATH}")
    
    try:
        with open(FULL_LOG_PATH, 'r') as f:
            log_data = json.load(f)
        
        print("\033[1;32m[SUCCESS]\033[0m Log file loaded successfully.")
        
        df = pd.DataFrame(log_data['log_history'])
        print(f"\033[1;93m[DATA]\033[0m Total log entries loaded: {len(df)}")
        
    except FileNotFoundError:
        print(f"\033[1;91m[ERROR]\033[0m Log file not found at {FULL_LOG_PATH}.")
        print("Ensure you have successfully run the training script and the path is correct.")
        return
    except KeyError:
        print(f"\033[1;91m[ERROR]\033[0m 'log_history' not found in trainer_state.json. File may be corrupted or incomplete.")
        return
    except Exception as e:
        print(f"\033[1;91m[ERROR]\033[0m An unexpected error occurred during file loading: {e}")
        return


    # STEP 2: Separate Training and Evaluation Metrics
    print(f"\n--- STEP 2: Preparing DataFrames ---")

    # Filter for training loss records (they have a 'loss' value)
    train_df = df.dropna(subset=['loss']).rename(columns={'loss': 'Training Loss'})
    print(f"\033[1;93m[TRAIN]\033[0m Extracted {len(train_df)} training loss records.")

    # Filter for evaluation loss records (they have an 'eval_loss' value)
    eval_df = df.dropna(subset=['eval_loss']).rename(columns={'eval_loss': 'Evaluation Loss'})
    print(f"\033[1;93m[EVAL]\033[0m Extracted {len(eval_df)} evaluation loss records.")
    
    if train_df.empty and eval_df.empty:
        print("\033[1;93m[WARNING]\033[0m No loss metrics found in the log history. Skipping plot generation.")
        return

    # STEP 3: Configure and Generate Plot
    print(f"\n--- STEP 3: Generating Plot ---")

    plt.figure(figsize=(10, 6))

    # Plot Training Loss
    plt.plot(
        train_df['step'], 
        train_df['Training Loss'], 
        label='Training Loss (Step)', 
        color='blue', 
        alpha=0.7
    )
    print("\033[1;93m[PLOT]\033[0m Plotted Training Loss.")

    # Plot Evaluation Loss
    plt.plot(
        eval_df['step'], 
        eval_df['Evaluation Loss'], 
        label='Evaluation Loss (Epoch)', 
        color='red', 
        marker='o', 
        linestyle='--'
    )
    print("\033[1;93m[PLOT]\033[0m Plotted Evaluation Loss.")

    # STEP 4: Apply Aesthetics and Save
    print(f"\n--- STEP 4: Applying Aesthetics and Saving ---")
    
    # Calculate appropriate tick marks based on the training data range
    if not train_df.empty:
        max_step = train_df['step'].max()
        # Set ticks from 0 up to the next 100-step increment
        tick_interval = 100
        x_ticks = np.arange(0, max_step + tick_interval, tick_interval)
        plt.xticks(x_ticks) # <-- NEW: Set custom X-axis ticks
        print(f"\033[1;93m[PLOT]\033[0m Adjusted X-axis ticks to increments of {tick_interval} steps.")


    plt.title('Language Model Pre-training Metrics (Loss)', fontsize=16)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    print("\033[1;93m[PLOT]\033[0m Applied title, labels, legend, and grid.")

    plt.savefig(PLOT_FILE_NAME)
    print(f"\033[1;32m[SUCCESS]\033[0m Plot saved to {PLOT_FILE_NAME}")

if __name__ == "__main__":
    plot_training_metrics()
