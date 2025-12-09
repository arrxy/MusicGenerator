# %% [markdown]
# # Transformer Scaling Study
# This notebook runs the scaling laws experiment.
# We train 5 models (Tiny -> XL) on 100M tokens each and measure validation loss.

# %%
import os
import time
import math
import gc
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Import our project modules
# Ensure you are running this from the project root directory
from src.model import GPT, GPTConfig
from src.dataset import MusicStreamingDataset

# %% [markdown]
# ## 1. Configuration & Hyperparameters
# Defined to ensure fair comparison across all model sizes.

# %%
# --- HARDWARE ---
# Check for Apple Silicon (MPS)
if torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("Using MacOS MPS")
elif torch.cuda.is_available():
    DEVICE = 'cuda'
    print("Using CUDA GPU")
else:
    DEVICE = 'cpu'
    print("Using CPU")

# --- CONSTANTS ---
TOKENS_PER_EPOCH = 100_000_000  # 100M tokens per model
BLOCK_SIZE = 256  # Context window length
BATCH_SIZE = 64  # Adjust down if you hit Out of Memory (OOM)
LEARNING_RATE = 3e-4  # Constant LR for simplicity in scaling study
VOCAB_PATH = "data/processed/vocab.json"
TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"
INSIGHTS_FILE = "training_insights.txt"

# --- MODEL FAMILY ---
# Tiny (1M), Small (5M), Medium (20M), Large (50M), XL (100M)
# Calculated approximately based on standard GPT dimensions
model_configs = {
    "Tiny": dict(n_layer=4, n_head=4, n_embd=128, dropout=0.0),
    "Small": dict(n_layer=6, n_head=6, n_embd=288, dropout=0.0),
    "Medium": dict(n_layer=8, n_head=8, n_embd=512, dropout=0.0),
    "Large": dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0),
    "XL": dict(n_layer=12, n_head=12, n_embd=864, dropout=0.0),
}

print(f"🎯 Target: Train each model for {TOKENS_PER_EPOCH / 1e6:.0f}M tokens.")


# %% [markdown]
# ## 2. Helper Functions
# Loss estimation and training loop logic.

# %%
def estimate_loss(model, val_loader, eval_iters=100):
    """
    Estimates validation loss by averaging over 'eval_iters' batches.
    Crucial for the Y-axis of our scaling plot.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for k, (X, Y) in enumerate(val_loader):
            if k >= eval_iters: break
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            _, loss = model(X, Y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_model(name, cfg):
    """
    Initializes and trains a single GPT model configuration.
    """
    # 1. Setup Data
    # Re-initialize dataset for each model to ensure they see the exact same data stream
    train_ds = MusicStreamingDataset(TRAIN_PATH, VOCAB_PATH, BLOCK_SIZE, max_tokens=TOKENS_PER_EPOCH)
    val_ds = MusicStreamingDataset(VAL_PATH, VOCAB_PATH, BLOCK_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

    vocab_size = train_ds.vocab_size

    # 2. Setup Model
    gpt_conf = GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE, bias=True, **cfg)
    model = GPT(gpt_conf).to(DEVICE)

    # Count Parameters
    params = model.get_num_params()
    print(f"\n🏗️  Model: {name}")
    print(f"    Params: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    model.train()
    start_time = time.time()
    tokens_processed = 0
    step = 0
    losses = []

    print(f"    Training start...")

    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        optimizer.zero_grad()
        _, loss = model(X, Y)
        loss.backward()
        optimizer.step()

        tokens_in_batch = X.numel()
        tokens_processed += tokens_in_batch
        step += 1

        if step % 100 == 0:
            print(f"    Step {step} | Loss: {loss.item():.4f} | Tokens: {tokens_processed / 1e6:.1f}M")
            losses.append(loss.item())

        # Explicit Hard Stop ensures every model sees exactly the same amount of data
        if tokens_processed >= TOKENS_PER_EPOCH:
            print(f"    ✅ Reached token limit ({TOKENS_PER_EPOCH}). Stopping.")
            break

    total_time = time.time() - start_time

    # 4. Final Evaluation
    val_loss = estimate_loss(model, val_loader)
    print(f"    🏁 Finished. Val Loss: {val_loss:.4f} | Time: {total_time / 60:.1f} min")

    # --- INSIGHT GENERATION ---
    tokens_per_sec = tokens_processed / total_time if total_time > 0 else 0
    insight_text = (
        f"--------------------------------------------------\n"
        f"Model: {name}\n"
        f"Parameters: {params:,}\n"
        f"Validation Loss: {val_loss:.4f}\n"
        f"Training Time: {total_time:.2f}s ({total_time / 60:.1f}m)\n"
        f"Training Speed: {tokens_per_sec:,.0f} tokens/sec\n"
        f"Tokens Processed: {tokens_processed:,}\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"--------------------------------------------------\n"
    )

    with open(INSIGHTS_FILE, "a") as f:
        f.write(insight_text)
    print(f"📝 Insights saved to {INSIGHTS_FILE}")

    # --- SAVE CHECKPOINT ---
    # We save the model weights so we can generate music later
    ckpt_path = f"ckpt_{name}.pt"
    print(f"💾 Saving checkpoint to {ckpt_path}...")
    torch.save(model.state_dict(), ckpt_path)

    # 5. Cleanup (CRITICAL FOR NOTEBOOKS)
    # Delete model from GPU memory so next model fits
    del model
    del optimizer
    del X
    del Y
    if DEVICE == 'mps':
        torch.mps.empty_cache()
    elif DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": name,
        "params": params,
        "val_loss": val_loss,
        "time_sec": total_time,
        "losses": losses
    }


# %% [markdown]
# ## 3. Execution Phase
# Run the training for all models in the family.
# This cell will take time (approx 1-3 hours per model on M4 depending on size).

# %%
if __name__ == "__main__":
    # Initialize insights file
    with open(INSIGHTS_FILE, "w") as f:
        f.write(f"=== TRANSFORMER SCALING STUDY INSIGHTS ===\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"Device: {DEVICE}\n\n")

    results = []

    for name, cfg in model_configs.items():
        try:
            res = train_model(name, cfg)
            results.append(res)
        except Exception as e:
            print(f"❌ Failed to train {name}: {e}")
            # Try to cleanup anyway
            if DEVICE == 'mps': torch.mps.empty_cache()
            gc.collect()

    print("\n🎉 All training runs complete!")

    if results:
        # %% [markdown]
        # ## 4. Analysis & Scaling Plot
        # Fit the Power Law curve and visualize the results.
        # $L(N) = a N^{-\alpha} + c$

        # %%
        # Convert to DataFrame
        df = pd.DataFrame(results)
        print(df[["model", "params", "val_loss"]])


        # Define Power Law function
        def power_law(N, a, alpha, c):
            return a * np.power(N, -alpha) + c


        x_data = df['params'].values
        y_data = df['val_loss'].values

        # Try fitting
        try:
            # Initial guesses: a=10, alpha=0.1, c=2.0
            popt, _ = curve_fit(power_law, x_data, y_data, p0=[10, 0.1, 1.0], maxfev=10000)
            a_fit, alpha_fit, c_fit = popt
            print(f"\n📈 Fitted Power Law Scaling Exponent (alpha): {alpha_fit:.4f}")
        except Exception as e:
            print(f"Could not fit curve perfectly: {e}")
            popt = None

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, s=150, c='red', zorder=5, label='Experimental Models')

        # Draw fit line
        if popt is not None:
            x_range = np.linspace(min(x_data) * 0.9, max(x_data) * 1.1, 100)
            plt.plot(x_range, power_law(x_range, *popt), 'b--', linewidth=2,
                     label=f'Power Law Fit ($\\alpha={alpha_fit:.2f}$)')

        plt.xscale('log')
        # plt.yscale('log') # Optional: Log-Log plot often looks straighter
        plt.xlabel('Parameters (N)', fontsize=12)
        plt.ylabel('Validation Loss (L)', fontsize=12)
        plt.title('Scaling Laws: Test Loss vs Model Size', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        plt.savefig('scaling_plot.png')
        print("📈 Plot saved to 'scaling_plot.png'")
        plt.show()

        # %% [markdown]
        # ## 5. Training Curves
        # Visualize convergence speed.

        # %%
        plt.figure(figsize=(12, 6))
        for r in results:
            plt.plot(r['losses'], label=f"{r['model']} ({r['params'] / 1e6:.1f}M)")

        plt.xlabel('Step (x100)')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()