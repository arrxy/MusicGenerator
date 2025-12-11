# %% [markdown]
# # Test Set Evaluation Script (H200 Optimized)
# This script calculates the final Test Loss and Perplexity for the XL_extended model.

# %%
import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

# Import project modules
try:
    from src.model import GPT, GPTConfig
    from src.dataset import MusicStreamingDataset
except ImportError:
    print("⚠️  Warning: 'src' modules not found. Ensure you are in the project root.")

# %% [markdown]
# ## 1. Configuration

# %%
# --- FILES ---
CHECKPOINT_PATH = "ckpt_XL_extended.pt"  # <--- Your specific checkpoint
TEST_DATA_PATH = "data/processed/test.txt"  # <--- Test data
VOCAB_PATH = "data/processed/vocab.json"

# --- HARDWARE (H200 Optimized) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128  # H200 has 141GB VRAM, we can handle large batches easily
BLOCK_SIZE = 256
NUM_WORKERS = 4  # Speed up data loading

# --- MODEL ARCHITECTURE ---
# Ensure this matches the config used to train 'XL_extended.pt'
# Assuming standard GPT-2 Base width (768) based on previous turns.
# If your model is actually 864 width, change n_embd to 864.
XL_CONFIG = dict(n_layer=12, n_head=12, n_embd=768, dropout=0.0)


# %% [markdown]
# ## 2. Evaluation Logic

# %%
def load_checkpoint(checkpoint_path, vocab_size):
    print(f"🚀 Initializing model on {DEVICE}...")

    # Init Skeleton
    gpt_conf = GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE, bias=True, **XL_CONFIG)
    model = GPT(gpt_conf)

    # Load Weights
    print(f"📂 Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Updated for PyTorch 2.6+: explicit weights_only=False to allow loading older pickles
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Fix dict prefixes if they exist (from torch.compile or DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k[10:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)

    # H200 Optimization: Compile for faster inference
    # Note: Compilation has a startup cost, but runs faster afterwards.
    # Useful if your test set is large.
    try:
        print("⚡ Compiling model for H200...")
        model = torch.compile(model)
    except Exception as e:
        print(f"Could not compile (skipping): {e}")

    model.eval()
    return model


def evaluate(model, data_loader):
    model.eval()
    losses = []
    start_time = time.time()

    # Using Automatic Mixed Precision (AMP) for H200 acceleration
    # bfloat16 is native and preferred on H100/H200
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"🔧 Precision: {dtype}")

    with torch.no_grad():
        for k, (X, Y) in enumerate(data_loader):
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=dtype):
                _, loss = model(X, Y)

            losses.append(loss.item())

            if k % 50 == 0 and k > 0:
                print(f"   Batch {k} | Loss: {loss.item():.4f}")

    total_time = time.time() - start_time

    if not losses:
        return 0.0, 0.0, 0.0

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity, total_time


# %% [markdown]
# ## 3. Main Execution

# %%
if __name__ == "__main__":
    print("=" * 50)
    print(f"H200 TEST SET EVALUATION")
    print("=" * 50)

    # 1. Setup Data
    try:
        test_ds = MusicStreamingDataset(TEST_DATA_PATH, VOCAB_PATH, BLOCK_SIZE)
        # Use num_workers to feed the GPU faster
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

        # Override vocab size to match your specific checkpoint history (1620)
        # Change this to test_ds.vocab_size if you retrained from scratch
        vocab_size = 1620
        print(f"📚 Vocab Size: {vocab_size}")
    except Exception as e:
        print(f"❌ Data Error: {e}")
        exit()

    # 2. Load Model
    try:
        model = load_checkpoint(CHECKPOINT_PATH, vocab_size)
        print(f"✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model Error: {e}")
        exit()

    # 3. Run Test
    print(f"\n🏃 Starting Test Loop on {TEST_DATA_PATH}...")
    loss, ppl, duration = evaluate(model, test_loader)

    # 4. Results
    print("\n" + "=" * 50)
    print(f"🏁 FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Model:      XL_extended.pt")
    print(f"Dataset:    {TEST_DATA_PATH}")
    print("-" * 50)
    print(f"📉 Test Loss:       {loss:.5f}")
    print(f"🔮 Test Perplexity: {ppl:.5f}")
    print(f"⏱️  Duration:        {duration:.2f}s")
    print("=" * 50)