import os
import time
import math
import torch
from torch.utils.data import DataLoader

# Handle imports based on where the script is run
try:
    from src.model import GPT, GPTConfig
    from src.dataset import MusicStreamingDataset
except ImportError:
    print("Warning: 'src' modules not found. Ensure you are in the project root.")

# CONFIG
CHECKPOINT_PATH = "ckpt_Tiny_robust.pt"
VAL_DATA_PATH = "data/processed/val.txt"
VOCAB_PATH = "data/processed/vocab.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 256
BATCH_SIZE = 128
NUM_WORKERS = 4
VOCAB_SIZE = 1620

# Ensure this matches the architecture used during training
model_configs = {
    "Tiny": dict(n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True),
    "Small": dict(n_layer=6, n_head=6, n_embd=288, dropout=0.0, bias=True),
    "Medium": dict(n_layer=8, n_head=8, n_embd=512, dropout=0.0, bias=True),
    "Large": dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0, bias=True),
    "XL": dict(n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True),
}
MODEL_CONFIG = model_configs["Tiny"]


def load_model(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")

    # 1. Initialize model structure first
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        **MODEL_CONFIG
    )
    model = GPT(config).to(DEVICE)

    # 2. Load the checkpoint file
    # weights_only=False is needed because we are loading a complex dict, not just weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # 3. FIX: Extract the state dict if it's nested inside a dictionary
    # Your training script saves it under 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("   -> Found nested 'model_state_dict', extracting weights...")
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 4. Clean up prefixes (from torch.compile or DataParallel)
    clean_state = {}
    for k, v in state_dict.items():
        # Remove '_orig_mod.' prefix from torch.compile()
        if k.startswith("_orig_mod."):
            clean_state[k[10:]] = v
        # Remove 'module.' prefix from DataParallel
        elif k.startswith("module."):
            clean_state[k[7:]] = v
        else:
            clean_state[k] = v

    # 5. Load weights into model
    msg = model.load_state_dict(clean_state, strict=True)
    print(f"   -> Weights loaded. {msg}")

    # Optional: Compile for speed if on Linux/CUDA
    if os.name == 'posix':
        try:
            model = torch.compile(model)
            print("   -> Model compiled.")
        except Exception as e:
            print(f"   -> Compilation skipped: {e}")

    model.eval()
    return model


def evaluate(model, loader):
    print(f"Starting evaluation on {DEVICE}...")
    losses = []
    start = time.time()

    with torch.no_grad():
        # Use Autocast for efficiency (bfloat16 on Ampere+, float16 otherwise)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast(device_type=DEVICE, dtype=dtype):
            for i, (X, Y) in enumerate(loader):
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                _, loss = model(X, Y)
                losses.append(loss.item())

                # Removed len(loader) which caused the TypeError
                if i % 10 == 0:
                    print(f"   Batch {i}: Loss {loss.item():.4f}", end="\r")

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    duration = time.time() - start

    print(f"\n   -> Eval Complete.")
    return avg_loss, perplexity, duration


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file '{CHECKPOINT_PATH}' not found.")
    else:
        # Load Dataset
        val_ds = MusicStreamingDataset(VAL_DATA_PATH, VOCAB_PATH, BLOCK_SIZE)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

        # Load Model
        model = load_model(CHECKPOINT_PATH)

        # Run Eval
        loss, ppl, t = evaluate(model, val_loader)

        print("\n===========================")
        print(f"Test Loss:   {loss:.5f}")
        print(f"Perplexity:  {ppl:.5f}")
        print(f"Time Taken:  {t:.2f}s")
        print("===========================")