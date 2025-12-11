import os
import time
import math
import torch
from torch.utils.data import DataLoader

from src.model import GPT, GPTConfig
from src.dataset import MusicStreamingDataset

CHECKPOINT_PATH = "ckpt_Tiny_robust.pt"
VAL_DATA_PATH = "data/processed/val.txt"
VOCAB_PATH = "data/processed/vocab.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 256
BATCH_SIZE = 128
NUM_WORKERS = 4
VOCAB_SIZE = 1620

MODEL_CONFIG = dict(
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)

def load_model(checkpoint_path):
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        **MODEL_CONFIG
    )
    model = GPT(config).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    clean_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            clean_state[k[10:]] = v
        elif k.startswith("module."):
            clean_state[k[7:]] = v
        else:
            clean_state[k] = v

    model.load_state_dict(clean_state)

    try:
        model = torch.compile(model)
    except Exception:
        pass

    model.eval()
    return model

def evaluate(model, loader):
    losses = []
    start = time.time()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for X, Y in loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                _, loss = model(X, Y)
                losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    duration = time.time() - start
    return avg_loss, perplexity, duration

if __name__ == "__main__":
    val_ds = MusicStreamingDataset(VAL_DATA_PATH, VOCAB_PATH, BLOCK_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

    model = load_model(CHECKPOINT_PATH)

    loss, ppl, t = evaluate(model, val_loader)

    print("Test Loss:", f"{loss:.5f}")
    print("Perplexity:", f"{ppl:.5f}")
    print("Time:", f"{t:.2f}s")