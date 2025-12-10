import os
import time
import gc
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.model import GPT, GPTConfig
from src.dataset import MusicStreamingDataset

# --- H200 HYPERPERFORMANCE CONFIG ---
BLOCK_SIZE = 256
BATCH_SIZE = 512        # Massive batch size for 80GB VRAM
LEARNING_RATE = 6e-4    # Increased LR for larger batch size
VOCAB_PATH = "data/processed/vocab.json"
TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"
INSIGHTS_FILE = "optimal_training_log.txt"
CSV_FILE = "optimal_results.csv"

# --- TRAINING DURATION CONFIG ---
TOKEN_MULTIPLIER = 200  
NUM_EPOCHS = 50

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("No GPU detected. This script requires an H100/A100/H200.")

# Model Family Configurations
model_configs = {
    # Name           Layer,      Head,      Embd          Approx Params
    "Tiny":     dict(n_layer=4,  n_head=4,  n_embd=128),  # ~1M
    "Small":    dict(n_layer=6,  n_head=6,  n_embd=288),  # ~5M
    "Medium":   dict(n_layer=8,  n_head=8,  n_embd=512),  # ~20M
    "Large":    dict(n_layer=10, n_head=10, n_embd=640),  # ~50M
    "XL":       dict(n_layer=12, n_head=12, n_embd=768),  # ~100M
}

def estimate_loss(model, val_loader, eval_iters=200):
    model.eval()
    losses = []
    accuracies = []
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()
    
    with torch.no_grad():
        with ctx:
            for k, (X, Y) in enumerate(val_loader):
                if k >= eval_iters: break
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                logits, loss = model(X, Y)
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == Y).float().mean()
                accuracies.append(acc.item())
    model.train()
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def log_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'params', 'tokens', 'val_loss', 'val_acc', 'time_sec'])
        writer.writerow(data)

def train_optimal(name, cfg):
    print(f"\n============================================")

    temp_vocab_size = 1620 
    config = GPTConfig(vocab_size=temp_vocab_size, block_size=BLOCK_SIZE, bias=True, **cfg)
    model = GPT(config).to(DEVICE)
    params = model.get_num_params()
    target_tokens = max(params * TOKEN_MULTIPLIER, 500_000_000)
    model = torch.compile(model)
    train_ds = MusicStreamingDataset(TRAIN_PATH, VOCAB_PATH, BLOCK_SIZE, max_tokens=target_tokens)
    val_ds = MusicStreamingDataset(VAL_PATH, VOCAB_PATH, BLOCK_SIZE)
    
    num_workers = 8 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    start_time = time.time()
    tokens_seen = 0
    step = 0
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    try:
        for epoch in range(NUM_EPOCHS):
            for X, Y in train_loader:
                X, Y = X.to(DEVICE, non_blocking=True), Y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with ctx:
                    logits, loss = model(X, Y)
                loss.backward()
                optimizer.step()
                tokens_seen += X.numel()
                step += 1
                if step % 50 == 0:
                    t_rate = tokens_seen / (time.time() - start_time)
                    print(f"   Step {step} | Loss: {loss.item():.4f} | Speed: {t_rate/1000:.1f}k tok/s | Total: {tokens_seen/1e6:.1f}M / {target_tokens/1e6:.1f}M")
                if tokens_seen >= target_tokens:
                    print(f"Reached Target Token Budget ({target_tokens:,} tokens)")
                    raise StopIteration
                    
    except StopIteration:
        pass
    
    total_time = time.time() - start_time
    val_loss, val_acc = estimate_loss(model, val_loader)
    log_to_csv([name, params, tokens_seen, val_loss, val_acc, total_time])
    tokens_per_sec = tokens_seen / total_time if total_time > 0 else 0
    insight_text = (
        f"--------------------------------------------------\n"
        f"Model: {name}\n"
        f"Parameters: {params:,}\n"
        f"Epochs Completed: {epoch+1}\n"
        f"Tokens Trained: {tokens_seen:,}\n"
        f"Validation Loss: {val_loss:.4f}\n"
        f"Validation Accuracy: {val_acc:.4f}\n"
        f"Training Time: {total_time/60:.1f} min\n"
        f"Speed: {tokens_per_sec:,.0f} tok/s\n"
        f"--------------------------------------------------\n"
    )
    with open(INSIGHTS_FILE, "a") as f:
        f.write(insight_text)
    ckpt_path = f"ckpt_{name}_extended.pt"
    print(f" Saving checkpoint to {ckpt_path}...")
    torch.save(model.state_dict(), ckpt_path)
    
    del model, optimizer, X, Y
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    models_to_train = ["Tiny","Small","Medium","Large","XL"] 
    
    for name in models_to_train:
        if name in model_configs:
            train_optimal(name, model_configs[name])