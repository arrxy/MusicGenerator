import os
import time
import csv
import torch
import math
from torch.utils.data import DataLoader
from src.rnn_model import RNNModel, RNNConfig
from src.dataset import MusicStreamingDataset

BLOCK_SIZE = 256
BATCH_SIZE = 512        
LEARNING_RATE = 1e-3    
VOCAB_PATH = "data/processed/vocab.json"
TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"
CSV_FILE = "rnn_scaling_results.csv"

# --- SCALING STUDY TARGETS ---
TARGET_PARAMS = {
    "Tiny": 1_000_000,
    "Small": 6_000_000,
    "Medium": 26_000_000,
    "Large": 50_000_000
}

TOKEN_BUDGET = 100_000_000  # 100 Million Tokens

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"{torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'

def find_optimal_rnn_config(target_params, vocab_size=1620):
    best_diff = float('inf')
    best_config = None
    best_params = 0
    
    for h_dim in range(64, 4096, 16):
        n_layers = 2 if target_params < 5_000_000 else 3
        cfg = RNNConfig(vocab_size, h_dim, h_dim, n_layers)
        model = RNNModel(cfg)
        params = model.get_num_params()
        
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_config = cfg
            best_params = params
        if params > target_params: 
            break
            
    return best_config, best_params

def estimate_loss(model, val_loader, eval_iters=100):
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

def log_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'params', 'tokens', 'val_loss', 'time_sec', 'speed_tok_sec'])
        writer.writerow(data)

def train_rnn(name, target_p):
    config, actual_params = find_optimal_rnn_config(target_p, vocab_size=1620)
    model = RNNModel(config).to(DEVICE)
    train_ds = MusicStreamingDataset(TRAIN_PATH, VOCAB_PATH, BLOCK_SIZE, max_tokens=TOKEN_BUDGET)
    val_ds = MusicStreamingDataset(VAL_PATH, VOCAB_PATH, BLOCK_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    start_time = time.time()
    tokens_seen = 0
    step = 0
    
    total_steps = int(TOKEN_BUDGET // (BATCH_SIZE * BLOCK_SIZE))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    try:
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            model.zero_grad(set_to_none=True)
            _, loss = model(X, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tokens_seen += X.numel()
            step += 1
            if step % 50 == 0:
                dt = time.time() - start_time
                speed = tokens_seen / dt
                print(f"   Step {step} | Loss: {loss.item():.4f} | Speed: {speed/1000:.1f}k tok/s | Progress: {tokens_seen/1e6:.1f}M / 100M")

            if tokens_seen >= TOKEN_BUDGET:
                break
                
    except KeyboardInterrupt:
        print("Stopped by user.")

    total_time = time.time() - start_time
    avg_speed = tokens_seen / total_time
    val_loss = estimate_loss(model, val_loader)
    ckpt_filename = f"ckpt_rnn_{name}.pt"
    print(f"Saving checkpoint to {ckpt_filename}...")
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'params': actual_params
    }, ckpt_filename)

    log_to_csv([name, actual_params, tokens_seen, val_loss, total_time, avg_speed])
    
    del model, optimizer, scheduler, X, Y
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for name, params in TARGET_PARAMS.items():
        train_rnn(name, params)