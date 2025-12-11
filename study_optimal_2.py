import os
import time
import gc
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import project modules
try:
    from src.model import GPT, GPTConfig
    from src.dataset import MusicStreamingDataset
except ImportError:
    print("⚠️  Warning: 'src' modules not found.")

# --- H200 HYPERPERFORMANCE CONFIG ---
BLOCK_SIZE = 256
BATCH_SIZE = 512  # Massive batch size for 80GB/141GB VRAM
BASE_LEARNING_RATE = 6e-4
VOCAB_PATH = "data/processed/vocab.json"
TRAIN_PATH = "data/processed/train.txt"
VAL_PATH = "data/processed/val.txt"
TEST_PATH = "data/processed/test.txt"
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
    print("No GPU detected.")

# --- FIX 1: Add Dropout to prevent overfitting ---
model_configs = {
    "Tiny": dict(n_layer=4, n_head=4, n_embd=128, dropout=0.1),
    "Small": dict(n_layer=6, n_head=6, n_embd=288, dropout=0.1),
    "Medium": dict(n_layer=8, n_head=8, n_embd=512, dropout=0.1),
    "Large": dict(n_layer=10, n_head=10, n_embd=640, dropout=0.1),
    "XL": dict(n_layer=12, n_head=12, n_embd=768, dropout=0.1),
}


def estimate_loss(model, val_loader, eval_iters=200):
    """
    Evaluates loss.
    If eval_iters is None, it runs the FULL dataset (used for final report).
    If eval_iters is set (e.g. 200), it runs a partial check (used for speed during training).
    """
    model.eval()
    losses = []
    accuracies = []
    # Use bfloat16 for H200 evaluation
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()

    with torch.no_grad():
        with ctx:
            for k, (X, Y) in enumerate(val_loader):
                # Break only if eval_iters is set (not None)
                if eval_iters is not None and k >= eval_iters:
                    break

                X, Y = X.to(DEVICE), Y.to(DEVICE)
                logits, loss = model(X, Y)
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                acc = (preds == Y).float().mean()
                accuracies.append(acc.item())

    model.train()
    if not losses: return 0.0, 0.0
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def log_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['model', 'params', 'tokens', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'time_sec'])
        writer.writerow(data)


def train_optimal(name, cfg):
    print(f"\n============================================")
    print(f"🚀 TRAINING MODEL: {name}")

    # --- Scale LR for larger models ---
    current_lr = BASE_LEARNING_RATE
    if name in ["Large", "XL"]:
        current_lr = 3e-4
        print(f"ℹ️  Adjusted LR to {current_lr} for large model stability")

    temp_vocab_size = 1620

    # Init Model Config
    config = GPTConfig(vocab_size=temp_vocab_size, block_size=BLOCK_SIZE, bias=True, **cfg)
    model = GPT(config).to(DEVICE)
    params = model.get_num_params()

    # --- CHECKPOINT LOADING LOGIC ---
    ckpt_filename = f"ckpt_{name}_extended.pt"
    if os.path.exists(ckpt_filename):
        print(f"🔄 Found checkpoint: {ckpt_filename}. Resuming training...")
        try:
            # Load with weights_only=False to support legacy/robust formats
            checkpoint = torch.load(ckpt_filename, map_location=DEVICE, weights_only=False)

            # Handle Robust Checkpoint (Dict) vs Legacy (State Dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Clean prefixes (from torch.compile or DataParallel)
            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    clean_state[k[10:]] = v
                elif k.startswith("module."):
                    clean_state[k[7:]] = v
                else:
                    clean_state[k] = v

            model.load_state_dict(clean_state)
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("⚠️  Starting from scratch instead.")
    else:
        print(f"✨ No checkpoint found ({ckpt_filename}). Initializing from scratch.")

    # Calculate token budget
    target_tokens = max(params * TOKEN_MULTIPLIER, 500_000_000)
    target_tokens = min(target_tokens, 5_000_000_000)

    # Compile
    # Note: We keep a reference to the compiled model for training
    compiled_model = torch.compile(model)

    # Datasets
    train_ds = MusicStreamingDataset(TRAIN_PATH, VOCAB_PATH, BLOCK_SIZE, max_tokens=target_tokens)
    val_ds = MusicStreamingDataset(VAL_PATH, VOCAB_PATH, BLOCK_SIZE)
    test_ds = MusicStreamingDataset(TEST_PATH, VOCAB_PATH, BLOCK_SIZE)

    num_workers = 8
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=current_lr)

    compiled_model.train()
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
                    logits, loss = compiled_model(X, Y)
                loss.backward()
                optimizer.step()

                tokens_seen += X.numel()
                step += 1

                if step % 50 == 0:
                    t_rate = tokens_seen / (time.time() - start_time)
                    print(
                        f"   Step {step} | Loss: {loss.item():.4f} | Speed: {t_rate / 1000:.1f}k tok/s | Total: {tokens_seen / 1e6:.1f}M / {target_tokens / 1e6:.1f}M")

                if tokens_seen >= target_tokens:
                    print(f"Reached Target Token Budget ({target_tokens:,} tokens)")
                    raise StopIteration

    except StopIteration:
        pass

    total_time = time.time() - start_time

    # --- FINAL EVALUATION (FULL DATASET) ---
    print("Running FINAL FULL evaluation (this may take a moment)...")
    # Setting eval_iters=None ensures we test the WHOLE dataset, matching your reload script
    val_loss, val_acc = estimate_loss(compiled_model, val_loader, eval_iters=None)
    testng_loss, testng_acc = estimate_loss(compiled_model, test_loader, eval_iters=None)

    log_to_csv([name, params, tokens_seen, val_loss, val_acc, testng_loss, testng_acc, total_time])

    # Insights
    tokens_per_sec = tokens_seen / total_time if total_time > 0 else 0
    insight_text = (
        f"--------------------------------------------------\n"
        f"Model: {name}\n"
        f"Parameters: {params:,}\n"
        f"Tokens Trained: {tokens_seen:,}\n"
        f"Validation Loss: {val_loss:.4f}\n"
        f"Test Loss: {testng_loss:.4f}\n"
        f"Speed: {tokens_per_sec:,.0f} tok/s\n"
        f"--------------------------------------------------\n"
    )
    with open(INSIGHTS_FILE, "a") as f:
        f.write(insight_text)

    # --- SAVE ROBUST CHECKPOINT ---
    # Saves config so you don't get "size mismatch" errors later
    ckpt_path = f"ckpt_{name}_robust.pt"
    print(f"💾 Saving ROBUST checkpoint to {ckpt_path}...")

    # Unwrap: Get the original model from the compiled wrapper
    # This removes the '_orig_mod.' prefix that breaks loading scripts
    raw_model = compiled_model._orig_mod if hasattr(compiled_model, '_orig_mod') else compiled_model

    checkpoint_data = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
        'vocab_size': temp_vocab_size,
        'val_loss': val_loss
    }
    torch.save(checkpoint_data, ckpt_path)

    # Cleanup
    del compiled_model, raw_model, optimizer, X, Y
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    models_to_train = ["Tiny", "Small", "Medium", "Large", "XL"]

    for name in models_to_train:
        if name in model_configs:
            train_optimal(name, model_configs[name])