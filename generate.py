import torch
import torch.nn.functional as F
import os
import time
from src.model import GPT, GPTConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION ---
MODEL_SIZE = "Large"         # Must match the checkpoint filename
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}_extended.pt" # Points to your optimal file
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 64         # Length of the song
TEMPERATURE = 0.8            # 1.0 = Random/Creative, 0.8 = Focused/Safe
TOP_K = 600                  # Limit to top N most likely next notes
REPETITION_PENALTY = 1.1     # Reduce probability of recently used tokens (1.0 = No penalty)

# Model Architecture Registry (Must match training exactly)
# Note: Training used bias=True, so we must use it here too.
model_configs = {
    "Tiny":   dict(n_layer=4,  n_head=4,  n_embd=128, dropout=0.0, bias=True),
    "Small":  dict(n_layer=6,  n_head=6,  n_embd=288, dropout=0.0, bias=True),
    "Medium": dict(n_layer=8,  n_head=8,  n_embd=512, dropout=0.0, bias=True),
    "Large":  dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0, bias=True),
    "XL":     dict(n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True),
}

# Detect Hardware
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def load_checkpoint_robust(model, path):
    """
    Smart loader that handles:
    1. Metadata wrappers (extracts 'model_state_dict')
    2. torch.compile prefixes ('_orig_mod.')
    """
    print(f"📂 Loading weights from {path}...")
    
    # Load raw file
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    
    # 1. Unwrap metadata if present
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # Assume it's just weights

    # 2. Fix torch.compile prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:] # Strip "_orig_mod."
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
            
    # 3. Load into model
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"   Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    return model

def get_checkpoint_vocab_size(path):
    """Peek at checkpoint to find the actual trained vocab size"""
    print(f"🔍 Peeking at checkpoint to verify vocab size...")
    try:
        # Load just the map to avoid full RAM usage if possible, though torch.load loads all
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
        # Try to find the token embedding weight
        # Standard name: transformer.wte.weight
        # Compiled name: _orig_mod.transformer.wte.weight
        for key in ['transformer.wte.weight', '_orig_mod.transformer.wte.weight']:
            if key in state_dict:
                return state_dict[key].shape[0]
                
    except Exception as e:
        print(f"   ⚠️ Could not determine vocab size from file: {e}")
    
    return None

def generate():
    # 1. Load Tokenizer
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    vocab_size = len(tokenizer.vocab)
    print(f"   Tokenizer Vocab Size: {vocab_size}")

    # 2. Resolve Checkpoint Path Logic
    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint {ckpt_path} not found!")
        alt_path = f"ckpt_{MODEL_SIZE}_latest.pt"
        if os.path.exists(alt_path):
            print(f"   Found {alt_path} instead. Switching to that.")
            ckpt_path = alt_path
        else:
            return

    # 3. Smart Size Adjustment
    # The checkpoint might have a slightly different size (e.g. 1620 vs 1619)
    # We must initialize the model with the SIZE IN THE FILE, not the json size.
    trained_vocab_size = get_checkpoint_vocab_size(ckpt_path)
    
    if trained_vocab_size is not None and trained_vocab_size != vocab_size:
        print(f"   ⚠️ Mismatch detected! Model trained with {trained_vocab_size}, but vocab.json has {vocab_size}.")
        print(f"   🔧 Adjusting model initialization to {trained_vocab_size} to match weights.")
        vocab_size = trained_vocab_size

    # 4. Initialize Model Architecture
    if MODEL_SIZE not in model_configs:
        raise ValueError(f"Unknown model size: {MODEL_SIZE}")
    
    config = GPTConfig(vocab_size=vocab_size, block_size=256, **model_configs[MODEL_SIZE])
    model = GPT(config)
    model.to(DEVICE)

    # 5. Load Learned Weights
    model = load_checkpoint_robust(model, ckpt_path)
    model.eval()

    # 6. Generation Loop
    print(f"🎵 Generating {MAX_NEW_TOKENS} tokens (Temp={TEMPERATURE}, Penalty={REPETITION_PENALTY})...")
    
    start_id = tokenizer.vocab["<start>"]
    idx = torch.tensor([[start_id]], dtype=torch.long, device=DEVICE)

    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Crop context if it gets too long
            idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / TEMPERATURE
            
            # --- REPETITION PENALTY ---
            if REPETITION_PENALTY > 1.0:
                lookback = 20
                context = idx[0, -lookback:]
                for token_id in context:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= REPETITION_PENALTY
                    else:
                        logits[0, token_id] /= REPETITION_PENALTY
            
            # Top-K Sampling
            v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() == tokenizer.vocab.get("<end>", -1):
                print("   Model generated <end> token.")
                break

    print(f"   Done in {time.time()-start_time:.2f}s")

    # 7. Decode to Text
    generated_ids = idx[0].tolist()
    
    abc_tokens = []
    for i in generated_ids:
        # If the model predicts a padding index outside our known vocab, skip it
        if i >= len(tokenizer.reverse_vocab):
            continue
            
        word = tokenizer.reverse_vocab.get(str(i), "") 
        if not word: 
             word = tokenizer.reverse_vocab.get(i, "") 

        if word in ["<start>", "<pad>", "<unk>"]: continue
        if word == "<end>": break
        abc_tokens.append(word)
    
    output_text = "".join(abc_tokens)
    
    # FORMATTING FIX: Insert newlines after bars
    output_text = output_text.replace("|", "|\n")
    
    if "X:" not in output_text:
        header = "X:1\nT:Generated Song\nM:4/4\nL:1/8\nK:C\n"
        output_text = header + output_text
    
    print("\n-------- GENERATED ABC --------")
    print(output_text[:500] + ("..." if len(output_text) > 500 else ""))
    print("-------------------------------")
    
    out_file = "generated_song.abc"
    with open(out_file, "w") as f:
        f.write(output_text)
    print(f"💾 Saved to {out_file}")

if __name__ == "__main__":
    generate()