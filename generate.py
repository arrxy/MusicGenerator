import torch
import torch.nn.functional as F
import os
import re
import time
from src.model import GPT, GPTConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION ---
MODEL_SIZE = "Large"         # Must match the checkpoint filename
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}_extended.pt" 
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 300         # Length of the song
TEMPERATURE = 0.95            # 1.0 = Random/Creative, 0.8 = Focused/Safe
TOP_K = 600                  # Limit to top N most likely next notes
REPETITION_PENALTY = 1.0     # Reduce probability of recently used tokens (1.0 = No penalty)

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

def load_checkpoint_data(path):
    """
    Loads checkpoint and preprocesses it:
    1. Unwraps metadata
    2. Strips torch.compile prefixes
    3. Detects trained vocab size
    """
    print(f"📂 Loading checkpoint from {path}...")
    try:
        # Load the file once
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        return None, None

    # 1. Unwrap metadata if present
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("   Detected metadata wrapper. Extracting weights...")
        raw_state_dict = checkpoint['model_state_dict']
    else:
        raw_state_dict = checkpoint

    # 2. Fix torch.compile prefixes and normalize keys
    clean_state_dict = {}
    vocab_size_found = None
    
    for key, value in raw_state_dict.items():
        # Strip "_orig_mod." prefix from torch.compile
        if key.startswith('_orig_mod.'):
            new_key = key[10:] 
        else:
            new_key = key
            
        clean_state_dict[new_key] = value
        
        # Detect vocab size from embeddings
        if new_key == "transformer.wte.weight":
            vocab_size_found = value.shape[0]

    return clean_state_dict, vocab_size_found

def sanitize_abc(text):
    """
    Fixes common AI-generated ABC errors that break web players.
    """
    # 1. Clamp Excessive Octaves (3 or more ' or ,) to Double ('' or ,,)
    # Replaces a''' or a'''' with a'' 
    # Replaces C,,, or C,,,, with C,,
    # This ensures notes stay within the standard MIDI playable range
    text = re.sub(r"([a-gA-G])'{3,}", r"\1''", text)
    text = re.sub(r"([a-gA-G]),{3,}", r"\1,,", text)

    # 2. Fix Massive Rests (z96 -> z4)
    # Matches z followed by 2 or more digits (e.g., z96, z48)
    text = re.sub(r"z\d{2,}", "z4", text)

    # 3. Fix Layout (Insert newline every 4 bars if missing)
    if "|" in text and "\n" not in text:
        # Split by bars and rejoin with line breaks every 4 bars
        bars = text.split("|")
        new_text = ""
        for i, bar in enumerate(bars):
            new_text += bar + "|"
            if (i + 1) % 4 == 0:
                new_text += "\n"
        text = new_text

    # 4. Remove empty chords []
    text = text.replace("[]", "")

    return text

def generate():
    # 1. Load Tokenizer
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    default_vocab_size = len(tokenizer.vocab)
    print(f"   Tokenizer Vocab Size: {default_vocab_size}")

    # 2. Resolve Checkpoint Path
    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint {ckpt_path} not found!")
        alt_path = f"ckpt_{MODEL_SIZE}_latest.pt"
        if os.path.exists(alt_path):
            print(f"   Found {alt_path} instead. Switching to that.")
            ckpt_path = alt_path
        else:
            return

    # 3. Load Data & Detect Size
    state_dict, trained_vocab_size = load_checkpoint_data(ckpt_path)
    
    if state_dict is None:
        print("❌ Failed to load checkpoint data.")
        return

    # 4. Handle Vocab Size Mismatch
    # Use the size found in the checkpoint weights, otherwise use tokenizer size
    final_vocab_size = default_vocab_size
    if trained_vocab_size is not None and trained_vocab_size != default_vocab_size:
        print(f"   ⚠️ Mismatch detected! Model trained with {trained_vocab_size}, but vocab.json has {default_vocab_size}.")
        print(f"   🔧 Adjusting model initialization to {trained_vocab_size} to match weights.")
        final_vocab_size = trained_vocab_size

    # 5. Initialize Model
    if MODEL_SIZE not in model_configs:
        raise ValueError(f"Unknown model size: {MODEL_SIZE}")
    
    config = GPTConfig(vocab_size=final_vocab_size, block_size=256, **model_configs[MODEL_SIZE])
    model = GPT(config)
    model.to(DEVICE)

    # 6. Load Weights (Strict=False allows minor mismatches, but keys should now be clean)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"   Weights loaded. Missing keys: {len(msg.missing_keys)}")

    model.eval()

    # 7. Generation Loop
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

    # 8. Decode to Text
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
    
    # FORMATTING FIX: Insert newlines after bars (Initial pass)
    output_text = output_text.replace("|", "|\n")
    
    # --- SANITIZE OUTPUT FOR PLAYERS (NEW STEP) ---
    output_text = sanitize_abc(output_text)
    
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