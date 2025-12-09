import torch
import torch.nn.functional as F
import os
import time
from src.model import GPT, GPTConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION ---
MODEL_SIZE = "Small"         # Must match the checkpoint filename (e.g., ckpt_Small.pt)
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}.pt"
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 512         # Length of the song
TEMPERATURE = 0.8            # 1.0 = Random/Creative, 0.8 = Focused/Safe
TOP_K = 600                  # Limit to top N most likely next notes
REPETITION_PENALTY = 1.5     # Reduce probability of recently used tokens (1.0 = No penalty)

# Model Architecture Registry (Must match training exactly)
# FIX: Set bias=True to match your saved checkpoint
model_configs = {
    "Tiny":   dict(n_layer=4,  n_head=4,  n_embd=128, dropout=0.0, bias=True),
    "Small":  dict(n_layer=6,  n_head=6,  n_embd=288, dropout=0.0, bias=True),
    "Medium": dict(n_layer=8,  n_head=8,  n_embd=512, dropout=0.0, bias=True),
    "Large":  dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0, bias=True),
    "XL":     dict(n_layer=12, n_head=12, n_embd=864, dropout=0.0, bias=True),
}

# Detect Hardware
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
DEVICE = 'cpu'

def validate_abc(text):
    """
    Post-processing to fix common syntax errors in AI-generated ABC.
    - Removes empty chords []
    - Closes unclosed chords [C E G
    - Removes orphaned closing brackets ]
    """
    # 1. Remove empty chords
    text = text.replace("[]", "")
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip headers
        if len(line) > 1 and line[1] == ':':
            cleaned_lines.append(line)
            continue
            
        # Fix bracket matching
        new_line = ""
        in_chord = False
        
        for char in line:
            if char == '[':
                if in_chord: # AI forgot to close previous chord
                     new_line += '][' 
                else:
                    in_chord = True
                    new_line += char
            elif char == ']':
                if in_chord:
                    in_chord = False
                    new_line += char
                else:
                    # Orphaned closing bracket, skip it
                    pass 
            else:
                new_line += char
        
        # If line ends while inside a chord, close it
        if in_chord:
            new_line += ']'
            
        cleaned_lines.append(new_line)
        
    return "\n".join(cleaned_lines)

def generate():
    print(f"🎹 Loading {MODEL_SIZE} model from {CHECKPOINT_PATH}...")
    
    # 1. Load Tokenizer
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    vocab_size = len(tokenizer.vocab)
    print(f"   Vocab Size: {vocab_size}")

    # 2. Initialize Model Architecture
    if MODEL_SIZE not in model_configs:
        raise ValueError(f"Unknown model size: {MODEL_SIZE}")
    
    # Initialize a blank model with the right shape
    config = GPTConfig(vocab_size=vocab_size, block_size=256, **model_configs[MODEL_SIZE])
    model = GPT(config)

    # 3. Load Learned Weights
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # Load weights onto CPU first to avoid memory complications, then move to Device
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("   Model loaded successfully.")

    # 4. Generation Loop
    print(f"🎵 Generating {MAX_NEW_TOKENS} tokens (Temp={TEMPERATURE}, Penalty={REPETITION_PENALTY})...")
    
    # Start with the specific <start> token ID
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
                # Get the last few tokens generated
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
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if we hit <end>
            if idx_next.item() == tokenizer.vocab.get("<end>", -1):
                print("   Model generated <end> token.")
                break

    print(f"   Done in {time.time()-start_time:.2f}s")

    # 5. Decode to Text
    generated_ids = idx[0].tolist()
    
    # Convert IDs back to ABC strings
    abc_tokens = []
    for i in generated_ids:
        # Skip special tokens in the output text
        # JSON keys are strings, but generated_ids are ints
        word = tokenizer.reverse_vocab.get(str(i), "") 
        if not word: 
             word = tokenizer.reverse_vocab.get(i, "") # Try int key just in case

        if word in ["<start>", "<pad>", "<unk>"]: continue
        if word == "<end>": break
        abc_tokens.append(word)
    
    # Join them
    output_text = "".join(abc_tokens)
    
    # FORMATTING FIX: Insert newlines after bars to prevent parser errors
    # ABCjs hates lines that are 400+ characters long
    output_text = output_text.replace("|", "|\n")
    
    # Add a minimal header if the model didn't generate one
    if "X:" not in output_text:
        header = "X:1\nT:Generated Song\nM:4/4\nL:1/8\nK:C\n"
        output_text = header + output_text

    # --- VALIDATION STEP ---
    # output_text = validate_abc(output_text)
    
    print("\n-------- GENERATED ABC --------")
    # Print preview
    print(output_text[:500] + ("..." if len(output_text) > 500 else ""))
    print("-------------------------------")
    
    # Save to file
    out_file = "generated_song.abc"
    with open(out_file, "w") as f:
        f.write(output_text)
    print(f"💾 Saved to {out_file}")

if __name__ == "__main__":
    generate()