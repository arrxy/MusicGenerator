import torch
import torch.nn.functional as F
import os
import re
import time
from src.model import GPT, GPTConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION ---
MODEL_SIZE = "XL"         # Must match the checkpoint filename
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}_extended.pt" 
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 512         # Length of the song
TEMPERATURE = 0.8            # 1.0 = Random/Creative, 0.8 = Focused/Safe
TOP_K = 600                  # Limit to top N most likely next notes
REPETITION_PENALTY = 1.0     # Reduce probability of recently used tokens (1.0 = No penalty)

# Model Architecture Registry (Must match training exactly)
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
    """Loads checkpoint and handles metadata/prefixes."""
    print(f"📂 Loading checkpoint from {path}...")
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        return None, None

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("   Detected metadata wrapper. Extracting weights...")
        raw_state_dict = checkpoint['model_state_dict']
    else:
        raw_state_dict = checkpoint

    clean_state_dict = {}
    vocab_size_found = None
    
    for key, value in raw_state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:] 
        else:
            new_key = key
        clean_state_dict[new_key] = value
        
        if new_key == "transformer.wte.weight":
            vocab_size_found = value.shape[0]

    return clean_state_dict, vocab_size_found

def sanitize_abc(text):
    """Fixes common AI-generated ABC errors."""
    text = re.sub(r"([a-gA-G])'{3,}", r"\1''", text)
    text = re.sub(r"([a-gA-G]),{3,}", r"\1,,", text)
    text = re.sub(r"z\d{2,}", "z4", text)
    
    if "|" in text and "\n" not in text:
        bars = text.split("|")
        new_text = ""
        for i, bar in enumerate(bars):
            new_text += bar + "|"
            if (i + 1) % 4 == 0:
                new_text += "\n"
        text = new_text

    text = text.replace("[]", "")
    return text

def encode_string(tokenizer, text):
    """
    Greedy tokenizer implementation to handle missing tokenizer.encode().
    Matches the longest possible token from the vocabulary at each step.
    """
    ids = []
    i = 0
    n = len(text)
    vocab = tokenizer.vocab
    
    while i < n:
        match = None
        match_len = 0
        
        # Try to match the longest token possible (up to 6 chars for ABC notation)
        # ABC tokens like "^C,," are rarely longer than 5-6 chars
        for l in range(min(6, n - i), 0, -1):
            substr = text[i : i + l]
            if substr in vocab:
                match = substr
                match_len = l
                break
        
        if match:
            ids.append(vocab[match])
            i += match_len
        else:
            # Character not found in vocab (skip or UNK)
            if "<unk>" in vocab:
                ids.append(vocab["<unk>"])
            # Optionally print warning for skipped chars:
            # print(f"⚠️ Skipping unknown char: '{text[i]}'")
            i += 1
            
    return ids

def setup_model():
    """Initializes model and tokenizer for inference."""
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    default_vocab_size = len(tokenizer.vocab)
    print(f"   Tokenizer Vocab Size: {default_vocab_size}")

    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint {ckpt_path} not found!")
        alt_path = f"ckpt_{MODEL_SIZE}_latest.pt"
        if os.path.exists(alt_path):
            print(f"   Found {alt_path} instead. Switching to that.")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError("No checkpoint found.")

    state_dict, trained_vocab_size = load_checkpoint_data(ckpt_path)
    if state_dict is None:
        raise ValueError("Failed to load checkpoint data.")

    final_vocab_size = default_vocab_size
    if trained_vocab_size is not None and trained_vocab_size != default_vocab_size:
        print(f"   ⚠️ Mismatch: Model {trained_vocab_size} vs Vocab {default_vocab_size}.")
        print(f"   🔧 Adjusting model to {trained_vocab_size}.")
        final_vocab_size = trained_vocab_size

    if MODEL_SIZE not in model_configs:
        raise ValueError(f"Unknown model size: {MODEL_SIZE}")
    
    config = GPTConfig(vocab_size=final_vocab_size, block_size=256, **model_configs[MODEL_SIZE])
    model = GPT(config)
    model.to(DEVICE)

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"   Weights loaded. Missing keys: {len(msg.missing_keys)}")
    model.eval()
    
    return model, tokenizer

def run_generation_loop(model, tokenizer, idx, filename_prefix="generated"):
    """Core generation loop used by both modes."""
    print(f"🎵 Generating {MAX_NEW_TOKENS} tokens (Temp={TEMPERATURE})...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
            
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / TEMPERATURE
            
            if REPETITION_PENALTY > 1.0:
                lookback = 20
                context = idx[0, -lookback:]
                for token_id in context:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= REPETITION_PENALTY
                    else:
                        logits[0, token_id] /= REPETITION_PENALTY
            
            v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() == tokenizer.vocab.get("<end>", -1):
                print("   Model generated <end> token.")
                break

    print(f"   Done in {time.time()-start_time:.2f}s")

    generated_ids = idx[0].tolist()
    abc_tokens = []
    
    for i in generated_ids:
        if i >= len(tokenizer.reverse_vocab): continue
        # Handle string vs int keys in reverse_vocab
        word = tokenizer.reverse_vocab.get(str(i), tokenizer.reverse_vocab.get(i, "")) 
        
        if word in ["<start>", "<pad>", "<unk>"]: continue
        if word == "<end>": break
        abc_tokens.append(word)
    
    output_text = "".join(abc_tokens)
    output_text = output_text.replace("|", "|\n")
    output_text = sanitize_abc(output_text)
    
    # Ensure header exists if we started from scratch
    # If we continued, the header is likely in the prompt, but we should prepend it if missing
    # Actually, for continuation, the output_text includes the prompt tokens if we convert from `idx`
    # Let's ensure the prompt is included.
    
    print(f"\n-------- {filename_prefix.upper()} ABC --------")
    print(output_text[:300] + "...")
    print("-------------------------------")
    
    out_file = f"{filename_prefix}_song.abc"
    with open(out_file, "w") as f:
        f.write(output_text)
    print(f"💾 Saved to {out_file}")

def generate_scratch():
    """Generates a song from nothing (<start> token)."""
    model, tokenizer = setup_model()
    start_id = tokenizer.vocab["<start>"]
    idx = torch.tensor([[start_id]], dtype=torch.long, device=DEVICE)
    run_generation_loop(model, tokenizer, idx, "scratch")

def generate_continuation(prompt_abc):
    """Continues a song from a given ABC string."""
    model, tokenizer = setup_model()
    print(f"🎹 Continuing from: {prompt_abc[:50]}...")
    
    # 1. Encode the prompt manually
    # Use fallback encode_string since tokenizer.encode might be missing
    if hasattr(tokenizer, 'encode'):
        try:
            prompt_ids = tokenizer.encode(prompt_abc)
        except Exception:
            prompt_ids = encode_string(tokenizer, prompt_abc)
    else:
        prompt_ids = encode_string(tokenizer, prompt_abc)

    if not prompt_ids:
        print("❌ Error: Could not encode prompt text.")
        return

    print(f"   Encoded {len(prompt_ids)} tokens.")

    # 2. Prepend <start> so the model knows context
    start_id = tokenizer.vocab["<start>"]
    input_ids = [start_id] + prompt_ids
    
    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    run_generation_loop(model, tokenizer, idx, "continued")

if __name__ == "__main__":
    # --- MODE SELECTION ---
    mode = "2"  # 1 = Scratch, 2 = Continue
    
    if mode == "1":
        generate_scratch()
    else:
        # Put your starting ABC here!
        my_start = """X:1759
T:F\"ur Elise
T:Bagatelle No.25 in A, WoO.59
C:Ludwig van Beethoven
O:Germany
Z:Transcribed by Frank Nordberg - http://www.musicaviva.com
F:http://abc.musicaviva.com/tunes/beethoven-ludwig-van/be059/be059-pno2.abc
V:1 Program 1 0 %Piano
V:2 Program 1 0 bass %Piano
M:3/8
L:1/16
Q:3/8=40
K:Am
V:1
e^d|e^deB=dc|A2 z CEA|B2 z E^GB|c2 z Ee^d|
V:2
z2|z6|A,,E,A, z z2|E,,E,^G, z z2|A,,E,A, z z2|
%
V:1
e^deB=dc|A2 z CEA|B2 z EcB|[1A2 z2:|[2A2z Bcd|
V:2
z6|A,,E,A, z z2|E,,E,^G, z z2|[1A,,E,A, z :|[2A,,E,A, z z2|
%
V:1
|:e3 Gfe|d3 Fed|c3 Edc|B2 z Ee z|z ee' z z ^d|
V:2
|:C,E,C z z2|G,,G,B, z z2|A,,E,A, z z2|E,,E,E z z E|e z z ^de z|
%
V:1
e z z ^ded|e^deB=dc|A2 z CEA|B2 zE^GB|c2 z Ee^d|
V:2
z ^de z z2|z6|A,,E,A, z z2|E,,E,^G, z z2|A,,E,A, z z2|
%
V:1
e^deB=dc|A2 z CEA|B2 z EcB|[1A2 z Bcd:|
V:2
z6|A,,E,A, z z2|E,,E,^G, z z2|[1A,,E,A, z z2:|
%
V:1
[2A2 z [Ec][Fc][EGc]|c4 f>e|e2d2 _b>a|agfedc|
V:2
[2A,,E,A, [_B,C][A,C][G,B,C]|F,A,CA,CA,|F,_B,DB,DB,|F,E[F,G,_B,]E[F,G,B,]E|
%
V:1
_B2A2 A/G/A/B/|c4 d^d|e3 efA|c4 d>B|
V:2
F,A,CA,CA,|F,A,CA,CA,|E,A,CA,[D,D]F,|G,EG,EG,F|
%
V:1
c/g/G/g/ A/g/B/g/ c/g/d/g/|e/g/c'/b/ a/g/f/e/ d/g/f/d/|c/g/G/g/ A/g/B/g/ c/g/d/g/|
V:2
[C2E2] z [FG][EG][DFG]|[C2E2G2] [F,2A,2][F,2A,2]|C2 z [FG][EG][DFG]|
%
V:1
e/g/c'/b/ a/g/f/e/ d/g/f/d/|e/f/e/^d/ e/B/e/d/ e/B/e/d/|e3 Be^d|e3 Be z|
V:2
[C2E2G2] [F,2A,2][G,2B,2]|[^G,2B,2] z2 z2|z6|z4 z ^d|
%
V:1
z ^de z z d|e^deB=dc|A2 z CEA|B2 z E^GB|c2 z Ee^d|
V:2
e z z ^de z|z6|A,,E,A, z z2|E,,E,^G, z z2|A,,E,A, z z2|
%
V:1
e^deB=dc|A2 z CEA|B2 z EcB|A2 z Bcd|e3 Gfe|
V:2
z6|A,,E,A, z z2|E,,E,^G, z z2|A,,E,A, z z2|C,E,C z z2|
%
V:1
d3 Fed|c3 Edc|B2 z Ee z|z ee' zz ^d|e z z ^ded|
V:2
G,,G,B, z z2|A,,E,A, z z2|E,,E,E z z E|e z z ^de z|z ^de z z2|
%
V:1
e^deB=dc|A2 z CEA|B2 z E^GB|c2 z Ee^d|e^deB=dc|A2 z CEA|
V:2
z6|A,,E,A, z z2|E,,E,^G, z"""
        generate_continuation(my_start)