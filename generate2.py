import torch
import torch.nn.functional as F
import os
import re
import time
from src.model import GPT, GPTConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION ---
MODEL_SIZE = "XL"
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}_extended.pt" 
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 512         # Length of the song
TEMPERATURE = 0.8            # 1.0 = Random/Creative, 0.8 = Focused/Safe
TOP_K = 600                  # Limit to top N most likely next notes
REPETITION_PENALTY = 1.0     # Reduce probability of recently used tokens (1.0 = No penalty)

model_configs = {
    "Tiny":   dict(n_layer=4,  n_head=4,  n_embd=128, dropout=0.0, bias=True),
    "Small":  dict(n_layer=6,  n_head=6,  n_embd=288, dropout=0.0, bias=True),
    "Medium": dict(n_layer=8,  n_head=8,  n_embd=512, dropout=0.0, bias=True),
    "Large":  dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0, bias=True),
    "XL":     dict(n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True),
}

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def load_checkpoint_data(path):
    print(f"Loading checkpoint from {path}...")
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
    ids = []
    i = 0
    n = len(text)
    vocab = tokenizer.vocab
    
    while i < n:
        match = None
        match_len = 0
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
            if "<unk>" in vocab:
                ids.append(vocab["<unk>"])
            i += 1
            
    return ids

def setup_model():
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    default_vocab_size = len(tokenizer.vocab)
    print(f"   Tokenizer Vocab Size: {default_vocab_size}")

    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found!")
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
        word = tokenizer.reverse_vocab.get(str(i), tokenizer.reverse_vocab.get(i, "")) 
        
        if word in ["<start>", "<pad>", "<unk>"]: continue
        if word == "<end>": break
        abc_tokens.append(word)
    
    output_text = "".join(abc_tokens)
    output_text = output_text.replace("|", "|\n")
    output_text = sanitize_abc(output_text)

    print(f"\n-------- {filename_prefix.upper()} ABC --------")
    print(output_text[:300] + "...")
    print("-------------------------------")
    
    out_file = f"{filename_prefix}_song.abc"
    with open(out_file, "w") as f:
        f.write(output_text)
    print(f"Saved to {out_file}")

def generate_continuation(prompt_abc):
    """Continues a song from a given ABC string."""
    model, tokenizer = setup_model()
    print(f"Continuing from: {prompt_abc[:50]}...")
    if hasattr(tokenizer, 'encode'):
        try:
            prompt_ids = tokenizer.encode(prompt_abc)
        except Exception:
            prompt_ids = encode_string(tokenizer, prompt_abc)
    else:
        prompt_ids = encode_string(tokenizer, prompt_abc)

    if not prompt_ids:
        print("Error: Could not encode prompt text.")
        return

    print(f"   Encoded {len(prompt_ids)} tokens.")
    start_id = tokenizer.vocab["<start>"]
    input_ids = [start_id] + prompt_ids
    
    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    run_generation_loop(model, tokenizer, idx, "continued")

if __name__ == "__main__":
    my_start = """X:1
T:Abercairney House [1]
M:C
L:1/8
R:Reel
C:Nathaniel Gow
B:John McLachlan - Piper’s Assistant (1854, No. 79, p. 45)
Z:AK/Fiddler’s Companion
K:Amix
c/d/|eAcA eAca|dGBG dGBg|eAcA eAca|gegB B<AA:|
g|aega eace|decg BGdB|aega eace|dBgB A2Ag|
aega eace|decg BGdB|eAcA eAca|gegB B<AA||"""
    generate_continuation(my_start)