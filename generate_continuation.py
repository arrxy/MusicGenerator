import torch
import torch.nn.functional as F
import os
import re
import time
import json

# Try imports, but allow fallback if src.tokenizer is missing
try:
    from src.model import GPT, GPTConfig

    try:
        from src.tokenizer import MusicTokenizer
    except ImportError:
        MusicTokenizer = None
except ImportError:
    print("⚠️ Warning: 'src' modules not found. Ensure you are in the project root.")
    MusicTokenizer = None

# --- CONFIGURATION ---
MODEL_SIZE = "XL"  # Changed to Tiny to match your existing checkpoint
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}_robust.pt"  # Changed to _robust.pt
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 50  # Length of the song
TEMPERATURE = 0.8  # 1.0 = Random/Creative, 0.8 = Focused/Safe
TOP_K = 600  # Limit to top N most likely next notes
REPETITION_PENALTY = 1.1  # Reduce probability of recently used tokens (1.0 = No penalty)

model_configs = {
    "Tiny": dict(n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True),
    "Small": dict(n_layer=6, n_head=6, n_embd=288, dropout=0.0, bias=True),
    "Medium": dict(n_layer=8, n_head=8, n_embd=512, dropout=0.0, bias=True),
    "Large": dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0, bias=True),
    "XL": dict(n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True),
}

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
class JSONTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            self.vocab = {ch: i for i, ch in enumerate(data)}
        else:
            self.vocab = data

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.reverse_vocab = {int(k): v for k, v in self.reverse_vocab.items()}


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
        elif key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        clean_state_dict[new_key] = value

        if new_key == "transformer.wte.weight":
            vocab_size_found = value.shape[0]

    return clean_state_dict, vocab_size_found


def sanitize_abc(text):
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
            substr = text[i: i + l]
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

    if MusicTokenizer:
        tokenizer = MusicTokenizer()
        print("   Using src.MusicTokenizer")
    else:
        tokenizer = JSONTokenizer()
        print("   Using fallback JSONTokenizer")

    tokenizer.load(VOCAB_PATH)
    default_vocab_size = len(tokenizer.vocab)
    print(f"   Tokenizer Vocab Size: {default_vocab_size}")

    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found!")
        alts = [f"ckpt_{MODEL_SIZE}_extended.pt", f"ckpt_{MODEL_SIZE}_latest.pt"]
        found = False
        for alt in alts:
            if os.path.exists(alt):
                print(f"   Found {alt} instead. Switching to that.")
                ckpt_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No checkpoint found. Expected {CHECKPOINT_PATH}")

    state_dict, trained_vocab_size = load_checkpoint_data(ckpt_path)
    if state_dict is None:
        raise ValueError("Failed to load checkpoint data.")

    final_vocab_size = default_vocab_size
    if trained_vocab_size is not None and trained_vocab_size != default_vocab_size:
        print(f"   Adjusting vocab size to match checkpoint: {trained_vocab_size}")
        final_vocab_size = trained_vocab_size

    if MODEL_SIZE not in model_configs:
        raise ValueError(f"Unknown model size: {MODEL_SIZE}")

    # Init Model
    config = GPTConfig(vocab_size=final_vocab_size, block_size=256, **model_configs[MODEL_SIZE])
    model = GPT(config)
    model.to(DEVICE)

    # Load Weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"   Weights loaded. Missing keys: {len(msg.missing_keys)}")

    # Compile if possible
    if os.name == 'posix':
        try:
            model = torch.compile(model)
        except Exception:
            pass

    model.eval()
    return model, tokenizer


def run_generation_loop(model, tokenizer, idx, filename_prefix="generated"):
    print(f"   Generating up to {MAX_NEW_TOKENS} tokens...")
    start_time = time.time()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # Crop context if it gets too long
            idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / TEMPERATURE

            # Repetition Penalty
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

            # Check for End Token
            # Handle both string-keyed and int-keyed vocab
            end_token_id = tokenizer.vocab.get("<end>")
            if end_token_id is not None and idx_next.item() == end_token_id:
                print("   Model generated <end> token.")
                break

    print(f"   Done in {time.time() - start_time:.2f}s")

    # Decode
    generated_ids = idx[0].tolist()
    abc_tokens = []

    for i in generated_ids:
        # Safe decoding
        if hasattr(tokenizer, 'reverse_vocab'):
            word = tokenizer.reverse_vocab.get(i, tokenizer.reverse_vocab.get(str(i), ""))
        else:
            # Fallback if reverse_vocab isn't set up standardly
            word = ""
            for k, v in tokenizer.vocab.items():
                if v == i:
                    word = k
                    break

        if word in ["<start>", "<pad>", "<unk>", ""]: continue
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
    try:
        model, tokenizer = setup_model()
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    print(f"Continuing from: {prompt_abc[:50]}...")

    # Try using .encode() if available, else manual
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

    # Prepend Start Token if available
    start_id = tokenizer.vocab.get("<start>")
    input_ids = []
    if start_id is not None:
        input_ids.append(start_id)
    input_ids.extend(prompt_ids)

    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    run_generation_loop(model, tokenizer, idx, "continued")


if __name__ == "__main__":
    fur_elise = \
    """
   X:1759
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
    """
    happy_birthday = """
X: 3
T:Happy Birthday to You
M:3/4
L:1/8
K:G
D>D | E2 D2 G2 | F4 D>D | E2 D2 A2 | G4 D>D | d2 B2 G2 | (F2 E2) c>c |
B2 G2 A2 | G6 |]
"""

    generate_continuation(happy_birthday)