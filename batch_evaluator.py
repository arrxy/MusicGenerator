import torch
import torch.nn.functional as F
import os
import re
import time
import json
import csv
import sys
from src.model import GPT, GPTConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION (Match settings from generate_song.py) ---
MODEL_SIZE = "Large"
CHECKPOINT_PATH = f"ckpt_{MODEL_SIZE}_extended.pt"
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.95
TOP_K = 600
REPETITION_PENALTY = 1.0

# --- BATCH EVALUATION CONFIG ---
NUM_SAMPLES = 100  # How many tunes to generate for evaluation
OUTPUT_DIR = "batch_evaluation_results"

model_configs = {
    "Tiny": dict(n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True),
    "Small": dict(n_layer=6, n_head=6, n_embd=288, dropout=0.0, bias=True),
    "Medium": dict(n_layer=8, n_head=8, n_embd=512, dropout=0.0, bias=True),
    "Large": dict(n_layer=10, n_head=10, n_embd=640, dropout=0.0, bias=True),
    "XL": dict(n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True),
}

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


# --- Helper Functions (Copied/Adapted from generate_song.py) ---

def load_checkpoint_data(path):
    # ... [Implementation of load_checkpoint_data] ...
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        return None, None
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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
    # ... [Implementation of sanitize_abc] ...
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
        text = new_text.rstrip("|\n")
        if text and text[-1] != '|':
            text += '|'
    text = text.replace("[]", "")
    return text


def validate_abc_syntax(abc_text):
    # ... [Implementation of validate_abc_syntax] ...
    validity = {}
    required_headers = ['X:', 'T:', 'M:', 'L:', 'K:']
    is_valid = True

    for header in required_headers:
        if re.search(r"^\s*" + re.escape(header), abc_text, re.MULTILINE):
            validity[header] = "Found"
        else:
            validity[header] = "Missing"
            is_valid = False

    if not re.search(r"[a-gA-G\|\n]", abc_text):
        validity['Body'] = "Missing notes/bars"
        is_valid = False
    else:
        validity['Body'] = "Present"

    return is_valid, validity

def batch_generate_and_evaluate(model, tokenizer, config):
    start_id = tokenizer.vocab.get("<start>", 0)

    total_valid = 0
    all_results = []

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"\nStarting batch generation for {NUM_SAMPLES} samples...")

    for i in range(NUM_SAMPLES):
        idx = torch.tensor([[start_id]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]

                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / TEMPERATURE

                if REPETITION_PENALTY > 1.0:
                    lookback = 20
                    context = idx[0, -lookback:].tolist()
                    for token_id in set(context):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= REPETITION_PENALTY
                        else:
                            logits[0, token_id] *= REPETITION_PENALTY

                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, idx_next), dim=1)

                if idx_next.item() == tokenizer.vocab.get("<end>", -1):
                    break

        generated_ids = idx[0].tolist()
        abc_tokens = [tokenizer.reverse_vocab.get(str(i), "") or tokenizer.reverse_vocab.get(i, "") for i in
                      generated_ids]
        abc_tokens = [w for w in abc_tokens if w not in ["<start>", "<pad>", "<unk>", "<end>"]]
        output_text = "".join(abc_tokens)

        output_text = output_text.replace("|\n", "|")
        output_text = output_text.replace("|", "|\n")
        output_text = sanitize_abc(output_text)

        if "X:" not in output_text and output_text.strip():
            header = "X:1\nT:Generated Song\nM:4/4\nL:1/8\nK:C\n"
            final_text = header + output_text
        else:
            final_text = output_text

        is_valid, _ = validate_abc_syntax(final_text)

        if is_valid:
            total_valid += 1
            status = "Valid"
        else:
            status = "Invalid"

        filename = f"tune_{i + 1}_{status}.abc"
        file_path = os.path.join(OUTPUT_DIR, filename)
        with open(file_path, "w") as f:
            f.write(final_text)

        all_results.append([i + 1, status, filename])

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{NUM_SAMPLES} samples. Current valid count: {total_valid}")

    validity_percentage = (total_valid / NUM_SAMPLES) * 100

    print("\n" + "=" * 50)
    print("         BATCH EVALUATION COMPLETE")
    print(f"  Total Samples Generated: {NUM_SAMPLES}")
    print(f"  Syntactically Valid Tunes: {total_valid}")
    print(f"  PERCENTAGE VALID: {validity_percentage:.2f}%")
    print("=" * 50)
    csv_path = os.path.join(OUTPUT_DIR, 'summary_report.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Status', 'Filename'])
        writer.writerows(all_results)

    print(f"Detailed results saved in '{OUTPUT_DIR}/'")
    print(f"Summary saved to '{csv_path}'")
    return validity_percentage

def main():
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: Vocab file not found at {VOCAB_PATH}. Cannot initialize tokenizer.")
        return

    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    default_vocab_size = len(tokenizer.vocab)

    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!")
        return

    state_dict, trained_vocab_size = load_checkpoint_data(ckpt_path)
    if state_dict is None:
        print("Failed to load checkpoint data.")
        return

    final_vocab_size = trained_vocab_size if trained_vocab_size is not None and trained_vocab_size != default_vocab_size else default_vocab_size

    if MODEL_SIZE not in model_configs:
        print(f"Error: Unknown model size: {MODEL_SIZE}")
        return

    config = GPTConfig(vocab_size=final_vocab_size, block_size=256, **model_configs[MODEL_SIZE])
    model = GPT(config)
    model.to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    batch_generate_and_evaluate(model, tokenizer, config)


if __name__ == "__main__":
    try:
        from src.model import GPT, GPTConfig
        from src.tokenizer import MusicTokenizer

        main()
    except ImportError as e:
        print(f"\nCRITICAL ERROR: Failed to import model components ({e}).")
        print("Please ensure 'src/model.py' and 'src/tokenizer.py' are correctly defined and accessible.")