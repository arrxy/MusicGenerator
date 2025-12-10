import torch
import torch.nn.functional as F
import os
import re
import time
from src.rnn_model import RNNModel, RNNConfig
from src.tokenizer import MusicTokenizer

# --- CONFIGURATION ---
MODEL_NAME = "Large"          # Tiny, Small, Medium, Large
CHECKPOINT_PATH = f"ckpt_rnn_{MODEL_NAME}.pt"
VOCAB_PATH = "data/processed/vocab.json"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.8
TOP_K = 50

# Detect Hardware
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

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

def generate():
    print(f"🎹 Loading RNN {MODEL_NAME}...")

    # 1. Load Tokenizer
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab not found: {VOCAB_PATH}")
    tokenizer = MusicTokenizer()
    tokenizer.load(VOCAB_PATH)
    
    # 2. Load Checkpoint & Config
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint {CHECKPOINT_PATH} not found. Train it first!")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint['config']  # Load the saved config object
    state_dict = checkpoint['state_dict']
    
    print(f"   Architecture: Hidden={config.hidden_dim}, Layers={config.n_layers}")
    
    # 3. Init Model
    model = RNNModel(config)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 4. Generation Loop
    print(f"🎵 Generating {MAX_NEW_TOKENS} tokens...")
    start_id = tokenizer.vocab["<start>"]
    
    # RNNs expect (batch, seq) if batch_first=True
    idx = torch.tensor([[start_id]], dtype=torch.long, device=DEVICE)
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            # For simple generation, we re-feed the sequence. 
            # (Stateful generation is faster but requires modifying the model class API)
            # Crop context if it gets too long for efficiency, though RNNs can handle arbitrary length
            idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            # Focus on last step
            logits = logits[:, -1, :] / TEMPERATURE
            
            # Top-K
            v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx_next.item() == tokenizer.vocab.get("<end>", -1):
                print("   Model generated <end> token.")
                break

    print(f"   Done in {time.time()-start_time:.2f}s")

    # 5. Decode
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
    
    if "X:" not in output_text:
        header = "X:1\nT:RNN Generated Song\nM:4/4\nL:1/8\nK:C\n"
        output_text = header + output_text

    print("\n-------- RNN OUTPUT --------")
    print(output_text[:500] + "...")
    print("----------------------------")
    
    out_file = f"rnn_{MODEL_NAME}_song.abc"
    with open(out_file, "w") as f:
        f.write(output_text)
    print(f"💾 Saved to {out_file}")

if __name__ == "__main__":
    generate()