import re
from typing import List, Dict
import json

class MusicTokenizer:
    def __init__(self):
        # Captures: Notes (C#,,4), Bars (|), Structure ([), Numbers, Rests (z)
        self.pattern = re.compile(r"([A-Ga-g][,']*\d*|[\^_]?[A-Ga-g][,']*\d*|\|+|\[|\]|\d+|z\d*)")
        self.vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.reverse_vocab = {}

    def tokenize(self, text: str) -> List[str]:
        """Splits ABC string into list of tokens."""
        return self.pattern.findall(text)

    def build_vocab_(self, texts: List[str]):
        """Builds vocabulary from a list of ABC strings."""
        unique_tokens = set()
        for text in texts:
            unique_tokens.update(self.tokenize(text))
        
        # Add new tokens starting from index 4
        start_idx = len(self.vocab)
        for i, token in enumerate(sorted(unique_tokens)):
            self.vocab[token] = start_idx + i
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        return len(self.vocab)

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary from a list of ABC strings."""
        total_texts = len(texts)
        print(f"Starting vocabulary build from {total_texts} sequences...")

        unique_tokens = set()
        log_step = max(1, total_texts // 10)

        for i, text in enumerate(texts):
            unique_tokens.update(self.tokenize(text))

            # Log progress every ~10%
            if (i + 1) % log_step == 0:
                percent = int(((i + 1) / total_texts) * 100)
                print(f"Tokenization progress: {percent}% ({i + 1}/{total_texts})")

        print(f" Found {len(unique_tokens)} unique tokens in input data.")

        # Add new tokens starting from index 4
        start_idx = len(self.vocab)
        print(f"Current vocabulary size: {start_idx}. Appending new tokens...")

        for i, token in enumerate(sorted(unique_tokens)):
            self.vocab[token] = start_idx + i

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        print(f"Vocabulary updated. Final size: {len(self.vocab)} tokens.")
        return len(self.vocab)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)

            
    def load(self, path: str):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}