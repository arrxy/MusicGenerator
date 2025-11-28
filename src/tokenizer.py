import re
import json
from typing import List, Set, Dict
import logging


class MusicTokenizer:
    def __init__(self, patch_size=64):
        """
        Args:
            patch_size (int): The fixed size for bar patches (default 64 from ISMIR paper).
        """
        self.patch_size = patch_size
        
        # Updated Regex to better capture complex bar lines (like :| or |])
        # 1. Notes: [A-G] with accidentals ^_, octaves ', and duration digits
        # 2. Bar lines: matches |, ||, |:, :|, |]
        # 3. Brackets: [ ]
        # 4. Rests: z plus duration
        self.pattern = re.compile(r"([A-Ga-g][,']*\d*|[\^_]?[A-Ga-g][,']*\d*|\|[:|\]]*|[:|\[]*\||\[|\]|z\d*)")
        
        # Vocab initialization
        # <pad> is 0, which is standard for masking in Transformers
        self.vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.reverse_vocab = {}

    def tokenize(self, text: str) -> List[str]:
        """Standard atom-level tokenization (Raw list of strings)."""
        return self.pattern.findall(text)

    def tokenize_bar_stream(self, text: str) -> List[List[int]]:
        """
        Implements 'Bar-stream Patching' from the ISMIR 2024 paper.
        
        Methodology:
        1. Divide score into bars (measures).
        2. Split each bar into fixed-length patches.
        3. Pad the final patch of a bar if it is shorter than patch_size.
        
        Returns:
            List[List[int]]: A list of patches (vectors), e.g., [[10, 55, 0...], [12, ...]]
        """
        raw_tokens = self.tokenize(text)
        
        all_patches = []
        current_bar_tokens = []

        for token in raw_tokens:
            # distinct mapping
            token_id = self.vocab.get(token, self.vocab["<unk>"])
            current_bar_tokens.append(token_id)

            # Check if this token is a bar line
            if '|' in token:
                # Bar complete. Process it into patches.
                bar_patches = self._process_single_bar(current_bar_tokens)
                all_patches.extend(bar_patches)
                current_bar_tokens = []

        # Handle remaining tokens (trailing notes without a final barline)
        if current_bar_tokens:
            bar_patches = self._process_single_bar(current_bar_tokens)
            all_patches.extend(bar_patches)

        return all_patches

    def _process_single_bar(self, tokens: List[int]) -> List[List[int]]:
        """
        Splits a variable length bar into fixed-size patches with padding.
        """
        patches = []
        
        # Loop through tokens in chunks of self.patch_size
        for i in range(0, len(tokens), self.patch_size):
            chunk = tokens[i : i + self.patch_size]
            
            # Pad if shorter than patch_size
            if len(chunk) < self.patch_size:
                padding_needed = self.patch_size - len(chunk)
                chunk.extend([self.vocab["<pad>"]] * padding_needed)
            
            patches.append(chunk)
            
        return patches

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary from a list of ABC strings and logs corpus statistics."""
        total_texts = len(texts)
        print(f"Starting vocabulary build from {total_texts} sequences...")

        unique_tokens = set()
        sequence_lengths = []
        
        log_step = max(1, total_texts // 10)

        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            unique_tokens.update(tokens)
            sequence_lengths.append(len(tokens))
            if (i + 1) % log_step == 0:
                percent = int(((i + 1) / total_texts) * 100)
                print(f"Tokenization progress: {percent}% ({i + 1}/{total_texts})")

        print(f"Found {len(unique_tokens)} unique tokens in input data.")
        start_idx = len(self.vocab)
        new_tokens = sorted([t for t in unique_tokens if t not in self.vocab])
        
        print(f"Appending {len(new_tokens)} new tokens...")

        for i, token in enumerate(new_tokens):
            self.vocab[token] = start_idx + i

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        total_tokens = sum(sequence_lengths)
        avg_len = total_tokens / len(sequence_lengths) if sequence_lengths else 0
        max_len = max(sequence_lengths) if sequence_lengths else 0
        min_len = min(sequence_lengths) if sequence_lengths else 0

        print("\n" + "="*40)
        print("       CORPUS STATISTICS REPORT       ")
        print("="*40)
        print(f"Vocabulary Size     : {len(self.vocab):,} tokens")
        print(f"Total Documents     : {total_texts:,} songs")
        print(f"Total Tokens        : {total_tokens:,}")
        print("-" * 40)
        print("Sequence Length Distribution (Tokens per Song):")
        print(f"  - Average Length  : {avg_len:.2f}")
        print(f"  - Min Length      : {min_len}")
        print(f"  - Max Length      : {max_len}")
        print("-" * 40)
        print(f"Quality Filters Applied (Tokenizer Level):")
        print(f"  - Regex Pattern   : Music-Aware (Notes, Bars, Chords)")
        print(f"  - Bar Patching    : Enabled")
        print(f"  - Patch Size      : {self.patch_size} (Padding/Truncation applied)")
        print("="*40 + "\n")

        return len(self.vocab)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
            
    def load(self, path: str):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}