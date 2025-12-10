import torch
from torch.utils.data import IterableDataset
from .tokenizer import MusicTokenizer


class MusicStreamingDataset(IterableDataset):
    """
    An IterableDataset designed for training Large Language Models (LLMs) on massive datasets
    that cannot fit into memory.
    
    It streams data line-by-line from a text file, tokenizes it on the fly, and yields 
    contiguous batches of tokens for next-token prediction.
    """
    def __init__(self, file_path, vocab_path, block_size, max_tokens=None):
        """
        Args:
            file_path: Path to train.txt
            vocab_path: Path to vocab.json
            block_size: Sequence length (context window)
            max_tokens: Stop iteration after this many tokens have been yielded (for scaling studies).
        """
        self.file_path = file_path
        self.tokenizer = MusicTokenizer()
        self.tokenizer.load(vocab_path)
        self.block_size = block_size
        self.vocab_size = len(self.tokenizer.vocab)
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        tokens_yielded = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return

                if not line.strip(): continue

                tokens = self.tokenizer.tokenize(line)
                token_ids = [self.tokenizer.vocab.get(t, self.tokenizer.vocab["<unk>"]) for t in tokens]

                buffer.extend(token_ids)

                while len(buffer) >= self.block_size + 1:
                    chunk = buffer[:self.block_size + 1]
                    buffer = buffer[self.block_size:]

                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)

                    tokens_yielded += self.block_size
                    yield x, y