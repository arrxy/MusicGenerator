import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .converter import MidiToAbcConverter
from .tokenizer import MusicTokenizer
import random

def process_single_file(file_path):
    """
    Worker function. Must be top-level for pickling on macOS.
    """
    converter = MidiToAbcConverter()
    # Basic size check before invoking subprocess
    if os.path.getsize(file_path) > 200 * 1024: # 200KB limit
        return None
    return converter.convert(file_path)

class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_dir = Path(cfg.paths.raw_dir)
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # 1. Gather Files
        all_midis = list(self.raw_dir.glob("**/*.mid"))
        print(f"Found {len(all_midis)} MIDI files")

        # 2. Parallel Conversion
        valid_songs = []
        
        # Using M4 cores efficiently
        # chunksize helps reduce IPC (Inter-Process Communication) overhead
        with ProcessPoolExecutor(max_workers=self.cfg.processing.max_workers) as exc:
            futures = {exc.submit(process_single_file, p): p for p in all_midis}
            
            for future in tqdm(as_completed(futures), total=len(all_midis), desc="Converting"):
                result = future.result()
                if result:
                    valid_songs.append(result)
                    if len(valid_songs) >= 100000000000:
                        print("Reached 100000000000 valid songs. Stopping early...")
                        for f in futures:
                            f.cancel()
                        break

        print(f"Successfully converted {len(valid_songs)} songs.")

        # 3. Tokenization & Vocab Building
        tokenizer = MusicTokenizer()
        print("Building vocabulary...")
        vocab_size = tokenizer.build_vocab(valid_songs)
        print(f"Vocabulary Size: {vocab_size}")
        tokenizer.save(self.output_dir / "vocab.json")

        # 4. Splitting
        self._save_splits(valid_songs)

    def _save_splits(self, songs):
        random.seed(42)
        random.shuffle(songs)
        
        n = len(songs)
        train_end = int(n * 0.98)
        val_end = int(n * 0.99)
        
        splits = {
            "train.txt": songs[:train_end],
            "val.txt": songs[train_end:val_end],
            "test.txt": songs[val_end:]
        }
        
        for name, data in splits.items():
            path = self.output_dir / name
            # Using <|endoftext|> as separator for GPT-style training
            content = "\n<|endoftext|>\n".join(data)
            with open(path, "w") as f:
                f.write(content)
            print(f"Saved {name}: {len(data)} songs")