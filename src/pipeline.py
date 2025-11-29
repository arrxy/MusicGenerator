import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .converter import MidiToAbcConverter
from .tokenizer import MusicTokenizer
import random


def process_single_file(file_path):
    converter = MidiToAbcConverter()
    if os.path.getsize(file_path) > 200 * 1024:  # 200KB limit
        return None
    return converter.convert(file_path)


class DataPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_dir = Path(cfg.paths.raw_dir)
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        all_midis = list(self.raw_dir.glob("**/*.mid"))
        total_files = len(all_midis)
        print(f"Found {total_files} MIDI files")

        valid_songs = []
        glitch_count = 0
        too_short_count = 0

        # SAFETY LIMITS
        MAX_ABC_LEN = 250000  # Approx 50k tokens
        MIN_ABC_LEN = 100  # Approx 20 tokens (Filter out empty/2-token files)

        with ProcessPoolExecutor(max_workers=self.cfg.processing.max_workers) as exc:
            futures = {exc.submit(process_single_file, p): p for p in all_midis}

            for future in tqdm(as_completed(futures), total=len(all_midis), desc="Converting"):
                result = future.result()
                if result:
                    res_len = len(result)

                    if res_len > MAX_ABC_LEN:
                        glitch_count += 1
                        continue

                    if res_len < MIN_ABC_LEN:
                        too_short_count += 1
                        continue

                    valid_songs.append(result)

                    if len(valid_songs) >= 100000000000:
                        print("Reached limit valid songs. Stopping early...")
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
        total_tokens_est = sum(len(s.split()) for s in valid_songs)
        avg_len = total_tokens_est / len(valid_songs) if valid_songs else 0
        print("\n" + "=" * 40)
        print("       PIPELINE STATISTICS        ")
        print("=" * 40)
        print(f"Total Input Files   : {total_files:,}")
        print(f"Valid Songs         : {len(valid_songs):,}")
        print(f"Discarded (Too Big) : {glitch_count:,}")
        print(f"Discarded (Too Small): {too_short_count:,}")
        print(f"Avg Tokens/Song     : {avg_len:.0f}")
        print(f"Vocabulary Size     : {vocab_size:,}")
        print("=" * 40 + "\n")

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
            content = "\n<|endoftext|>\n".join(data)
            with open(path, "w") as f:
                f.write(content)
            print(f"Saved {name}: {len(data)} songs")