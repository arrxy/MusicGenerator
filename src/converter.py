import subprocess
import os
from pathlib import Path
import logging
import tempfile

logger = logging.getLogger(__name__)

class MidiToAbcConverter:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def convert(self, midi_path: Path) -> str | None:
        """
        Converts MIDI to ABC using the native 'midi2abc' CLI tool.
        Returns the ABC string or None if conversion failed.
        """
        abc_path = None
        try:
            # -b 100: sets bars per line (prevents massive long lines)
            # -o -  : outputs to stdout instead of file
            # cmd = ["midi2abc", str(midi_path), "-b", "100", "-o", "-"]
            project_root = Path(__file__).parent
            with tempfile.NamedTemporaryFile(delete=False, suffix=".abc", dir=project_root) as tmp:
                abc_path = tmp.name
            # cmd = ["midi2abc", "-o", "-", "-b", "100", str(midi_path)]
            cmd = [
                "midi2abc",
                str(midi_path),
                "-b", "100",
                "-o", abc_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False
            )

            if result.returncode != 0:
                os.remove(abc_path)
                return None

            with open(abc_path, "r") as f:
                abc_content = f.read()

            os.remove(abc_path)

            if "X:" not in abc_content or "K:" not in abc_content:
                return None
                
            return self._clean_abc(abc_content)

        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logger.error(f"Error converting {midi_path}: {e}")
            if abc_path:
                os.remove(abc_path)
            return None

    def _clean_abc(self, content: str) -> str:
        """
        CRITICAL FIX: Removes Titles (T:) and Metadata that cause 
        vocabulary explosion. Only keeps musical structure.
        """
        lines = content.splitlines()
        cleaned = []
        
        # Only allow headers that describe MUSIC, not file info
        allowed_headers = ("K:", "M:", "L:", "Q:", "V:", "P:")
        
        for l in lines:
            l = l.strip()
            if not l: continue
            if l.startswith('%'): continue # Remove comments
            
            # If it looks like a header X: ...
            if len(l) > 2 and l[1] == ':':
                if l.startswith(allowed_headers):
                    cleaned.append(l)
                # We deliberately SKIP 'T:', 'X:', 'Z:' to prevent data leakage
            else:
                # It is notes/music
                cleaned.append(l)
                
        return "\n".join(cleaned)