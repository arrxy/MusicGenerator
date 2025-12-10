import subprocess
import os
from pathlib import Path
import logging
import tempfile

logger = logging.getLogger(__name__)


def _clean_abc(content: str) -> str:
    """
    Cleans ABC content by removing unwanted headers and comments.
    Only allow headers that describe MUSIC, not file info Other headers like T: (title), C: (composer) are omitted as they were creating garbage tokens
    Allowed headers are Key, Meter, Unit Note Length, Tempo, Voice, Part
    """
    lines = content.splitlines()
    cleaned = []
    allowed_headers = ("K:", "M:", "L:", "Q:", "V:", "P:")

    for l in lines:
        l = l.strip()
        if not l: continue
        if l.startswith('%'): continue

        if len(l) > 2 and l[1] == ':':
            if l.startswith(allowed_headers):
                cleaned.append(l)
        else:
            cleaned.append(l)

    return "\n".join(cleaned)


class MidiToAbcConverter:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def convert(self, midi_path: Path) -> str | None:
        """
        Converts MIDI to ABC using the native 'midi2abc' CLI tool.
        Returns the ABC string or None if conversion failed.
        It creates a temporary file for the ABC output and deletes it after reading.
        """
        abc_path = None
        try:
            project_root = Path(__file__).parent
            with tempfile.NamedTemporaryFile(delete=False, suffix=".abc", dir=project_root) as tmp:
                abc_path = tmp.name

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
                
            return _clean_abc(abc_content)

        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logger.error(f"Error converting {midi_path}: {e}")
            if abc_path:
                os.remove(abc_path)
            return None

