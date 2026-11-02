"""Make src/ importable for the tests (scripts in src/ import each other by bare name)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
