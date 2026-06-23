"""
conftest.py — makes `src/` importable when pytest is run from the project root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))