"""Week 8 test setup: the scripts aren't an installed package, so make them importable (and Week 7's, which the
controller uses for the drift check and the verdict).

Run:  uv run --group dev --group week07-drift pytest week08-auto-rollback/tests
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "week08-auto-rollback" / "scripts"))
sys.path.insert(0, str(ROOT / "week07-drift" / "scripts"))
