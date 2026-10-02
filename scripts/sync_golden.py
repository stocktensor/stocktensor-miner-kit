"""Copy the golden vectors from a stocktensor-subnet checkout (default ../stocktensor-subnet).

    uv run python scripts/sync_golden.py [path/to/stocktensor-subnet]

The vectors are vendored so the kit's tests can check that the installed
``stocktensor`` package still produces exactly these scores.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    subnet = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "stocktensor-subnet"
    source = subnet / "tests" / "golden"
    if not source.is_dir():
        raise SystemExit(f"no golden vectors at {source}")
    target = ROOT / "tests" / "golden"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.json")):
        shutil.copyfile(path, target / path.name)
        print(f"copied {path.name}")


if __name__ == "__main__":
    main()
