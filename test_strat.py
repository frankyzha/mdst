"""Compatibility entrypoint for partition-dp benchmark.

Use directly:
  python partition-dp/benchmark.py ...
Legacy-compatible:
  python test_strat.py ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().parent / "partition-dp" / "benchmark.py"
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
