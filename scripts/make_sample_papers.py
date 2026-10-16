"""Generate the synthetic demo papers without launching the app.

Usage:  python scripts/make_sample_papers.py [target_dir]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scibot.samples import generate_sample_papers  # noqa: E402

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("papers")
    made = generate_sample_papers(target)
    for p in made:
        print(f"created {p}")
    if not made:
        print("sample papers already present")
