"""Delete all local RepoPilot data (uploaded repos, ChromaDB index, reports).

    python scripts/reset_data.py --yes
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="Confirm deletion")
    args = ap.parse_args()
    target = get_settings().data_dir
    if not args.yes:
        print(f"This would delete: {target}\nRe-run with --yes to confirm.")
        return 1
    shutil.rmtree(target, ignore_errors=True)
    print(f"Deleted {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
