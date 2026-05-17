from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PICKLE_PATH = PROJECT_ROOT / "data" / "raw" / "HW-NAS-Bench-v1_0.pickle"


def preview(value: Any, depth: int = 0, max_depth: int = 3) -> None:
    indent = "  " * depth
    type_name = type(value).__name__

    if depth >= max_depth:
        print(f"{indent}{type_name}: ...")
        return

    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{indent}dict with {len(keys)} keys")
        print(f"{indent}keys sample: {keys[:10]}")
        for key in keys[:3]:
            print(f"{indent}- key: {key!r}")
            preview(value[key], depth + 1, max_depth)
        return

    if isinstance(value, (list, tuple)):
        print(f"{indent}{type_name} with {len(value)} items")
        for index, item in enumerate(value[:3]):
            print(f"{indent}- index: {index}")
            preview(item, depth + 1, max_depth)
        return

    print(f"{indent}{type_name}: {repr(value)[:300]}")


def main() -> None:
    if not PICKLE_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {PICKLE_PATH}")

    print(f"Loading: {PICKLE_PATH}")
    print(f"Size: {PICKLE_PATH.stat().st_size / 1024 / 1024:.2f} MB")

    with PICKLE_PATH.open("rb") as file:
        data = pickle.load(file)

    print("\nDataset structure:")
    preview(data)


if __name__ == "__main__":
    main()
