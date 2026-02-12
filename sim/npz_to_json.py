#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict


def _normalize_value(value: Any) -> Any:
    try:
        if hasattr(value, "shape") and value.shape == ():
            value = value.item()
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
    except Exception:
        pass
    return value


def _as_list(value: Any) -> Any:
    if value is None:
        return None
    try:
        return value.tolist()
    except Exception:
        return value


def _summary_entry(value: Any, sample: int = 5) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"type": type(value).__name__}
    if value is None:
        entry["value"] = None
        return entry
    if hasattr(value, "shape"):
        entry["shape"] = list(value.shape)
        entry["dtype"] = str(getattr(value, "dtype", ""))
        if value.size == 0:
            entry["sample"] = []
        else:
            try:
                flat = value.reshape(-1)
                entry["sample"] = _as_list(flat[:sample])
                try:
                    entry["min"] = float(value.min())
                    entry["max"] = float(value.max())
                except Exception:
                    entry["min"] = "<unavailable>"
                    entry["max"] = "<unavailable>"
            except Exception:
                entry["sample"] = "<unavailable>"
        return entry
    if isinstance(value, (list, tuple)):
        entry["len"] = len(value)
        entry["sample"] = list(value[:sample])
        return entry
    entry["value"] = value
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a single NPZ file to JSON")
    ap.add_argument("--in", dest="in_path", required=True, help="Input .npz file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output .json file")
    ap.add_argument("--summary", action="store_true", help="Write a summary-only JSON")
    ap.add_argument("--sample", type=int, default=5, help="Sample size for summary (default: 5)")
    args = ap.parse_args()

    import numpy as np  # local import to keep optional dependency

    npz = np.load(args.in_path, allow_pickle=True)
    data: Dict[str, Any] = {}

    for key in npz.files:
        value = _normalize_value(npz[key])
        if args.summary:
            data[key] = _summary_entry(value, sample=args.sample)
        else:
            data[key] = _as_list(value)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
