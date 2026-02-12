#!/usr/bin/env python3
import argparse
import csv


def read_csv(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            data.append([float(x) for x in row])
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot co-activation heatmap from CSV")
    ap.add_argument("--csv", required=True, help="Path to co-activation CSV")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Co-activation Heatmap")
    ap.add_argument("--max", type=float, default=0.0, help="Clamp max value (0 = auto)")
    args = ap.parse_args()

    data = read_csv(args.csv)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    vmax = args.max if args.max > 0 else None
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap="viridis", interpolation="nearest", vmax=vmax)
    plt.title(args.title)
    plt.xlabel("Expert")
    plt.ylabel("Expert")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
