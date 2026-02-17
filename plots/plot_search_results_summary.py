#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RunMetrics:
    mean: float
    dispatch: float
    compute: float
    combine: float
    search_type: str
    run_id: str
    path: str


def _load_eval_md(path: str) -> Dict[str, float]:
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- mean:"):
                metrics["mean"] = float(line.split(":")[1].strip())
            elif line.startswith("- dispatch_mean:"):
                metrics["dispatch"] = float(line.split(":")[1].strip())
            elif line.startswith("- compute_mean:"):
                metrics["compute"] = float(line.split(":")[1].strip())
            elif line.startswith("- combine_mean:"):
                metrics["combine"] = float(line.split(":")[1].strip())
    return metrics


def _system_signature(run_cfg: Dict) -> Tuple[Tuple[str, str], ...]:
    keys = [
        "trace",
        "layer",
        "rows",
        "cols",
        "experts",
        "num_shared_experts",
        "slots",
        "routing_strategy",
        "capacity_factor",
        "max_records",
        "fast_test_pct",
        "max_seq_len",
    ]
    sig = []
    for k in keys:
        if k in run_cfg:
            sig.append((k, str(run_cfg[k])))
    return tuple(sig)


def _sig_hash(sig: Tuple[Tuple[str, str], ...]) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(sig, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:8]


def _discover_runs(results_dir: str) -> Tuple[Dict[Tuple[Tuple[str, str], ...], List[RunMetrics]], Dict[Tuple[Tuple[str, str], ...], Dict]]:
    groups: Dict[Tuple[Tuple[str, str], ...], List[RunMetrics]] = {}
    sys_cfgs: Dict[Tuple[Tuple[str, str], ...], Dict] = {}
    for root, dirs, files in os.walk(results_dir):
        if "run_config.json" not in files or "eval.md" not in files:
            continue
        run_cfg_path = os.path.join(root, "run_config.json")
        eval_path = os.path.join(root, "eval.md")
        with open(run_cfg_path, "r", encoding="utf-8") as f:
            run_cfg = json.load(f)
        metrics = _load_eval_md(eval_path)
        if not metrics:
            continue
        search_type = run_cfg.get("search", "unknown")
        run_id = os.path.basename(root)
        rm = RunMetrics(
            mean=metrics.get("mean", 0.0),
            dispatch=metrics.get("dispatch", 0.0),
            compute=metrics.get("compute", 0.0),
            combine=metrics.get("combine", 0.0),
            search_type=search_type,
            run_id=run_id,
            path=root,
        )
        sig = _system_signature(run_cfg)
        groups.setdefault(sig, []).append(rm)
        if sig not in sys_cfgs:
            sys_cfgs[sig] = run_cfg
    return groups, sys_cfgs


def _infer_dataset_layer(results_dir: str) -> Tuple[str, str]:
    parts = os.path.normpath(results_dir).split(os.sep)
    dataset = "dataset"
    layer = "layer"
    for i, p in enumerate(parts):
        if p.startswith("layer"):
            layer = p
            if i > 0:
                dataset = parts[i - 1]
            break
    return dataset, layer


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize search results and plot metrics by system config")
    ap.add_argument("--results-dir", required=True, help="results/DATASET/layerX directory")
    ap.add_argument("--out-dir", default="", help="Output directory for summary plots (default: results/plots)")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    dataset, layer = _infer_dataset_layer(args.results_dir)
    out_dir = args.out_dir or os.path.join("results", "plots", "search_summaries", dataset, layer)
    os.makedirs(out_dir, exist_ok=True)

    groups, sys_cfgs = _discover_runs(args.results_dir)
    if not groups:
        raise SystemExit("No runs found (expected run_config.json and eval.md)")

    for sig, runs in groups.items():
        runs = sorted(runs, key=lambda r: (r.search_type, r.run_id))
        x = list(range(len(runs)))
        fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.6), 6))
        dispatch_vals = [r.dispatch for r in runs]
        compute_vals = [r.compute for r in runs]
        combine_vals = [r.combine for r in runs]
        ax.bar(x, dispatch_vals, label="dispatch_mean")
        ax.bar(x, compute_vals, bottom=dispatch_vals, label="compute_mean")
        bottoms = [d + c for d, c in zip(dispatch_vals, compute_vals)]
        ax.bar(x, combine_vals, bottom=bottoms, label="combine_mean")
        ax.set_xlabel("Run")
        ax.set_ylabel("Cost (us)")
        ax.set_title("Search Summary by System Configuration")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        labels = [f"{r.search_type}\n{r.run_id}" for r in runs]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)

        sig_id = _sig_hash(sig)
        out_path = os.path.join(out_dir, f"search_summary_{sig_id}.png")
        meta_path = os.path.join(out_dir, f"search_summary_{sig_id}.json")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        sys_cfg = sys_cfgs.get(sig, {}).copy()
        for k in ["command", "search", "search_config", "search_config_content", "objective",
                  "origin_mode", "shared_row_replication"]:
            if k in sys_cfg:
                sys_cfg.pop(k)
        meta = {
            "system_signature": list(sig),
            "system_config": sys_cfg,
            "runs": [
                {
                    "search_type": r.search_type,
                    "run_id": r.run_id,
                    "path": r.path,
                    "mean": r.mean,
                    "dispatch_mean": r.dispatch,
                    "compute_mean": r.compute,
                    "combine_mean": r.combine,
                }
                for r in runs
            ],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
