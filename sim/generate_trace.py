#!/usr/bin/env python3
import argparse
import json
import math
import random
from typing import List


def _zipf_probs(n: int, s: float) -> List[float]:
    weights = [1.0 / (i + 1) ** s for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def _sample_topk(probs: List[float], k: int) -> List[int]:
    # Sample without replacement biased by probs; approximate via repeated choice.
    chosen = set()
    while len(chosen) < k:
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                chosen.add(i)
                break
    return list(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic MoE routing traces")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--tokens-per-batch", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--zipf", type=float, default=1.05, help="Zipf skew; higher=more hot experts")
    ap.add_argument("--coact", type=float, default=0.25, help="Prob of reusing a previous expert in top-k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    base_probs = _zipf_probs(args.experts, args.zipf)

    with open(args.out, "w", encoding="utf-8") as f:
        for layer in range(args.layers):
            # Slightly perturb expert distribution per layer
            layer_probs = [p * random.uniform(0.9, 1.1) for p in base_probs]
            s = sum(layer_probs)
            layer_probs = [p / s for p in layer_probs]

            for batch_id in range(args.batches):
                origin_rows = []
                topk_experts = []
                for _ in range(args.tokens_per_batch):
                    origin_rows.append(random.randrange(args.rows))
                    # bias co-activation by reusing some experts
                    current = []
                    if random.random() < args.coact:
                        seed_expert = _sample_topk(layer_probs, 1)[0]
                        current.append(seed_expert)
                    while len(current) < args.k:
                        pick = _sample_topk(layer_probs, 1)[0]
                        if pick not in current:
                            current.append(pick)
                    topk_experts.append(current)

                record = {
                    "layer": layer,
                    "batch_id": batch_id,
                    "origin_rows": origin_rows,
                    "topk_experts": topk_experts,
                }
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
