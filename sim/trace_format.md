# Routing trace format (JSONL)

Each line is a JSON object representing one batch for a single layer.

Required fields:
- `layer`: integer layer index
- `batch_id`: integer batch id
- `origin_rows`: list[int] length = num_tokens (row id for each token)
- `topk_experts`: list[list[int]] shape = [num_tokens][k]

Optional fields:
- `topk_weights`: list[list[float]] shape = [num_tokens][k]
- `request_id`: request id from dataset (often same as batch_id)
- `source_file`: source filename (helpful for debugging)

Notes:
- `origin_rows` can be generated when true device assignment is unknown (e.g., round-robin or random across rows).

Example:
```json
{"layer": 0, "batch_id": 12, "origin_rows": [0,0,1,1], "topk_experts": [[3,7,9,11,13,17,19,23],[7,3,9,11,13,17,19,23],[2,5,6,10,12,14,18,22],[2,6,5,10,12,14,18,22]]}
```

Example (with request_id + source_file):
```json
{"layer": 0, "batch_id": 12, "request_id": 12, "source_file": "problem_id_12.npz", "origin_rows": [0,1], "topk_experts": [[3,7,9,11,13,17,19,23],[2,5,6,10,12,14,18,22]]}
```
