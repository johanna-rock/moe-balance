# Search Configs

These JSONC files define the *search* procedure only. Placement is configured separately in `configs/initial_placement/`.

## Search Types
- `none`: no search (placement-only eval)
- `local`: local search only
- `anneal`: simulated annealing only
- `sequence`: custom list of `local` and `anneal` steps
- `random`: random placements (does not use initial placement)

## Examples
- `local_example.jsonc`
- `anneal_example.jsonc`
- `sequence_example.jsonc
- `hybrid_sequence_example.jsonc``
- `none_example.jsonc`
- `random_example.jsonc`
