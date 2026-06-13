# Distributed Training

The pipeline scales from a single CPU/GPU to a multi-node cluster behind one
abstraction, `rlhf.distributed.strategy.DistributedStrategy`.

## Strategies

| Strategy | When | Mechanism |
|---|---|---|
| `SINGLE` | one GPU / CPU (default) | no wrapping |
| `DDP` | data-parallel replicas fit in memory | `DistributedDataParallel` |
| `FSDP` | model too large for one device | `FullyShardedDataParallel` (parameter/optimizer sharding) |

`prepare_model(model)` moves the model to this process's device and wraps it for
the chosen strategy. Crucially, if DDP/FSDP is requested but **no process group
is initialized**, it logs a warning and degrades to single-device — so the same
code runs unchanged on a laptop and a cluster.

## Process group helpers

`rlhf.distributed.utils` wraps `torch.distributed` with single-process-safe
fallbacks: `is_distributed`, `get_rank`, `get_world_size`, `is_main_process`,
`barrier`, `all_reduce_mean`, and `setup_process_group` / `cleanup_process_group`
(driven by the standard `RANK`/`WORLD_SIZE`/`LOCAL_RANK` env vars set by
`torchrun`).

## Launching

```bash
torchrun --nproc_per_node=8 -m rlhf ppo --config config.yaml
```

Only rank 0 writes checkpoints and logs (guard with `is_main_process()`); metrics
that must reflect the global batch are reduced with `all_reduce_mean`.

!!! note "Testing"
    The FSDP integration test (`tests/integration/test_distributed_fsdp.py`) is
    marked `gpu` and skipped unless ≥2 GPUs are present. The single-process and
    fallback paths are covered on CPU in `tests/unit/test_distributed.py`.
