# Training Run

## Data formats

All datasets are **JSON-lines** (one JSON object per line).

**SFT** (`scripts/train_sft.py --data sft.jsonl`):

```json
{"prompt": "Question: ...\nAnswer:", "completion": " ..."}
```

**Preferences** (`scripts/train_reward_model.py --data prefs.jsonl`) — validated
against `rlhf.data.schemas.Preference`:

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "annotator_id": "a1"}
```

**Prompts** (`scripts/train_ppo.py --prompts prompts.jsonl`):

```json
{"prompt": "..."}
```

## What each stage produces

| Stage | Output |
|---|---|
| SFT | `outputs/sft/` — HF model + tokenizer |
| Reward model | `outputs/reward_model/reward_model.pt` + SHA-256 manifest |
| PPO | `outputs/ppo/checkpoint-step-*`, `outputs/ppo/final.pt`, TensorBoard logs |

## The PPO loop (per global step)

1. **Rollout** — sample prompts, generate, score with policy/reference/reward model.
2. **Advantages** — per-token KL penalty → RLHF reward shaping → GAE → whiten.
3. **Update** — `ppo_epochs` × mini-batches of clipped-surrogate optimization.
4. **Log & checkpoint** — every metric; checkpoint every `save_every` steps.
5. **Early-stopping checks** — reward hacking, KL blow-up, NaN losses.

## Resuming

```python
trainer.load_checkpoint("outputs/ppo/checkpoint-step-100")
trainer.train()   # continues from step 100, RNG/optimizer/KL state restored
```

Resuming reproduces a continuous run bit-for-bit on CPU.
