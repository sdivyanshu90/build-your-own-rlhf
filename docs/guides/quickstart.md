# Quickstart

## Install

```bash
git clone https://github.com/rlhf-ppo/rlhf-ppo && cd rlhf-ppo
make dev            # editable install + dev/monitoring extras
```

## Three-stage toy run

Create a minimal `config.yaml`:

```yaml
run_name: quickstart
device: cpu
output_dir: outputs
model:
  model_name_or_path: gpt2
reward_model:
  model_name_or_path: gpt2
  epochs: 1
sft:
  model_name_or_path: gpt2
  epochs: 1
ppo:
  total_steps: 20
  rollout_batch_size: 8
  mini_batch_size: 4
  max_new_tokens: 32
```

Then run each stage (data files are JSON-lines — see
[Training Run](training_run.md) for the schema):

```bash
python -m rlhf sft    --config config.yaml   # stage 0: SFT
python -m rlhf reward --config config.yaml   # stage 1: reward model
python -m rlhf ppo    --config config.yaml   # stage 2: PPO
```

## Verify your install

```bash
make test-unit       # fast, no model downloads
```
