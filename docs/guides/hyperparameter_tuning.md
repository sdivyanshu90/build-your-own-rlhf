# Hyperparameter Tuning

See the [PPO Algorithm](../architecture/ppo_algorithm.md#8-hyperparameter-sensitivity)
page for the full sensitivity analysis. This guide is the operational cheat-sheet.

## First, read these three metrics

| Metric | Healthy | Action if unhealthy |
|---|---|---|
| `train/kl_divergence` | rises slowly toward `kl_target` | rising fast → lower `learning_rate`, lower `kl_target` |
| `train/gradient_norm` | stable, < `max_grad_norm`-ish | spiking → lower LR; check for NaN |
| `train/explained_variance` | climbs toward 1 | negative → lower value LR / check `value_coeff` |

## Symptom → fix

- **KL explodes:** halve `learning_rate`; lower `kl_target`; ensure the adaptive
  controller is on (`kl_adaptive: true`).
- **Reward rises but quality drops (hacking):** lower `kl_target`; enable a
  reward ensemble (`ensemble_size > 1`) and watch `reward/ensemble_std`; lower
  `reward_hacking_threshold`.
- **Policy never improves:** raise `kl_target` (leash too tight); raise
  `learning_rate`; check the reward model actually separates good/bad
  (`accuracy` from RM eval).
- **Entropy collapses (repetitive output):** raise `entropy_coeff`; lower LR.
- **`clip_fraction` high (>0.3) and KL rising:** lower `clip_eps`; reduce
  `ppo_epochs`.

## Safe starting point (≥1B model)

```yaml
learning_rate: 1.4e-5
clip_eps: 0.2
kl_target: 6.0
kl_adaptive: true
ppo_epochs: 4
gamma: 1.0
lam: 0.95
entropy_coeff: 0.01
value_coeff: 0.5
max_grad_norm: 1.0
```
