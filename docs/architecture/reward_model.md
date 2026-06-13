# Reward Model

The reward model turns human preference comparisons into a scalar reward signal
for PPO.

## Architecture

```
backbone (transformer, hidden states) --> last-token pool --> LayerNorm --> Linear(hidden, 1)
```

The reward is read off the representation at the **last non-padding token** (the
EOS position) — the only position that has attended to the full variable-length
response (Stiennon et al., 2020). See `rlhf.models.reward_model.RewardModel`.

## Training objective

Preferences are pairs $(x, y_w, y_l)$ where $y_w$ is preferred. The model is
trained with the Bradley-Terry loss

$$
L_{\text{BT}} = -\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma(r_\theta(x,y_w) - r_\theta(x,y_l))\big],
$$

implemented with the numerically stable `F.logsigmoid`. The loss equals
$\log 2$ when the two rewards are equal — a useful sanity check at init.

## Reward normalization

PPO is sensitive to reward scale. The model maintains **online (Welford) running
statistics** of the rewards it produces and can standardize them
($r' = (r-\mu)/(\sigma+\varepsilon)$) before they reach the policy. Normalization
is off during reward-model training (BT compares raw scores) and enabled at PPO
time via `normalize_rewards`.

## Ensembles and uncertainty

`RewardModelEnsemble` aggregates $N$ independently-seeded models and returns
`(mean, std)`. The standard deviation is a calibrated reward-hacking canary — see
[ADR-004](../adr/004-reward-model-ensemble.md). Train members with
`EnsembleRewardModelTrainer`, which seeds each member distinctly so they do not
collapse to the same solution.
