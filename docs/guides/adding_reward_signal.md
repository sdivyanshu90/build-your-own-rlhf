# Adding a Reward Signal

PPO does not require the `RewardModel` class — it accepts any **reward scorer**,
a callable `(input_ids, attention_mask) -> (mean_reward, std_reward)`. This makes
it easy to add rule-based rewards, external graders, or hybrid signals.

## Inject a custom scorer

```python
import torch
from rlhf.training.ppo.trainer import PPOTrainer

def length_reward(input_ids, attention_mask):
    # Reward longer (non-padding) responses; zero uncertainty.
    n = attention_mask.sum(dim=1).float()
    return n / n.max().clamp(min=1.0), torch.zeros_like(n)

trainer = PPOTrainer(policy, reference, tokenizer, config, prompts,
                     reward_scorer=length_reward)
```

The scorer is called under `torch.no_grad()` during rollout collection. Return
`std_reward = 0` if you have no uncertainty estimate.

## Combine a learned model with a rule

```python
def hybrid(input_ids, attention_mask):
    mean, std = reward_model(input_ids, attention_mask)   # ensemble -> (mean, std)
    bonus = format_bonus(input_ids)                       # your rule
    return mean + bonus, std
```

## Guidance

- **Keep rewards O(1).** A Bradley-Terry head emits roughly unit-scale rewards;
  the reward-hacking **saturation ceiling** treats a mean reward beyond ±20 as an
  exploit and aborts. Scale custom rewards accordingly (or pass a higher
  `reward_saturation_ceiling`).
- **Provide a real `std`** when you can — it powers the `reward/ensemble_std`
  hacking canary.
- **Normalize** if your reward scale drifts over training; the built-in
  `RunningMoments` (used by `RewardModel`) is a ready template.
