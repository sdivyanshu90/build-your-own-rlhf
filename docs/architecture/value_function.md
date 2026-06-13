# Value Function

PPO uses a learned state-value function $V(s_t)$ as the baseline for advantage
estimation. We implement it as a **value head** on the shared policy backbone
rather than a separate network — see [ADR-002](../adr/002-separate-value-head.md).

## Architecture

`rlhf.models.value_head.ValueHead` is a 2-layer MLP:

```
hidden --> LayerNorm --> Linear(hidden, hidden/2) --> GELU --> Dropout
       --> LayerNorm --> Linear(hidden/2, 1)   [zero-initialized]
```

The output layer is **zero-initialized** so $V(s) = 0$ everywhere at step 0; the
advantage signal in the first updates is therefore not polluted by large random
value predictions while the head is still learning.

## Alignment with actions

For a full sequence, the model's logits at position $p-1$ predict token $p$.
`PolicyModel.score_sequence` returns log-probs and values **right-aligned** so
that index $p$ describes the action producing token $p$ and the value of the
state it was produced from. This keeps the policy ratio, value targets, and GAE
all consistent token-for-token.

## Value loss and clipping

The value target is the GAE return $R_t = \hat{A}_t + V_\text{old}(s_t)$. The
loss is the **pessimistic** clipped squared error,

$$
L^{\text{VF}} = \mathbb{E}_t\big[\max((V_\theta - R_t)^2, (\operatorname{clip}(V_\theta, V_\text{old}\pm\varepsilon_v) - R_t)^2)\big],
$$

so a single noisy target cannot move the value far in one step.

## Diagnosing value quality

`train/explained_variance` $= 1 - \operatorname{Var}(R - V)/\operatorname{Var}(R)$
measures how much return variance the value head explains: 1.0 is perfect, 0.0 is
a constant baseline, and **negative** indicates divergence (lower the value
learning rate or check `value_coeff`).
