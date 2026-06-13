# ADR-002: Separate value head vs separate value model

## Status
Accepted

## Context
PPO needs a state-value estimate $V(s_t)$ for every token to form GAE
advantages. Two architectures are common: (a) a **separate value network** (a
second transformer), or (b) a **value head** — a small MLP on top of the policy's
shared hidden states. The policy is ≥1B parameters; memory and rollout latency
are first-order constraints.

## Decision
Attach a **2-layer MLP value head** (`hidden → hidden/2 → 1`, LayerNorm before
each linear, **zero-initialized** output) to the policy backbone, so a single
forward pass yields both logits and per-position values.

## Rationale
- **Memory.** A separate value model would roughly double parameter and optimizer
  memory. The value head adds <0.1% of parameters.
- **One forward pass.** Sharing the backbone means rollout scoring and the PPO
  update each need a single forward, not two — a large latency saving when the
  backbone dominates compute.
- **Representation sharing.** The features that predict the next token are also
  informative for value; sharing acts as a regularizer.
- **Zero-init stability.** Initializing the output layer to zero makes
  $V(s) \approx 0$ at step 0, so early advantages are not corrupted by large
  random value predictions while the head is still untrained.

## Alternatives Considered
- **Separate value model.** Decouples value and policy capacity and avoids
  gradient interference between the two losses, but at ~2× memory and a second
  forward pass per step. Justified only when value/policy objectives demonstrably
  conflict — not observed at our scale.
- **No value model (Monte-Carlo / GRPO-style baselines).** Removes the head
  entirely but increases advantage variance and forgoes per-token credit
  assignment (see [ADR-001](001-ppo-over-reinforce.md)).

## Consequences
- **Positive:** minimal memory overhead, single-pass scoring, stable early
  training from the zero-init head.
- **Negative:** policy and value gradients flow through shared weights, so a
  mis-scaled `value_coeff` can perturb the policy; mitigated by value clipping
  ($\varepsilon_v$) and keeping `value_coeff` ≈ 0.5.
