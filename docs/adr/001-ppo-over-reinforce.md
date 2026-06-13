# ADR-001: PPO over REINFORCE / A2C / GRPO

## Status
Accepted

## Context
The policy-optimization stage must improve a ≥1B-parameter language model from a
scalar reward with **stable, sample-efficient** updates. Rollouts are expensive
(each requires autoregressive generation plus three scoring passes), so we must
reuse each batch across several gradient steps without the policy diverging. The
candidate algorithms are REINFORCE (with baseline), A2C, GRPO, and PPO.

## Decision
Use **PPO** with a clipped surrogate objective, GAE advantages, a learned value
head, and an adaptive KL penalty to a frozen reference.

## Rationale
- **Multi-epoch reuse.** PPO's clipped ratio makes it safe to take several
  gradient steps per rollout (`ppo_epochs`), amortizing expensive generation.
  REINFORCE/A2C are strictly on-policy and must discard each batch after one step.
- **Bounded updates without second-order cost.** The clip enforces a trust
  region with first-order gradients only — no Fisher-vector products (TRPO).
- **Empirical track record.** PPO is the algorithm behind InstructGPT
  (Ouyang et al., 2022) and the summarization work (Stiennon et al., 2020); its
  behaviour on RLHF is well characterized.
- **Variance reduction.** The value baseline + GAE substantially reduce gradient
  variance versus REINFORCE's Monte-Carlo returns.

## Alternatives Considered
- **REINFORCE / RLOO.** Simpler and value-free, but on-policy-only and higher
  variance. RLOO is attractive for short responses but discards the per-token
  credit assignment GAE provides; we keep it in mind as a future ablation.
- **A2C.** Single-step on-policy actor-critic; no trust region, so it is prone to
  the destructive updates the clip is designed to prevent.
- **GRPO.** Removes the value model and normalizes rewards within a group of
  samples per prompt. Lower memory (no value head) and increasingly popular, but
  it needs multiple samples per prompt and forgoes a learned baseline; the
  group-relative advantage is noisier on long, sparse-reward responses. A strong
  candidate we deliberately defer rather than adopt as the default.

## Consequences
- **Positive:** stable training, multi-epoch sample reuse, well-understood
  diagnostics (clip fraction, approx KL, explained variance).
- **Negative:** a value head doubles the parameters that receive gradients and
  adds a tuning surface (`value_coeff`, `clip_eps_vf`); PPO has more
  hyperparameters than REINFORCE/GRPO.
