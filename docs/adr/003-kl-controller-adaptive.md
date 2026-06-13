# ADR-003: Adaptive KL controller vs fixed coefficient

## Status
Accepted

## Context
The KL penalty coefficient $\beta$ trades reward maximization against staying
near the reference policy. Too small and the policy drifts (and may hack the
reward); too large and it never improves. The "right" $\beta$ depends on the
reward scale, the prompt distribution, and the training stage — none known a
priori. We must choose between a **fixed** $\beta$ and one that **adapts** toward
a target KL.

## Decision
Default to an **adaptive KL controller** that nudges $\beta$ toward a target KL
each step,

$$
\beta \leftarrow \operatorname{clip}\big(\beta + \alpha\,(\text{KL}_\text{measured} - \text{KL}_\text{target}),\ \beta_\min,\ \beta_\max\big),
$$

while keeping a **fixed** controller available behind the same interface
(`kl_adaptive: false`).

## Rationale
- **Robustness to reward scale.** The reward model's output scale varies between
  runs; a fixed $\beta$ that is well-tuned for one reward model is wrong for
  another. Targeting a KL *value* is scale-invariant.
- **Self-correcting.** When KL overshoots the target the penalty rises and pulls
  the policy back; when KL is below target the penalty relaxes and lets the
  policy explore. This keeps training in a stable band automatically.
- **Bounded.** Clipping $\beta$ to $[\beta_\min, \beta_\max]$ prevents the
  controller from collapsing to zero or exploding.
- **Cheap.** A scalar proportional update per step — negligible cost.

## Alternatives Considered
- **Fixed coefficient.** Simpler and fully reproducible; ideal as a baseline and
  for ablations, but requires per-run tuning and silently fails when the reward
  scale shifts. Retained as a configurable option, not the default.
- **PID / multiplicative controllers** (e.g. Ziegler et al.'s multiplicative
  form). More knobs for marginal benefit at our scale; the proportional update is
  sufficient and easier to reason about. The factory (`make_kl_controller`) makes
  adding one straightforward.

## Consequences
- **Positive:** stable KL band across reward models without per-run tuning.
- **Negative:** $\beta$ is now time-varying, so the effective objective changes
  over training; we log `train/kl_coeff` every step so this is observable, and
  the controller state is checkpointed for exact resume.
