# ADR-004: Reward-model ensemble for uncertainty vs single model

## Status
Accepted

## Context
The reward model is an imperfect proxy for human preference. PPO will
**over-optimize** any systematic error it contains — "reward hacking" — producing
outputs that score highly but are actually poor. We need a signal that warns when
the policy is exploiting reward-model error, and a single reward model provides
no such signal (it is, by construction, confident about its own mistakes).

## Decision
Support a **reward-model ensemble** of $N$ independently-initialized (and
independently-trained) models. At PPO time the ensemble returns
$(\text{mean}, \text{std})$; the **standard deviation across members** is logged
as `reward/ensemble_std` and used as a reward-hacking canary. $N=1$ (a single
model) remains the default for cost reasons.

## Rationale
- **Disagreement localizes hacking.** On in-distribution responses the members
  agree (low std). When the policy discovers out-of-distribution inputs that
  exploit one model's quirk, the members disagree (high std) — precisely the
  states we want to flag or penalize. This is empirically detectable even with
  small ensembles (verified in the poisoning adversarial test).
- **Cheap inference-time aggregation.** Scoring is embarrassingly parallel across
  members; the mean is a better point estimate and the std is free.
- **Composes with the KL penalty.** The KL penalty bounds drift from the
  reference; ensemble std bounds trust in the reward. They catch different
  failure modes.

## Alternatives Considered
- **Single reward model.** Cheapest, and the default, but blind to its own
  over-optimization. Acceptable for low-stakes runs with a strong KL leash.
- **Bayesian / MC-dropout uncertainty.** One model, dropout at inference for an
  uncertainty estimate. Cheaper than an ensemble but the uncertainty is poorly
  calibrated for the OOD inputs reward hacking produces — exactly where we need it.
- **Reward-model fine-tuning during PPO.** Continuously retrain the reward model
  on fresh human labels. The most robust long-term answer, but it requires a live
  human-labeling loop outside this system's scope.

## Consequences
- **Positive:** a calibrated reward-hacking signal; better point estimates;
  configurable cost via `ensemble_size`.
- **Negative:** $N\times$ reward-model memory and scoring cost when $N>1$; the
  ensemble must be trained with distinct seeds (handled by
  `EnsembleRewardModelTrainer`) or its members collapse to the same solution and
  the std signal vanishes.
