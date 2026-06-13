# Reference Model

The reference model $\pi_\text{ref}$ is a **frozen** copy of the SFT policy. It
anchors PPO: the per-token KL penalty
$\text{KL}_t = \log\pi_\theta(a_t\mid s_t) - \log\pi_\text{ref}(a_t\mid s_t)$
pulls the policy back toward this fixed point, preventing it from drifting into
degenerate, reward-hacking regions of output space.

## Construction

`rlhf.models.reference_model.ReferenceModel` wraps a causal LM that is always in
eval mode with `requires_grad=False`; no gradients ever flow through it.
`ReferenceModel.from_policy(policy)` snapshots the policy's backbone via a deep
copy, decoupling the reference from subsequent in-place policy updates.

!!! note "Memory"
    The deep copy is the safe, correctness-first default. In production one may
    instead share the frozen weights copy-on-write to halve memory, since the
    reference is never mutated.

## Log-prob alignment

`compute_logprobs(input_ids, attention_mask)` returns `(B, L)` reference
log-probs **right-aligned** exactly the way `PolicyModel.score_sequence` aligns
the policy's, so the per-token KL lines up token-for-token (position 0 is 0 — no
token precedes it).

## Why a fixed reference (not a moving average)?

A fixed reference gives a stable target for the KL constraint. A slowly-moving
reference (EMA of the policy) would let the constraint "follow" the policy and
permit unbounded cumulative drift — defeating the purpose of the penalty.
