# PPO Algorithm

This page derives the Proximal Policy Optimization objective used by the
pipeline, explains every design choice in the implementation, and catalogs the
common failure modes with their observable symptoms.

## 1. From the policy gradient theorem to PPO

We treat language generation as a token-level MDP: the **state** $s_t$ is the
prompt plus the tokens generated so far, the **action** $a_t$ is the next token,
and the **policy** $\pi_\theta(a_t \mid s_t)$ is the language model's next-token
distribution. The objective is to maximize the expected return $J(\theta) =
\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$.

The **policy gradient theorem** gives

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}
\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t \right],
$$

where $\hat{A}_t$ is an estimate of the advantage of action $a_t$. Plain policy
gradient (REINFORCE) is high variance and allows arbitrarily large policy
updates from a single batch. The **surrogate objective** instead optimizes the
importance-weighted advantage,

$$
L^{\text{PG}}(\theta) = \mathbb{E}_t\!\left[ r_t(\theta)\, \hat{A}_t \right],
\qquad r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)},
$$

which can be evaluated off-policy from rollouts collected under
$\pi_{\theta_\text{old}}$. Maximizing $L^{\text{PG}}$ directly is unsafe: a large
$r_t$ moves the policy far from $\pi_{\theta_\text{old}}$, invalidating the
importance-sampling approximation.

## 2. The clipped surrogate

PPO bounds the update by **clipping the ratio** into a trust region
$[1-\varepsilon, 1+\varepsilon]$:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[
  \min\!\big( r_t(\theta)\,\hat{A}_t,\ \operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t \big)
\right].
$$

The $\min$ takes the **pessimistic** of the clipped and unclipped terms. When
$\hat{A}_t > 0$ the objective is capped at $(1+\varepsilon)\hat{A}_t$, so there is
no incentive to push $r_t$ above $1+\varepsilon$; when $\hat{A}_t < 0$ it is
floored at $(1-\varepsilon)\hat{A}_t$. **Why clipping is necessary:** it removes
the gradient signal once the policy has moved "far enough" in one direction for
a given batch, preventing destructive updates that a hard KL constraint would
otherwise need a second-order method to enforce.

The implementation (`rlhf.training.ppo.algorithm.compute_ppo_loss`) minimizes
$-L^{\text{CLIP}}$ as `policy_loss`, computed as
`max(-A·r, -A·clip(r))` — algebraically identical to negating the $\min$.

## 3. Value function and entropy

The full loss adds a clipped value-function term and an entropy bonus:

$$
L = \underbrace{-L^{\text{CLIP}}}_{\text{policy}} + c_1\, L^{\text{VF}} - c_2\, H .
$$

**Clipped value loss** keeps the value estimate inside a trust region around the
rollout-time value $V_\text{old}$:

$$
L^{\text{VF}} = \mathbb{E}_t\!\left[ \max\!\big( (V_\theta(s_t) - R_t)^2,\
  (\operatorname{clip}(V_\theta, V_\text{old}\!\pm\!\varepsilon_v) - R_t)^2 \big) \right].
$$

**Entropy** is computed over the full vocabulary,
$H = \mathbb{E}_t[-\sum_a \pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t)]$, and
maximized (subtracted from the loss) to discourage premature determinism.

## 4. RLHF reward shaping and the KL penalty

The reward model emits a single scalar $r_\text{RM}(x,y)$ for a full response.
The per-token reward fed to advantage estimation is

$$
r_t = r_\text{RM}(x, y)\cdot \mathbb{1}[t = T] \;-\; \beta\, \text{KL}_t,
\qquad \text{KL}_t = \log\pi_\theta(a_t\mid s_t) - \log\pi_\text{ref}(a_t\mid s_t).
$$

The scalar reward lands only on the final token (the only position where it is
well-defined for variable-length responses); the **per-token KL penalty** pulls
every token back toward the frozen reference $\pi_\text{ref}$.

**KL penalty vs. the clip — how they interact.** The clip bounds *each gradient
step*; the KL penalty bounds the *cumulative drift* across steps. Clipping alone
cannot stop slow, steady divergence over many updates (each step is individually
small but they compound); the KL penalty shapes the reward so divergence is
explicitly costly. The coefficient $\beta$ is adapted toward a target KL
(see [ADR-003](../adr/003-kl-controller-adaptive.md)).

## 5. Generalized Advantage Estimation

With the per-token rewards and value estimates, advantages use GAE:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
\qquad
\hat{A}_t = \sum_{l\ge 0} (\gamma\lambda)^l\, \delta_{t+l}.
$$

The implementation evaluates the closed form $\hat{A} = \delta W^\top$ with a
discount matrix $W_{ij} = (\gamma\lambda)^{j-i}$ for $j \ge i$. This is loop-free,
numerically stable for all $\gamma\lambda \in [0,1]$, and exact at
$\gamma\lambda = 0$ (where $W$ becomes the identity) — unlike the reversed-cumsum
/ division trick, which overflows for $\gamma\lambda < 1$ over long sequences and
divides by zero at $\gamma\lambda = 0$.

## 6. The training cycle

```mermaid
flowchart LR
    P[Sample prompts] --> G[Generate responses<br/>(policy, no_grad)]
    G --> RM[Score reward model]
    G --> REF[Score reference<br/>log-probs]
    G --> V[Score values + old log-probs]
    RM & REF & V --> B[(Rollout buffer)]
    B --> KL[Per-token KL penalty]
    KL --> SHAPE[RLHF reward shaping]
    SHAPE --> GAE[GAE advantages<br/>+ whitening]
    GAE --> U[PPO update<br/>clip + value + entropy]
    U --> KLC[Update KL controller]
    KLC --> P
```

## 7. Failure modes and their symptoms

| Failure mode | Observable symptoms | Mitigation |
|---|---|---|
| **Reward hacking** | reward keeps rising while sample quality falls; `reward/hacking_score` spikes; ensemble std grows | KL penalty; ensemble uncertainty; saturation ceiling + early stop |
| **KL collapse / blow-up** | `train/kl_divergence` rises monotonically past the target then explodes | adaptive $\beta$; `kl_abort_threshold`; lower LR |
| **Value divergence** | `train/explained_variance` goes negative and stays there; value loss grows | value clipping ($\varepsilon_v$); lower value LR; zero-init head |
| **Entropy collapse** | `train/entropy` $\to 0$; responses become repetitive/degenerate | raise $c_2$ (`entropy_coeff`); lower LR |
| **Gradient explosion** | `train/gradient_norm` $\gg$ historic norm; NaN losses | gradient clipping (`max_grad_norm`); bf16 instead of fp16 |

## 8. Hyperparameter sensitivity

Ordered roughly by impact on stability:

1. **`learning_rate`** — the single most sensitive knob. If KL spikes or loss is
   noisy, halve it first.
2. **`kl_target` / `kl_init`** — control how tightly the policy is leashed.
   Lower the target if quality degrades; raise it if the policy never moves.
3. **`clip_eps`** ($\varepsilon$) — smaller (0.1) is more conservative; larger
   (0.3) allows bigger steps. Reduce it when `clip_fraction` is high *and* KL is
   rising.
4. **`ppo_epochs`** — more epochs reuse each rollout harder but increase
   off-policy error; drop from 4 to 1–2 if `approx_kl` within the update is large.
5. **`gamma` / `lam`** — for the short, dense-terminal-reward RLHF setting,
   $\gamma = 1$, $\lambda \in [0.95, 1]$ is standard; rarely the bottleneck.
6. **`entropy_coeff`** ($c_2$) — raise only if entropy is collapsing; too high
   prevents convergence.

**Rule of thumb when training is unstable:** check `train/kl_divergence` and
`train/gradient_norm` first; if both are rising, lower the learning rate before
touching anything else.

## References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* <https://arxiv.org/abs/1707.06347>
- Schulman et al. (2016). *High-Dimensional Continuous Control Using GAE.* <https://arxiv.org/abs/1506.02438>
- Ziegler et al. (2019). *Fine-Tuning Language Models from Human Preferences.* <https://arxiv.org/abs/1909.08593>
- Stiennon et al. (2020). *Learning to summarize from human feedback.* <https://arxiv.org/abs/2009.01325>
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* <https://arxiv.org/abs/2203.02155>
