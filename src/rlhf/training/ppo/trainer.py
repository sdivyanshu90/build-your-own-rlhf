"""
training.ppo.trainer — the PPO-for-RLHF training orchestrator.

Overview
--------
:class:`PPOTrainer` runs the canonical five-phase PPO loop per global step:

1. **Rollout collection** — sample prompts, generate responses, and score them
   with the policy (old log-probs + values), the reference (ref log-probs) and
   the reward model (scalar reward). Everything is pushed into a
   :class:`~rlhf.training.ppo.rollout.RolloutBuffer`.
2. **Advantage computation** — per-token KL penalty, RLHF reward shaping, GAE,
   and advantage whitening (all inside the buffer).
3. **PPO update** — ``ppo_epochs`` passes of clipped-surrogate optimization over
   mini-batches, with gradient clipping, accumulation, optimizer + LR steps, and
   a KL-controller update.
4. **Logging & checkpointing** — emit every metric; periodically evaluate and
   checkpoint (policy, value head, optimizer, scheduler, KL state, RNG state).
5. **Early-stopping checks** — abort (preserving the last checkpoint) on reward
   hacking, KL blow-up, or non-finite losses.

References
----------
- Schulman et al. (2017). https://arxiv.org/abs/1707.06347
- Ziegler et al. (2019). https://arxiv.org/abs/1909.08593
- Ouyang et al. (2022). https://arxiv.org/abs/2203.02155

Legend: B = batch, L = full seq length, T = response length, V = vocab size.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase

from rlhf.config.schema import KLConfig, PPOConfig
from rlhf.data.preprocessing import pad_sequences
from rlhf.exceptions import RewardHackingDetected, RolloutError, TrainingError
from rlhf.inference.generation import BatchGenerator
from rlhf.models.policy import PolicyModel
from rlhf.models.reference_model import ReferenceModel
from rlhf.monitoring.alerts import AlertManager
from rlhf.monitoring.logger import RLHFLogger
from rlhf.training.ppo.algorithm import compute_ppo_loss
from rlhf.training.ppo.kl_controller import make_kl_controller
from rlhf.training.ppo.rollout import RolloutBuffer, RolloutMiniBatch
from rlhf.training.ppo.scheduler import build_lr_scheduler
from rlhf.utils import set_seed

logger = logging.getLogger(__name__)

# A reward scorer maps (input_ids, attention_mask) -> (mean_reward, std_reward).
RewardScorer = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]

# Reward-scale floor in the reward-hacking dispersion ratio's denominator. A
# Bradley-Terry reward model emits O(1) rewards, so flooring the denominator at
# this scale keeps a healthy near-zero-mean batch well below the threshold while
# a genuinely high-dispersion (partially-exploited) batch still trips it.
_HACK_EPS: float = 5.0

# Absolute mean-reward magnitude beyond which the run is treated as reward
# hacking regardless of dispersion. A BT reward head is trained to O(1) outputs,
# so a mean reward this large means the policy found inputs that fool the reward
# model (or a broken reward signal) — e.g. a degenerate reward that returns +100.
_REWARD_SATURATION_CEILING: float = 20.0


class PPOTrainer:
    """Full PPO-for-RLHF trainer (see module docstring for the phase breakdown)."""

    def __init__(
        self,
        policy: PolicyModel,
        reference: ReferenceModel,
        tokenizer: PreTrainedTokenizerBase,
        config: PPOConfig,
        prompts: list[str],
        *,
        reward_model: nn.Module | None = None,
        reward_scorer: RewardScorer | None = None,
        device: torch.device | None = None,
        logger_backend: RLHFLogger | None = None,
        alert_manager: AlertManager | None = None,
        output_dir: str | Path = "outputs",
        reward_saturation_ceiling: float = _REWARD_SATURATION_CEILING,
    ) -> None:
        if reward_model is None and reward_scorer is None:
            raise RolloutError("Provide either reward_model or reward_scorer.")
        if not prompts:
            raise RolloutError("PPOTrainer requires a non-empty prompt list.")
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.config = config
        self.prompts = prompts
        self.reward_model = reward_model
        self._reward_scorer = reward_scorer
        self.device = device or torch.device("cpu")
        self.logger = logger_backend
        self.output_dir = Path(output_dir)
        self.reward_saturation_ceiling = reward_saturation_ceiling
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )

        self.policy.to(self.device)
        self.reference.to(self.device)
        if reward_model is not None:
            reward_model.to(self.device)

        self.generator = BatchGenerator(
            policy,
            tokenizer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
        )
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.learning_rate)
        self.scheduler = build_lr_scheduler(
            self.optimizer, config.lr_scheduler, config.warmup_steps, config.total_steps
        )
        self.kl_controller = make_kl_controller(KLConfig.from_ppo_config(config))
        self.alert_manager = alert_manager or AlertManager(
            kl_abort_threshold=config.kl_abort_threshold,
            reward_hacking_threshold=config.reward_hacking_threshold,
        )
        self.buffer = RolloutBuffer(capacity=config.rollout_batch_size, pad_token_id=self.pad_id)
        self.global_step = 0

    # ------------------------------------------------------------------ rewards
    def _score_rewards(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Dispatch to the reward model or the injected scorer; returns (mean, std)."""
        if self._reward_scorer is not None:
            return self._reward_scorer(input_ids, attention_mask)
        assert self.reward_model is not None  # guaranteed by __init__
        with torch.no_grad():
            out = self.reward_model(input_ids, attention_mask)
        if isinstance(out, tuple):  # RewardModelEnsemble -> (mean, std)
            return out
        return out, torch.zeros_like(out)

    # ----------------------------------------------------------- prompt sampling
    def _sample_prompts(self, n: int) -> list[str]:
        """Sample ``n`` prompt indices via the global RNG (so resume is exact)."""
        idx = torch.randint(0, len(self.prompts), (n,))
        return [self.prompts[int(i)] for i in idx]

    # ------------------------------------------------------ Phase 1: collection
    def _collect_rollouts(self) -> dict[str, float]:
        """Generate, score, and push a full rollout batch into the buffer."""
        self.buffer.clear()
        self.policy.eval()
        prompts = self._sample_prompts(self.config.rollout_batch_size)
        gen = self.generator.generate(prompts)
        samples = self.generator.to_samples(gen)

        rewards_all: list[float] = []
        stds_all: list[float] = []
        lengths: list[int] = []
        # Score in mini-batch-sized chunks to bound peak memory on large models.
        chunk = self.config.mini_batch_size
        for start in range(0, len(samples), chunk):
            block = [s for s in samples[start : start + chunk] if len(s.response_ids) > 0]
            if not block:
                continue
            self._score_and_push(block, rewards_all, stds_all, lengths)

        if len(self.buffer) == 0:
            raise RolloutError("rollout produced no non-empty responses this step.")
        reward_t = torch.tensor(rewards_all)
        return {
            "reward_mean": float(reward_t.mean()),
            "reward_std": float(reward_t.std(unbiased=False)),
            "reward_min": float(reward_t.min()),
            "reward_max": float(reward_t.max()),
            "ensemble_std_mean": float(torch.tensor(stds_all).mean()) if stds_all else 0.0,
            "response_length_mean": float(sum(lengths) / max(len(lengths), 1)),
        }

    def _score_and_push(
        self,
        block: list[Any],
        rewards_all: list[float],
        stds_all: list[float],
        lengths: list[int],
    ) -> None:
        """Score one chunk of samples and push their response slices to the buffer."""
        full_seqs = [s.prompt_ids + s.response_ids for s in block]
        full_ids, full_mask = pad_sequences(full_seqs, self.pad_id, padding_side="right")
        full_ids = full_ids.to(self.device)
        full_mask = full_mask.to(self.device)
        with torch.no_grad():
            old_logprobs, old_values, _ = self.policy.score_sequence(full_ids, full_mask)
            ref_logprobs = self.reference.compute_logprobs(full_ids, full_mask)
            reward, std = self._score_rewards(full_ids, full_mask)
        for i, sample in enumerate(block):
            p_len = len(sample.prompt_ids)
            r_len = len(sample.response_ids)
            sl = slice(p_len, p_len + r_len)
            self.buffer.push(
                prompt_ids=torch.tensor(sample.prompt_ids, dtype=torch.long),
                response_ids=torch.tensor(sample.response_ids, dtype=torch.long),
                logprobs=old_logprobs[i, sl],
                ref_logprobs=ref_logprobs[i, sl],
                values=old_values[i, sl],
                reward=float(reward[i]),
            )
            rewards_all.append(float(reward[i]))
            stds_all.append(float(std[i]))
            lengths.append(r_len)

    # --------------------------------------------------------- Phase 3: update
    def _ppo_micro_step(self, mini_batch: RolloutMiniBatch) -> dict[str, float]:
        """Forward + loss for one mini-batch (no optimizer step here)."""
        mb = mini_batch.to(self.device)
        logprobs, values, logits = self.policy.score_sequence(mb.input_ids, mb.attention_mask)
        loss_out = compute_ppo_loss(
            logprobs=logprobs,
            old_logprobs=mb.old_logprobs,
            advantages=mb.advantages,
            returns=mb.returns,
            values=values,
            old_values=mb.old_values,
            mask=mb.response_mask,
            clip_eps=self.config.clip_eps,
            clip_eps_vf=self.config.clip_eps_vf,
            entropy_coeff=self.config.entropy_coeff,
            value_coeff=self.config.value_coeff,
            vocab_logits=logits,
        )
        # Scale for gradient accumulation so the effective batch loss is a mean.
        scaled_loss = loss_out.total_loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
        return {
            "total_loss": float(loss_out.total_loss.detach()),
            "policy_loss": loss_out.policy_loss,
            "value_loss": loss_out.value_loss,
            "entropy": loss_out.entropy_loss,
            "approx_kl": loss_out.approx_kl,
            "clip_fraction": loss_out.clip_fraction,
            "explained_variance": loss_out.explained_variance,
        }

    def _ppo_update(self) -> dict[str, float]:
        """Run all PPO epochs over the buffer; returns averaged update metrics."""
        self.policy.train()
        accum: dict[str, float] = {}
        n_updates = 0
        grad_norm = 0.0
        micro_since_step = 0
        for _epoch in range(self.config.ppo_epochs):
            for mini_batch in self.buffer.get_mini_batches(
                self.config.mini_batch_size, shuffle=True, device=self.device
            ):
                metrics = self._ppo_micro_step(mini_batch)
                micro_since_step += 1
                if micro_since_step >= self.config.gradient_accumulation_steps:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.config.max_grad_norm
                        )
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    micro_since_step = 0
                for key, value in metrics.items():
                    accum[key] = accum.get(key, 0.0) + value
                n_updates += 1
        # Flush any remaining accumulated gradients (partial accumulation group).
        if micro_since_step > 0:
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
        averaged = {k: v / max(n_updates, 1) for k, v in accum.items()}
        averaged["gradient_norm"] = grad_norm
        averaged["learning_rate"] = float(self.scheduler.get_last_lr()[0])
        return averaged

    # ------------------------------------------------------------- global step
    def train_step(self) -> dict[str, float]:
        """Execute one full PPO global step and return its metrics."""
        # Phase 1 + 2.
        rollout_stats = self._collect_rollouts()
        adv_stats = self.buffer.compute_advantages(
            gamma=self.config.gamma,
            lam=self.config.lam,
            kl_coef=self.kl_controller.value,
            whiten_advantages=True,
        )
        # Phase 3.
        update_stats = self._ppo_update()
        # Update the KL controller from the measured rollout KL.
        self.kl_controller.update(adv_stats["kl_mean"])
        self.global_step += 1

        # Phase 4: assemble the full metric dict.
        hacking_score = rollout_stats["reward_std"] / (
            abs(rollout_stats["reward_mean"]) + _HACK_EPS
        )
        metrics = {
            "train/policy_loss": update_stats["policy_loss"],
            "train/value_loss": update_stats["value_loss"],
            "train/entropy": update_stats["entropy"],
            "train/total_loss": update_stats["total_loss"],
            "train/kl_divergence": adv_stats["kl_mean"],
            "train/approx_kl": update_stats["approx_kl"],
            "train/clip_fraction": update_stats["clip_fraction"],
            "train/explained_variance": update_stats["explained_variance"],
            "train/gradient_norm": update_stats["gradient_norm"],
            "train/learning_rate": update_stats["learning_rate"],
            "train/kl_coeff": self.kl_controller.value,
            "reward/mean": rollout_stats["reward_mean"],
            "reward/std": rollout_stats["reward_std"],
            "reward/min": rollout_stats["reward_min"],
            "reward/max": rollout_stats["reward_max"],
            "reward/hacking_score": hacking_score,
            "reward/ensemble_std": rollout_stats["ensemble_std_mean"],
            "response/length_mean": rollout_stats["response_length_mean"],
            "advantage/mean": adv_stats["advantage_mean"],
            "advantage/std": adv_stats["advantage_std"],
        }
        if self.logger is not None:
            self.logger.log_ppo_step(self.global_step, metrics)

        # Phase 5: early-stopping / anomaly checks. The AlertManager logs, fires
        # any webhook, and raises the mapped exception so the just-saved
        # checkpoint is the last known-good state.
        self._early_stopping_checks(metrics)
        return metrics

    def _early_stopping_checks(self, metrics: dict[str, float]) -> None:
        """Phase 5 guards: reward hacking, KL blow-up, non-finite losses."""
        # Reward hacking fires on EITHER signal, so both a high-dispersion
        # (partially exploited) batch and a saturated (uniformly absurd) reward
        # are caught. The spec'd exception is raised explicitly with a precise
        # message; the remaining conditions (KL spike, NaN loss, entropy
        # collapse, gradient explosion) are delegated to the AlertManager.
        dispersion = metrics["reward/hacking_score"]
        mean_reward = metrics["reward/mean"]
        if dispersion > self.config.reward_hacking_threshold:
            message = (
                f"reward-hacking score {dispersion:.3f} exceeds threshold "
                f"{self.config.reward_hacking_threshold} at step {self.global_step}."
            )
            logger.error(message)
            raise RewardHackingDetected(message)
        if abs(mean_reward) > self.reward_saturation_ceiling:
            message = (
                f"mean reward {mean_reward:.3f} exceeds the saturation ceiling "
                f"{self.reward_saturation_ceiling} at step {self.global_step}; the policy "
                "is exploiting the reward model."
            )
            logger.error(message)
            raise RewardHackingDetected(message)
        self.alert_manager.check(metrics)

    def train(self, num_steps: int | None = None) -> list[dict[str, float]]:
        """
        Run the PPO loop.

        Args:
            num_steps: Steps to run (defaults to ``config.total_steps``).

        Returns:
            Per-step metric dicts.
        """
        set_seed(self.config.seed)
        steps = num_steps if num_steps is not None else self.config.total_steps
        history: list[dict[str, float]] = []
        for _ in range(steps):
            metrics = self.train_step()
            history.append(metrics)
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(self.output_dir / f"checkpoint-step-{self.global_step}")
        return history

    # -------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict[str, Any]:
        """Serialize everything needed to resume training bit-for-bit."""
        return {
            "policy_backbone": self.policy.backbone.state_dict(),
            "value_head": self.policy.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "kl_controller": self.kl_controller.state_dict(),
            "global_step": self.global_step,
            "rng_state": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`."""
        self.policy.backbone.load_state_dict(state["policy_backbone"])
        self.policy.value_head.load_state_dict(state["value_head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.kl_controller.load_state_dict(state["kl_controller"])
        self.global_step = int(state["global_step"])
        torch.set_rng_state(state["rng_state"].to(torch.uint8).cpu())

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write a resumable checkpoint to ``path`` (creating parents)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "trainer_state.pt"
        torch.save(self.state_dict(), target)
        logger.info("checkpoint saved: %s (step %d)", target, self.global_step)
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint written by :meth:`save_checkpoint`."""
        target = Path(path) / "trainer_state.pt"
        # nosec B614: these checkpoints are produced by this trainer; the state
        # mixes tensors with optimizer/scheduler/RNG objects so weights_only=False
        # is required. For untrusted sources, load via security.CheckpointVerifier.
        state = torch.load(target, map_location=self.device, weights_only=False)  # nosec B614
        self.load_state_dict(state)

    def _raise_if_nonfinite(self, value: float, name: str) -> None:
        """Raise :class:`TrainingError` if ``value`` is NaN/Inf (defensive guard)."""
        if not torch.isfinite(torch.tensor(value)):  # pragma: no cover - safety net
            raise TrainingError(f"{name} is not finite ({value}).")


__all__ = ["PPOTrainer", "RewardScorer"]
