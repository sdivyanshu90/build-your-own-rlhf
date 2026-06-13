"""
models.policy — causal LM policy with an attached value head.

Overview
--------
:class:`PolicyModel` wraps a HuggingFace causal language model and attaches a
:class:`~rlhf.models.value_head.ValueHead`, so a single forward pass returns both
token logits and per-position value estimates — avoiding a second backbone pass
during the PPO update. It also exposes:

* :meth:`generate` — a thin wrapper over ``model.generate`` that additionally
  captures the per-token sampling log-probabilities of the generated tokens.
* :meth:`score_sequence` — teacher-forced scoring of a full (prompt+response)
  sequence, returning per-token action log-probs and state values aligned to the
  response tokens (the quantities PPO consumes).

Mathematical Background
-----------------------
For input tokens ``x_0..x_{L-1}`` the model emits logits where position ``p``
predicts token ``p+1``. The log-prob of the realized token at position ``p`` is
therefore ``log softmax(logits[:, p-1])[x_p]``; :meth:`score_sequence` returns
these right-aligned so that index ``p`` describes the action that produced token
``p`` and the value of the state it was produced from.

Usage Example
-------------
>>> from transformers import GPT2Config
>>> from rlhf.models.policy import PolicyModel
>>> policy = PolicyModel.from_config(GPT2Config(n_layer=2, n_embd=64, n_head=2))
>>> import torch
>>> logits, values = policy(torch.randint(0, 50, (2, 5)))
>>> logits.shape, values.shape
(torch.Size([2, 5, 50257]), torch.Size([2, 5]))

References
----------
- Ouyang et al. (2022). https://arxiv.org/abs/2203.02155

Legend: B = batch, T/L = sequence length, V = vocabulary size, H = hidden size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel

from rlhf.exceptions import PolicyModelError
from rlhf.models.base import logprobs_from_logits, resolve_dtype
from rlhf.models.value_head import ValueHead


@dataclass
class GenerationOutput:
    """
    Result of :meth:`PolicyModel.generate`.

    Attributes:
        sequences: ``(B, L)`` full prompt+response token ids.
        response_ids: ``(B, T)`` generated tokens only.
        logprobs: ``(B, T)`` sampling log-prob of each generated token.
        response_mask: ``(B, T)`` bool; True for real (non-padding) generated tokens.
        prompt_length: Number of prompt tokens (shared across the batch).
    """

    sequences: Tensor
    response_ids: Tensor
    logprobs: Tensor
    response_mask: Tensor
    prompt_length: int


class PolicyModel(nn.Module):
    """
    A causal-LM policy with a value head producing logits and values in one pass.

    Args:
        model_name_or_path: HF model id/path (mutually exclusive with ``backbone``).
        backbone: A pre-built causal-LM module (used by :meth:`from_config`).
        value_head_dropout: Dropout inside the value head.
        freeze_layers: Number of bottom transformer blocks to freeze.
        dtype: Optional weight dtype (string or ``torch.dtype``).
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        backbone: PreTrainedModel | None = None,
        value_head_dropout: float = 0.1,
        freeze_layers: int = 0,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if backbone is None and model_name_or_path is None:
            raise PolicyModelError("Provide either model_name_or_path or backbone.")
        if backbone is not None:
            self.backbone = backbone
        else:
            # Local import keeps transformers' heavy import lazy and avoids a
            # circular dependency with rlhf.models.base at module load time.
            from rlhf.models.base import load_causal_lm

            self.backbone = load_causal_lm(model_name_or_path, dtype=resolve_dtype(dtype))
        hidden_size = int(self.backbone.config.hidden_size)
        self.value_head = ValueHead(hidden_size, dropout=value_head_dropout)
        if dtype is not None:
            self.value_head = self.value_head.to(resolve_dtype(dtype))
        if freeze_layers > 0:
            self.freeze_bottom_layers(freeze_layers)

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        value_head_dropout: float = 0.1,
        freeze_layers: int = 0,
    ) -> PolicyModel:
        """Build a policy with random weights from a config (no download)."""
        from rlhf.models.base import load_causal_lm

        backbone = load_causal_lm(None, config=config)
        return cls(
            backbone=backbone,
            value_head_dropout=value_head_dropout,
            freeze_layers=freeze_layers,
        )

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.parameters()).device

    def _decoder_layers(self) -> nn.ModuleList:
        """Locate the list of transformer blocks across common architectures."""
        # GPT-2 style: backbone.transformer.h ; LLaMA/Mistral style: model.layers.
        transformer = getattr(self.backbone, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return transformer.h  # type: ignore[no-any-return]
        inner = getattr(self.backbone, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers  # type: ignore[no-any-return]
        raise PolicyModelError("Could not locate decoder layers to freeze.")

    def freeze_bottom_layers(self, n: int) -> None:
        """Freeze the input embeddings and the first ``n`` transformer blocks."""
        for param in self.backbone.get_input_embeddings().parameters():
            param.requires_grad_(False)
        layers = self._decoder_layers()
        for layer in list(layers)[:n]:
            for param in layer.parameters():
                param.requires_grad_(False)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Single forward pass returning logits and per-position values.

        Args:
            input_ids: ``(B, T)`` token ids.
            attention_mask: ``(B, T)`` mask; defaults to all-ones.

        Returns:
            ``(logits, values)`` of shapes ``(B, T, V)`` and ``(B, T)``.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]
        values = self.value_head(last_hidden)
        return outputs.logits, values

    def score_sequence(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Teacher-force a full sequence and return action log-probs, values, logits.

        Args:
            input_ids: ``(B, L)`` full prompt+response token ids.
            attention_mask: ``(B, L)`` mask.

        Returns:
            ``(logprobs, values, logits)`` where ``logprobs`` ``(B, L)`` and
            ``values`` ``(B, L)`` are right-aligned: index ``p`` describes the
            action producing token ``p`` and the value of its originating state.
            ``logits`` ``(B, L, V)`` are the raw logits (for the entropy term).
        """
        logits, per_pos_values = self.forward(input_ids, attention_mask)
        b, length = input_ids.shape
        logprobs = torch.zeros(b, length, dtype=logits.dtype, device=logits.device)
        # logits[:, p-1] predicts token p; gather the realized token's log-prob.
        logprobs[:, 1:] = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        # V(s) for the action at position p is the value read off the hidden state
        # at position p-1 (the state from which token p is produced).
        values = torch.zeros_like(per_pos_values)
        values[:, 1:] = per_pos_values[:, :-1]
        return logprobs, values, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> GenerationOutput:
        """
        Generate responses and capture per-token sampling log-probabilities.

        Args mirror the usual generation knobs. ``top_k=0`` disables top-k.

        Returns:
            A :class:`GenerationOutput`. The captured ``logprobs`` are the
            log-probs under the (temperature/top-p/top-k warped) sampling
            distribution; the PPO trainer recomputes old log-probs with
            :meth:`score_sequence` so the ratio is exact.
        """
        prompt_length = int(input_ids.shape[1])
        config = self.backbone.config
        pad_id = pad_token_id if pad_token_id is not None else config.pad_token_id
        eos_id = eos_token_id if eos_token_id is not None else config.eos_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        gen = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = gen.sequences
        response_ids = sequences[:, prompt_length:]

        # gen.scores is a tuple (one (B, V) tensor per generated step) of the
        # post-warper logits actually used for sampling; gather the chosen token.
        if gen.scores:
            stacked = torch.stack(gen.scores, dim=1)  # (B, T, V)
            logprobs = logprobs_from_logits(stacked, response_ids)
        else:  # pragma: no cover - only when max_new_tokens produces nothing
            logprobs = torch.zeros_like(response_ids, dtype=torch.float32)

        # Mark real generated tokens: everything up to and including the first EOS
        # is real; padding the generator appends afterward is masked out.
        response_mask = self._build_response_mask(response_ids, pad_id, eos_id)
        logprobs = logprobs * response_mask.to(logprobs.dtype)
        return GenerationOutput(
            sequences=sequences,
            response_ids=response_ids,
            logprobs=logprobs,
            response_mask=response_mask,
            prompt_length=prompt_length,
        )

    @staticmethod
    def _build_response_mask(response_ids: Tensor, pad_id: int, eos_id: int | None) -> Tensor:
        """True for tokens up to and including the first EOS in each row."""
        b, t = response_ids.shape
        mask = torch.ones(b, t, dtype=torch.bool, device=response_ids.device)
        if eos_id is None:
            return response_ids != pad_id
        is_eos = response_ids == eos_id
        for row in range(b):
            eos_positions = torch.nonzero(is_eos[row], as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                mask[row, first_eos + 1 :] = False
        return mask

    def save_pretrained(self, path: str | Path) -> None:
        """Persist the backbone and value head under ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.value_head.state_dict(), path / "value_head.pt")

    def load_value_head(self, path: str | Path) -> None:
        """Load value-head weights saved by :meth:`save_pretrained`."""
        # weights_only=True: the value head is a pure tensor state_dict, so we can
        # use the safe loader (no arbitrary pickle deserialization).
        state = torch.load(
            Path(path) / "value_head.pt", map_location=self.device, weights_only=True
        )
        self.value_head.load_state_dict(state)


__all__ = ["GenerationOutput", "PolicyModel"]
