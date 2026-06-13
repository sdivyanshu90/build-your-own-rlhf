"""
conftest — shared pytest fixtures for the RLHF-PPO test suite.

Provides deterministic seeding, a CPU/GPU device fixture, a tiny randomly-
initialized GPT-2 (no weight download), policy / reward-model wrappers around it,
and synthetic preference / rollout data with known properties. Model fixtures
import :mod:`rlhf.models` lazily so that the pure-math unit tests can run before
those modules exist / are imported.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase

from rlhf.data.schemas import Preference, Rollout

# Vendored GPT-2 tokenizer (tests/fixtures/gpt2) so the suite is fully offline and
# hermetic — CI never depends on a Hugging Face Hub download.
_TOKENIZER_DIR = Path(__file__).parent / "fixtures" / "gpt2"

if TYPE_CHECKING:
    from rlhf.models.policy import PolicyModel
    from rlhf.models.reference_model import ReferenceModel
    from rlhf.models.reward_model import RewardModel

SEED = 42
TINY_VOCAB = 128
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def _set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for deterministic tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def seeded_rng() -> None:
    """Autouse fixture: reseed every RNG to a fixed value before each test."""
    _set_all_seeds(SEED)


@pytest.fixture
def device() -> torch.device:
    """The compute device: CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_config() -> GPT2Config:
    """A 2-layer, 64-hidden GPT-2 config with a small vocabulary (no download)."""
    return GPT2Config(
        vocab_size=TINY_VOCAB,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )


@pytest.fixture
def tiny_gpt2(tiny_config: GPT2Config) -> GPT2LMHeadModel:
    """A freshly initialized tiny GPT-2 causal LM (random weights, no download)."""
    model = GPT2LMHeadModel(tiny_config)
    model.eval()
    return model


@pytest.fixture
def tiny_policy(tiny_config: GPT2Config) -> PolicyModel:
    """A :class:`PolicyModel` wrapping a tiny GPT-2 with a value head."""
    from rlhf.models.policy import PolicyModel

    return PolicyModel.from_config(tiny_config, value_head_dropout=0.0)


@pytest.fixture
def tiny_reward_model(tiny_config: GPT2Config) -> RewardModel:
    """A :class:`RewardModel` wrapping a tiny GPT-2 with a scalar reward head."""
    from rlhf.models.reward_model import RewardModel

    return RewardModel.from_config(tiny_config, freeze_backbone=False)


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerBase:
    """The vendored GPT-2 tokenizer, loaded from a local path (no network)."""
    tok = AutoTokenizer.from_pretrained(str(_TOKENIZER_DIR))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def lm_config(tokenizer: PreTrainedTokenizerBase) -> GPT2Config:
    """A tiny GPT-2 config whose vocab matches the real tokenizer (for text I/O)."""
    return GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )


@pytest.fixture
def text_policy(lm_config: GPT2Config) -> PolicyModel:
    """A policy whose vocab matches the real tokenizer (for end-to-end tests)."""
    from rlhf.models.policy import PolicyModel

    return PolicyModel.from_config(lm_config, value_head_dropout=0.0)


@pytest.fixture
def text_reference(text_policy: PolicyModel) -> ReferenceModel:
    """A frozen reference snapshotted from ``text_policy``."""
    from rlhf.models.reference_model import ReferenceModel

    return ReferenceModel.from_policy(text_policy)


@pytest.fixture
def text_reward_model(lm_config: GPT2Config) -> RewardModel:
    """A reward model whose vocab matches the real tokenizer."""
    from rlhf.models.reward_model import RewardModel

    return RewardModel.from_config(lm_config, freeze_backbone=False)


def _make_rollout(seed: int, length: int) -> Rollout:
    """Build one synthetic rollout with deterministic, internally-consistent data."""
    g = torch.Generator().manual_seed(seed)
    prompt_ids = torch.randint(3, TINY_VOCAB, (4,), generator=g).tolist()
    response_ids = torch.randint(3, TINY_VOCAB, (length,), generator=g).tolist()
    # Log-probs are negative; reference is slightly different so KL != 0.
    logprobs = (-torch.rand(length, generator=g)).tolist()
    ref_logprobs = (-torch.rand(length, generator=g)).tolist()
    values = torch.randn(length, generator=g).tolist()
    reward = float(torch.randn(1, generator=g).item())
    return Rollout(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        logprobs=logprobs,
        ref_logprobs=ref_logprobs,
        values=values,
        reward=reward,
    )


@pytest.fixture
def dummy_rollouts() -> list[Rollout]:
    """Eight synthetic rollouts of varying lengths with known structure."""
    lengths = [5, 3, 7, 4, 6, 2, 8, 5]
    return [_make_rollout(seed=100 + i, length=length) for i, length in enumerate(lengths)]


@pytest.fixture
def dummy_preferences() -> list[Preference]:
    """
    One hundred synthetic preferences where ``chosen`` deterministically beats
    ``rejected`` (the chosen text encodes a strictly higher latent score).
    """
    prefs: list[Preference] = []
    for i in range(100):
        score_chosen = 1.0 + (i % 10) * 0.1
        score_rejected = score_chosen - 0.5
        prefs.append(
            Preference(
                prompt=f"prompt number {i}",
                chosen=f"high quality answer {i} " * 2,
                rejected=f"low quality answer {i}",
                annotator_id=f"annotator-{i % 5}",
                score_chosen=score_chosen,
                score_rejected=score_rejected,
            )
        )
    return prefs


@pytest.fixture
def filled_buffer(dummy_rollouts: list[Rollout]) -> Iterator[object]:
    """A :class:`RolloutBuffer` pre-loaded with the eight dummy rollouts."""
    from rlhf.training.ppo.rollout import RolloutBuffer

    buffer = RolloutBuffer(capacity=len(dummy_rollouts), pad_token_id=PAD_ID)
    for roll in dummy_rollouts:
        buffer.push(
            prompt_ids=torch.tensor(roll.prompt_ids),
            response_ids=torch.tensor(roll.response_ids),
            logprobs=torch.tensor(roll.logprobs),
            ref_logprobs=torch.tensor(roll.ref_logprobs),
            values=torch.tensor(roll.values),
            reward=roll.reward,
        )
    yield buffer
    buffer.clear()
