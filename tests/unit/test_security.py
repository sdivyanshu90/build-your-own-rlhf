"""Unit tests for security.audit (checkpoint integrity + audit log)."""

from __future__ import annotations

import pytest
import torch

from rlhf.exceptions import CheckpointError, CheckpointTamperingError
from rlhf.security.audit import AuditLogger, CheckpointVerifier, sha256_file


def test_checkpoint_save_and_verified_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    verifier = CheckpointVerifier()
    state = {"w": torch.arange(5).float()}
    path = tmp_path / "ckpt.pt"
    digest = verifier.save(state, path, step=7)
    assert path.is_file()
    # Manifest sidecar exists and load_verified round-trips the state.
    loaded = verifier.load_verified(path, digest)
    assert torch.equal(loaded["w"], state["w"])
    # With no explicit digest, the manifest's digest is used.
    loaded2 = verifier.load_verified(path)
    assert torch.equal(loaded2["w"], state["w"])


def test_checkpoint_tampering_detected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    verifier = CheckpointVerifier()
    path = tmp_path / "ckpt.pt"
    digest = verifier.save({"w": torch.zeros(3)}, path)
    with pytest.raises(CheckpointTamperingError):
        verifier.load_verified(path, "deadbeef" * 8)
    assert digest != "deadbeef" * 8


def test_load_missing_checkpoint_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    verifier = CheckpointVerifier()
    with pytest.raises(CheckpointError):
        verifier.load_verified(tmp_path / "nope.pt", "abc")


def test_load_missing_manifest_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    verifier = CheckpointVerifier()
    path = tmp_path / "ckpt.pt"
    torch.save({"w": torch.zeros(1)}, path)  # saved without a manifest
    with pytest.raises(CheckpointError):
        verifier.load_verified(path)


def test_sha256_file_stable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_audit_logger_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = AuditLogger(tmp_path / "audit.log")
    logger.record("checkpoint_saved", step=10, path="ckpt.pt")
    logger.record("data_access", dataset="prefs")
    records = logger.read_all()
    assert len(records) == 2
    assert records[0]["event"] == "checkpoint_saved"
    assert records[1]["dataset"] == "prefs"


def test_audit_logger_empty_read(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = AuditLogger(tmp_path / "missing.log")
    assert logger.read_all() == []
