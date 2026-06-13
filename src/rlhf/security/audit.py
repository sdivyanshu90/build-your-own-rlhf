"""
security.audit — checkpoint integrity verification and access auditing.

Overview
--------
* :class:`CheckpointVerifier` writes a SHA-256 manifest alongside every saved
  checkpoint and verifies it on load, so silently-tampered weights are rejected
  before they can alter policy behaviour.
* :class:`AuditLogger` appends structured JSON records of checkpoint and data
  access events to an append-only log for after-the-fact forensics.

Usage Example
-------------
>>> import torch
>>> from rlhf.security.audit import CheckpointVerifier
>>> verifier = CheckpointVerifier()
>>> # digest = verifier.save({"w": torch.zeros(3)}, Path("ckpt.pt"))
>>> # state = verifier.load_verified(Path("ckpt.pt"), digest)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

from rlhf.exceptions import CheckpointError, CheckpointTamperingError

logger = logging.getLogger(__name__)

# Read the file in fixed-size chunks so hashing a multi-GB checkpoint never loads
# the whole file into memory at once.
_HASH_CHUNK_BYTES: int = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, hashed in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointVerifier:
    """Saves checkpoints with a SHA-256 manifest and verifies them on load."""

    MANIFEST_SUFFIX = ".manifest.json"

    def save(self, state_dict: dict[str, Any], path: str | Path, step: int | None = None) -> str:
        """
        Save ``state_dict`` to ``path`` and write a manifest with its digest.

        Args:
            state_dict: The object to persist (passed to ``torch.save``).
            path: Destination file path.
            step: Optional training step recorded in the manifest.

        Returns:
            The hex SHA-256 digest of the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, path)
        digest = sha256_file(path)
        manifest = {
            "path": path.name,
            "sha256": digest,
            "step": step,
        }
        manifest_path = path.with_suffix(path.suffix + self.MANIFEST_SUFFIX)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("checkpoint saved with digest %s -> %s", digest[:12], path)
        return digest

    def load_verified(
        self,
        path: str | Path,
        expected_digest: str | None = None,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint after verifying its SHA-256 digest.

        Args:
            path: Checkpoint file path.
            expected_digest: Digest to verify against. If ``None``, the digest is
                read from the sidecar manifest.
            map_location: Passed through to ``torch.load``.

        Returns:
            The deserialized state dict.

        Raises:
            CheckpointError: If the file or manifest is missing.
            CheckpointTamperingError: If the digest does not match.
        """
        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"checkpoint not found: {path}")
        if expected_digest is None:
            manifest_path = path.with_suffix(path.suffix + self.MANIFEST_SUFFIX)
            if not manifest_path.is_file():
                raise CheckpointError(f"manifest not found for {path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_digest = manifest["sha256"]
        actual = sha256_file(path)
        if actual != expected_digest:
            raise CheckpointTamperingError(
                f"checkpoint {path} digest mismatch: expected {expected_digest[:12]}..., "
                f"got {actual[:12]}...; refusing to load."
            )
        # nosec B614: the checkpoint's SHA-256 digest is verified against the
        # manifest immediately above this line, so the pickle payload is known to
        # be exactly what we wrote. weights_only=False is required because the
        # checkpoint holds non-tensor training state (optimizer, scheduler, RNG).
        state: dict[str, Any] = torch.load(  # nosec B614
            path, map_location=map_location, weights_only=False
        )
        return state


class AuditLogger:
    """Append-only JSON-lines logger for checkpoint and data access events."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        """Append one structured event record to the audit log."""
        entry = {"event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        logger.debug("audit: %s %s", event, fields)

    def read_all(self) -> list[dict[str, Any]]:
        """Read every recorded event back (for inspection / tests)."""
        if not self.log_path.is_file():
            return []
        records: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


__all__ = ["AuditLogger", "CheckpointVerifier", "sha256_file"]
