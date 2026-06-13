# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Instead, email
`security@rlhf-ppo.example` with:

- a description of the vulnerability and its impact,
- steps to reproduce (a minimal PoC if possible),
- any suggested mitigation.

We aim to acknowledge reports within 2 business days and to provide a remediation
timeline within 7 days. Please allow us a reasonable disclosure window before
publishing.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Implemented controls

This project ships defenses, not just documentation (see
`docs/architecture/overview.md` for the full threat model):

- **Checkpoint integrity** — SHA-256 manifests written and verified on load
  (`rlhf.security.audit.CheckpointVerifier`).
- **Prompt-injection guards** — control-character stripping, length truncation,
  and a configurable injection blocklist (`rlhf.security.validation`).
- **Reward-hacking detection** — KL penalty, ensemble-uncertainty canary, and
  dispersion/saturation early-stop.
- **Least-privilege runtime** — non-root container (UID 10001), hardened Helm
  `securityContext`, workload-identity service account.
- **Supply-chain scanning** — Bandit, `pip-audit`, and Trivy run in CI on every
  PR and weekly.
