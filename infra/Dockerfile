# syntax=docker/dockerfile:1
# Multi-stage build for the RLHF-PPO pipeline. Runs as a non-root user and bakes
# no secrets into the image. Build context is the repository root.

# ---------------------------------------------------------------------------
# Stage 1: builder — produce a wheel from the source tree.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder
WORKDIR /build

# Copy only what the build backend needs to produce the wheel.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir hatchling hatch \
    && hatch build -t wheel

# ---------------------------------------------------------------------------
# Stage 2: runtime — CUDA base, install the wheel, drop privileges.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
WORKDIR /app

# Minimal Python runtime; clean apt lists to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Non-root user (fixed UID for reproducible Kubernetes securityContext).
RUN useradd --create-home --uid 10001 rlhf
USER 10001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HOME=/home/rlhf/.cache/huggingface

ENTRYPOINT ["python", "-m", "rlhf"]
CMD ["--help"]
