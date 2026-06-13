# Deployment

## Container

```bash
docker build -t rlhf-ppo:latest -f infra/Dockerfile .
docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" rlhf-ppo:latest ppo --config /app/config.yaml
```

The image is a multi-stage build that runs as **non-root UID 10001** and bakes no
secrets. Mount config via a volume / Kubernetes ConfigMap; supply tokens via
environment or a secrets manager.

## Kubernetes (Helm)

```bash
helm upgrade --install rlhf-ppo ./infra/helm/rlhf-ppo \
  --namespace rlhf --create-namespace \
  --set image.tag=$VERSION --set stage=ppo
```

The chart renders a batch `Job` with a GPU `resources` request, a hardened
`securityContext` (non-root, all capabilities dropped), a workspace PVC, and a
ConfigMap-mounted config. Set `stage` to `sft`, `reward`, or `ppo`.

## Infrastructure (Terraform)

`infra/terraform` provisions a GKE cluster with an **autoscaling GPU node pool**
(min 0 → scale to zero when idle), a **versioned, lifecycle-managed GCS bucket**
for checkpoints/datasets, a **Secret Manager** secret for the W&B key, and a
**workload-identity service account** with least-privilege bindings.

```bash
cd infra/terraform
terraform init
terraform apply -var project_id=$GCP_PROJECT -var artifacts_bucket=$BUCKET
```

## Export for serving

```bash
python scripts/export_onnx.py --config config.yaml --output policy.onnx
```

Exports the policy backbone (without the training-only value head) to ONNX with
dynamic batch/sequence axes.

## Rollback safety

Every checkpoint is written with a SHA-256 manifest by
`security.audit.CheckpointVerifier` and verified on load, so a tampered or
corrupted checkpoint is rejected (`CheckpointTamperingError`) rather than
silently deployed. Keep the last known-good `final.pt` + manifest to roll back.
