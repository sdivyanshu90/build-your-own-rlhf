terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- GKE cluster (control plane only; node pools defined separately) ---------
resource "google_container_cluster" "rlhf" {
  name                     = var.cluster_name
  location                 = var.zone
  remove_default_node_pool = true
  initial_node_count       = 1

  release_channel {
    channel = "STABLE"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# --- Autoscaling GPU node pool ----------------------------------------------
resource "google_container_node_pool" "gpu" {
  name     = "gpu-pool"
  cluster  = google_container_cluster.rlhf.id
  location = var.zone

  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }

  node_config {
    machine_type = var.node_machine_type
    # Least-privilege scope; workload identity grants fine-grained access.
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count_per_node
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    labels = { workload = "rlhf-ppo" }
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }
}

# --- Object storage for checkpoints and datasets (versioned + encrypted) -----
resource "google_storage_bucket" "artifacts" {
  name                        = var.artifacts_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# --- Secret Manager secret for W&B / external service tokens -----------------
resource "google_secret_manager_secret" "wandb_api_key" {
  secret_id = "rlhf-wandb-api-key"
  replication {
    auto {}
  }
}

# --- Workload-identity service account with least-privilege bindings ---------
resource "google_service_account" "trainer" {
  account_id   = "rlhf-trainer"
  display_name = "RLHF-PPO training workload"
}

resource "google_storage_bucket_iam_member" "trainer_artifacts" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_secret_manager_secret_iam_member" "trainer_wandb" {
  secret_id = google_secret_manager_secret.wandb_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.trainer.email}"
}

output "cluster_endpoint" {
  value     = google_container_cluster.rlhf.endpoint
  sensitive = true
}

output "artifacts_bucket" {
  value = google_storage_bucket.artifacts.name
}
