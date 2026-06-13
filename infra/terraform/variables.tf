variable "project_id" {
  description = "GCP project ID hosting the RLHF training cluster."
  type        = string
}

variable "region" {
  description = "GCP region for regional resources."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the GPU node pool."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "Name of the GKE cluster."
  type        = string
  default     = "rlhf-ppo"
}

variable "gpu_type" {
  description = "Accelerator type for the GPU node pool."
  type        = string
  default     = "nvidia-tesla-a100"
}

variable "gpu_count_per_node" {
  description = "Number of GPUs attached to each node."
  type        = number
  default     = 1
}

variable "node_machine_type" {
  description = "Machine type for GPU nodes."
  type        = string
  default     = "a2-highgpu-1g"
}

variable "min_nodes" {
  description = "Minimum number of GPU nodes (autoscaling)."
  type        = number
  default     = 0
}

variable "max_nodes" {
  description = "Maximum number of GPU nodes (autoscaling)."
  type        = number
  default     = 4
}

variable "artifacts_bucket" {
  description = "Name of the GCS bucket for checkpoints and datasets."
  type        = string
}
