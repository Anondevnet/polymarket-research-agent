variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  type        = string
  description = "Container image, e.g. us-central1-docker.pkg.dev/PROJECT/pm-research/api:TAG"
}

variable "llm_provider" {
  type    = string
  default = "openai"
}

variable "qdrant_url" {
  type        = string
  default     = ":memory:"
  description = "Qdrant Cloud URL for persistence, or :memory: for an ephemeral per-instance store"
}

variable "invoker_members" {
  type        = list(string)
  default     = []
  description = "IAM members allowed to call the service, e.g. [\"user:me@example.com\"]"
}
