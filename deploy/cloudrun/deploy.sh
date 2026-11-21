#!/usr/bin/env bash
# Build with Cloud Build, push to Artifact Registry, and deploy to Cloud Run.
#
#   PROJECT_ID=my-proj REGION=us-central1 ./deploy/cloudrun/deploy.sh
#
# Optional: create the secret first so the service can use a real LLM:
#   printf '%s' "$OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-pm-research-agent}"
REPO="${REPO:-pm-research}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/api:${TAG}"

cd "$(dirname "$0")/../.."

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com --project "$PROJECT_ID"

gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" \
  >/dev/null 2>&1 || gcloud artifacts repositories create "$REPO" \
  --repository-format docker --location "$REGION" --project "$PROJECT_ID"

gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID" .

SECRET_FLAGS=()
if gcloud secrets describe openai-api-key --project "$PROJECT_ID" >/dev/null 2>&1; then
  SECRET_FLAGS=(--set-secrets "OPENAI_API_KEY=openai-api-key:latest"
                --set-env-vars "PMRA_LLM_PROVIDER=openai")
fi

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" --project "$PROJECT_ID" \
  --platform managed --port 8080 --cpu 1 --memory 1Gi \
  --min-instances 0 --max-instances 3 --concurrency 20 --timeout 300 \
  --set-env-vars "PMRA_VECTOR_STORE=qdrant,PMRA_QDRANT_URL=${PMRA_QDRANT_URL:-:memory:},PMRA_SOURCES=google_news,gdelt" \
  "${SECRET_FLAGS[@]}" \
  --no-allow-unauthenticated

echo "Deployed: $(gcloud run services describe "$SERVICE" --region "$REGION" \
  --project "$PROJECT_ID" --format 'value(status.url)')"
