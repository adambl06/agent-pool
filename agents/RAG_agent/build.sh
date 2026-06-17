#!/bin/bash
set -e

# ── Configuration ───────────────────────────────────────────────────────────
REGISTRY="${REGISTRY:-your-registry}"     # e.g. docker.io/yourusername or ghcr.io/yourorg
IMAGE_NAME="${IMAGE_NAME:-rag-agent}"
TAG="${TAG:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
TMP_DIR=$(mktemp -d)
CPU_NODE="cpu-node"

# ── Build ────────────────────────────────────────────────────────────────────
echo "Building image: ${FULL_IMAGE}"
docker build \
  --platform linux/amd64 \
  -t "${FULL_IMAGE}" \
  -f "$(dirname "$0")/Dockerfile" \
  "$(dirname "$0")/../.."     # build context = repo root

# ── Push (optional) ──────────────────────────────────────────────────────────
if [[ "${PUSH:-false}" == "true" ]]; then
  echo "Pushing image: ${FULL_IMAGE}"
  docker push "${FULL_IMAGE}"
fi

echo "Done: ${FULL_IMAGE}"

echo "Exporting images to tar files..."
docker save $FULL_IMAGE  -o "$TMP_DIR/rag-agent.tar"
echo "Images exported"

echo "Transferring images to $CPU_NODE..."
scp "$TMP_DIR/rag-agent.tar"  $CPU_NODE:/tmp/rag-agent.tar
echo "Images transferred"


echo "Importing images into k3s containerd on $CPU_NODE..."
ssh $CPU_NODE -c "sudo k3s ctr images import /tmp/rag-agent.tar  && sudo rm /tmp/rag-agent.tar"
echo "Images imported"

echo "Applying Kubernetes manifests..."
export KUBECONFIG=~/.kube/ai-adam-config


kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
echo "Manifests applied"

echo "Restarting deployments..."
kubectl rollout restart deployment/rag-agent
echo "Deployments restarted"