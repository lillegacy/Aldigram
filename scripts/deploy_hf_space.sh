#!/usr/bin/env bash
# Assemble and upload the Aldigram app to a Hugging Face Docker Space.
#
# Usage:
#   python -m pip install -U huggingface_hub
#   hf auth login                      # or: export HF_TOKEN=...
#   scripts/deploy_hf_space.sh <hf-user>/<space-name> [--public|--private]
#
# The Space is created if it does not exist. Default visibility: private.
set -euo pipefail

SPACE_ID="${1:-}"
VISIBILITY="${2:---private}"
if [ -z "$SPACE_ID" ]; then
  echo "Usage: $0 <hf-username>/<space-name> [--public|--private]" >&2
  exit 1
fi
if [ "$VISIBILITY" != "--public" ] && [ "$VISIBILITY" != "--private" ]; then
  echo "Visibility must be --public or --private" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${TMPDIR:-/tmp}/iact-hf-space-$$"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"

# Stage only what the Space needs, excluding dev artifacts.
tar -C "$ROOT" -cf - \
  --exclude='frontend/node_modules' \
  --exclude='frontend/dist' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  backend frontend IACTDashboard.jsx Dockerfile .dockerignore \
  | tar -C "$STAGE" -xf -

cp "$ROOT/deploy/hf_space/README.md" "$STAGE/README.md"

python - "$SPACE_ID" "$VISIBILITY" "$STAGE" <<'PY'
import sys

from huggingface_hub import HfApi

space_id, visibility, folder = sys.argv[1:4]
api = HfApi()
api.create_repo(
    repo_id=space_id,
    repo_type="space",
    space_sdk="docker",
    private=(visibility != "--public"),
    exist_ok=True,
)
# Enforce visibility even when the Space already existed.
try:
    api.update_repo_settings(
        repo_id=space_id,
        repo_type="space",
        private=(visibility != "--public"),
    )
except (AttributeError, TypeError):
    pass
api.upload_folder(
    repo_id=space_id,
    repo_type="space",
    folder_path=folder,
    commit_message="Deploy Aldigram IACT dashboard",
)
print(f"\nDeployed: https://huggingface.co/spaces/{space_id}")
print(f"App URL : https://{space_id.replace('/', '-').lower()}.hf.space")
PY
