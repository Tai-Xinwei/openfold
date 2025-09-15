#!/bin/bash
# Usage: bash download_openfold_params_hf.sh /path/to/save YOUR_HF_TOKEN

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <download_dir> <huggingface_token>"
    exit 1
fi

TOKEN="$2"
DOWNLOAD_DIR="${1}/openfold_params/"
REPO_URL="https://${TOKEN}@hf-mirror.com/josephdiviano/openfold"

echo "📥 Downloading OpenFold params to: $DOWNLOAD_DIR"
mkdir -p "${DOWNLOAD_DIR}"
git clone "$REPO_URL" "${DOWNLOAD_DIR}"

# 删除 .git 目录（避免后续污染）
rm -rf "${DOWNLOAD_DIR}/.git"
