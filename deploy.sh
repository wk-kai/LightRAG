#!/bin/bash
##############################################
# LightRAG 3080 Server One-Click Deploy
# Usage: bash deploy.sh
##############################################
set -e

echo "=== LightRAG 3080 Server Deploy ==="

# 1. Clone
if [ ! -d "LightRAG" ]; then
    git clone https://github.com/wk-kai/LightRAG.git
fi
cd LightRAG
git checkout dev-wangkai
git pull origin dev-wangkai

# 2. Config
cp .env.server .env
echo ""
echo ">>> IMPORTANT: Edit .env and fill in your DeepSeek API key!"
echo "    LLM_BINDING_API_KEY=sk-xxx"
echo "    VLM_LLM_BINDING_API_KEY=sk-xxx"
read -p "Press Enter after editing .env..."

# 3. Python deps
if command -v uv &> /dev/null; then
    uv sync --extra api --extra offline-storage
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[api,offline-storage]"
fi

# 4. Frontend build
if command -v bun &> /dev/null; then
    cd lightrag_webui && bun install --frozen-lockfile && bun run build && cd ..
else
    echo "bun not installed. Install from https://bun.sh"
    exit 1
fi

# 5. Start
echo "=== Starting LightRAG on http://0.0.0.0:9621 ==="
python -m lightrag.api.lightrag_server
