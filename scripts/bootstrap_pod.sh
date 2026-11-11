#!/usr/bin/env bash
# ============================================================
# KML pod 一键 bootstrap：装 codex + 克隆项目 + 数据准备
# 用法：整段粘贴到 KML web terminal 执行
#   bash <(curl -fsSL https://raw.githubusercontent.com/LOGO127/posttrain-research/main/scripts/bootstrap_pod.sh)
# 或本文件直接粘贴执行
# ============================================================
set -x
echo "===== 0. 环境检查 ====="
nvidia-smi 2>/dev/null | head -12 || echo "no nvidia-smi"
uname -m; python3 --version 2>/dev/null; df -h . | tail -1

WORKDIR="${WORKDIR:-$HOME/posttrain-work}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

# ---------- 1. 安装 codex ----------
echo "===== 1. codex 安装 ====="
if command -v codex >/dev/null 2>&1; then
    echo "codex already installed: $(codex --version 2>/dev/null)"
else
    ARCH=$(uname -m)
    [ "$ARCH" = "x86_64" ] && TRIPLE="x86_64-unknown-linux-musl" || TRIPLE="aarch64-unknown-linux-musl"
    TAG=""
    for api in "https://api.github.com" "https://gh-proxy.com/https://api.github.com" "https://ghproxy.net/https://api.github.com"; do
        TAG=$(curl -fsSL --connect-timeout 8 "$api/repos/openai/codex/releases/latest" 2>/dev/null | grep -o 'rust-v[0-9.]*' | head -1)
        [ -n "$TAG" ] && break
    done
    echo "latest codex tag: ${TAG:-NOT_FOUND}"
    if [ -n "$TAG" ]; then
        for base in "https://github.com" "https://gh-proxy.com/https://github.com" "https://ghproxy.net/https://github.com" "https://mirror.ghproxy.com/https://github.com"; do
            url="$base/openai/codex/releases/download/$TAG/codex-$TRIPLE.tar.gz"
            echo "--- try $url"
            if curl -fSL --connect-timeout 10 -o /tmp/codex.tar.gz "$url" 2>/dev/null; then
                mkdir -p /tmp/codex_x && tar -xzf /tmp/codex.tar.gz -C /tmp/codex_x
                BIN=$(find /tmp/codex_x -type f -name "codex*" | head -1)
                [ -n "$BIN" ] && mkdir -p "$HOME/.local/bin" && mv "$BIN" "$HOME/.local/bin/codex" && chmod +x "$HOME/.local/bin/codex" && break
            fi
        done
    fi
    export PATH="$HOME/.local/bin:$PATH"
    command -v codex && codex --version || echo "!!! codex 安装失败（网络不通）；可本机下载 codex-$TRIPLE.tar.gz 后上传到 pod 解压"
fi

# ---------- 2. 克隆项目 ----------
echo "===== 2. clone 项目 ====="
if [ ! -d posttrain-research ]; then
    git clone https://github.com/LOGO127/posttrain-research.git 2>&1 | tail -3 || \
    git clone https://gh-proxy.com/https://github.com/LOGO127/posttrain-research.git 2>&1 | tail -3 || \
    git clone https://ghproxy.net/https://github.com/LOGO127/posttrain-research.git 2>&1 | tail -3
fi
cd posttrain-research 2>/dev/null || { echo "clone 失败"; }

# ---------- 3. python 环境 + 数据 ----------
echo "===== 3. python venv + 数据 ====="
if [ ! -d .venv ]; then
    python3 -m venv .venv 2>/dev/null || python3.10 -m venv .venv 2>/dev/null || python3.11 -m venv .venv 2>/dev/null
fi
. .venv/bin/activate
pip install -q --upgrade pip
# Blackwell sm120 需要 cu128 torch；若 pod 已预装 torch 则跳过
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt
python data/prepare_data.py build_gsm8k --out data/gsm8k_en_train.jsonl
python data/prepare_data.py build_mgsm --langs en zh fr es --out data/mgsm_eval/

echo "===== 完成 ====="
echo "codex 使用：export PATH=\$HOME/.local/bin:\$PATH"
echo "           export OPENAI_API_KEY=<key>   # 或 codex login（headless 用 api key）"
echo "项目目录：$WORKDIR/posttrain-research"
echo "冒烟测试：python train/grpo_lora.py --model Qwen/Qwen3-1.7B --dataset data/gsm8k_en_train.jsonl --output_dir runs/smoke --max_steps 50"
