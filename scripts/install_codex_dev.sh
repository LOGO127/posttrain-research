#!/usr/bin/env bash
# 在 KML dev 机器上安装 OpenAI codex CLI
# 依次尝试：① npm 全局安装 ② GitHub release 二进制 ③ ghproxy 镜像
set -euo pipefail

echo "=== arch/os ==="
uname -m && uname -s
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  CODEX_TRIPLE="x86_64-unknown-linux-musl" ;;
  aarch64) CODEX_TRIPLE="aarch64-unknown-linux-musl" ;;
  *) echo "unsupported arch: $ARCH"; exit 1 ;;
esac

INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$INSTALL_DIR"

# ---------- 方法1: npm ----------
if command -v npm >/dev/null 2>&1; then
    echo "=== 尝试 npm 安装 ==="
    if npm i -g @openai/codex 2>&1 | tail -3; then
        command -v codex && codex --version && exit 0
    fi
    echo "npm 直连失败，尝试内网 registry..."
    for reg in "https://npm.corp.kuaishou.com" "https://registry.npm.taobao.org" "https://registry.npmmirror.com"; do
        echo "--- registry: $reg"
        if npm i -g @openai/codex --registry="$reg" 2>&1 | tail -3; then
            command -v codex && codex --version && exit 0
        fi
    done
fi

# ---------- 方法2: GitHub release 二进制 ----------
echo "=== 尝试 GitHub release 二进制 ==="
# 获取最新版本号（带镜像回退）
get_latest() {
    for base in "https://api.github.com" "https://gh-proxy.com/https://api.github.com" "https://ghproxy.net/https://api.github.com"; do
        v=$(curl -fsSL --connect-timeout 8 "$base/repos/openai/codex/releases/latest" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | head -1 | grep -o 'rust-v[0-9.]*' | head -1)
        [ -n "$v" ] && echo "$v" && return 0
    done
    return 1
}
TAG=$(get_latest || true)
echo "latest tag: ${TAG:-unknown}"

if [ -n "$TAG" ]; then
    VER="${TAG#rust-v}"
    FNAME="codex-${CODEX_TRIPLE}.tar.gz"
    for base in "https://github.com" "https://gh-proxy.com/https://github.com" "https://ghproxy.net/https://github.com" "https://mirror.ghproxy.com/https://github.com"; do
        URL="$base/openai/codex/releases/download/${TAG}/${FNAME}"
        echo "--- try: $URL"
        if curl -fSL --connect-timeout 10 -o /tmp/codex.tar.gz "$URL" 2>/dev/null; then
            tar -xzf /tmp/codex.tar.gz -C /tmp
            # tarball 里通常是 codex-<triple> 单个二进制
            BIN=$(find /tmp -maxdepth 2 -name "codex*" -type f ! -name "*.tar.gz" | head -1)
            if [ -n "$BIN" ]; then
                mv "$BIN" "$INSTALL_DIR/codex" && chmod +x "$INSTALL_DIR/codex"
                "$INSTALL_DIR/codex" --version && echo "installed -> $INSTALL_DIR/codex" && exit 0
            fi
        fi
    done
fi

echo "=== 所有方法失败 ==="
echo "手动方案：在能访问 GitHub 的机器下载 codex-${CODEX_TRIPLE}.tar.gz，"
echo "scp 到本机解压放 $INSTALL_DIR/codex 即可。"
exit 1
