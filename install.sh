#!/usr/bin/env bash
set -e

# ═══════════════════════════════════════════════════════════
# RCMS — AstrBot 插件安装脚本
# ═══════════════════════════════════════════════════════════
#
# 用法:
#   bash install.sh                    # 交互式安装
#   bash install.sh /path/to/astrbot   # 指定 AstrBot 目录
#
# 这个脚本会:
#   1. 复制 RCMS 插件到 AstrBot 的插件目录
#   2. 安装 Python 依赖（numpy, openai）
#   3. 提示配置 api_key
# ═══════════════════════════════════════════════════════════

RCMS_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[RCMS]${NC} $1"; }
ok()    { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[✗]${NC} $1"; }

# ── 找 AstrBot 目录 ──

if [ $# -ge 1 ] && [ -n "$1" ]; then
    ASTRBOT_DIR="$1"
else
    # 常见位置
    CANDIDATES=(
        "$(pwd)"
        "$(pwd)/astrbot"
        "$HOME/astrbot"
        "$HOME/AstrBot"
        "/opt/astrbot"
    )
    ASTRBOT_DIR=""
    for d in "${CANDIDATES[@]}"; do
        if [ -f "$d/main.py" ] || [ -d "$d/astrbot" ]; then
            ASTRBOT_DIR="$d"
            break
        fi
    done
fi

if [ -z "$ASTRBOT_DIR" ] || [ ! -d "$ASTRBOT_DIR" ]; then
    echo ""
    err "未找到 AstrBot 目录"
    echo "  请指定: bash install.sh /path/to/astrbot"
    exit 1
fi

ok "AstrBot 目录: $ASTRBOT_DIR"

# ── 确认插件目录 ──

PLUGIN_DIR="$ASTRBOT_DIR/plugins/rcms-astrbot"
mkdir -p "$PLUGIN_DIR"

# ── 安装依赖 ──

info "安装 Python 依赖..."
pip3 install numpy openai 2>/dev/null || pip install numpy openai 2>/dev/null
ok "依赖安装完成"

# ── 复制插件文件 ──

info "复制 RCMS 文件..."
# 复制核心
cp "$RCMS_ROOT/rcms_core.py" "$PLUGIN_DIR/"
# 复制插件
cp -r "$RCMS_ROOT/plugins/rcms-astrbot/"* "$PLUGIN_DIR/" 2>/dev/null || true
# 复制 backends + scripts
mkdir -p "$PLUGIN_DIR/backends"
mkdir -p "$PLUGIN_DIR/scripts"
cp -r "$RCMS_ROOT/backends/"* "$PLUGIN_DIR/backends/" 2>/dev/null || true
cp -r "$RCMS_ROOT/scripts/"* "$PLUGIN_DIR/scripts/" 2>/dev/null || true
# 复制配置（不覆盖已有）
if [ ! -f "$PLUGIN_DIR/config.json" ]; then
    cp "$RCMS_ROOT/config.json" "$PLUGIN_DIR/"
    ok "已创建 config.json"
else
    warn "config.json 已存在，跳过"
fi

ok "RCMS 文件已复制到 $PLUGIN_DIR"

# ── API Key 提示 ──

CONFIG_FILE="$PLUGIN_DIR/config.json"
if [ -f "$CONFIG_FILE" ]; then
    # 检查是否已配置
    HAS_KEY=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('analysis',{}).get('retrieval',{}).get('api_key','') or '')" 2>/dev/null)
    if [ -z "$HAS_KEY" ]; then
        echo ""
        warn "⚠ 未配置 API Key"
        echo ""
        echo "  RCMS 需要 OpenAI 兼容 API 来使用 Embedding 和 ANALYSIS 功能。"
        echo "  配置方式："
        echo "    1. 设置环境变量: export OPENAI_API_KEY=sk-xxxxx"
        echo "    2. 编辑 config.json:"
        echo "       analysis.retrieval.api_key"
        echo "       analysis.post_analysis.api_key"
        echo ""
        echo "  如果使用 AstrBot 内建 provider（不填 key），RCMS 会自动读取"
        echo "  AstrBot 的 cmd_config.json 中的 active provider。"
        echo ""
    fi
fi

# ── 验证 ──

info "验证安装..."
python3 -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR')
from rcms_core import MinimalRCMS
r = MinimalRCMS()
tables = r.conn.execute(\"SELECT count(*) FROM sqlite_master WHERE type='table'\").fetchone()[0]
r.close()
print(f'  RCMS 初始化成功: {tables} 个表')
" 2>&1 && ok "核心库验证通过" || err "核心库验证失败"

# ── 完成 ──

echo ""
ok "RCMS AstrBot 插件安装完成！"
echo ""
echo "  📁 插件位置: $PLUGIN_DIR"
echo "  🔧 配置文件: $CONFIG_FILE"
echo ""
echo "  启用方式："
echo "    1. 在 AstrBot 管理面板中启用 rcms 插件"
echo "    2. 或编辑 AstrBot cmd_config.json 的 plugins 列表"
echo ""
echo "  检查运行状态："
echo "    cd $PLUGIN_DIR && python3 scripts/check_rcms.py"
echo ""
