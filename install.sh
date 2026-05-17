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
#   3. 提示配置 API（导入 AstrBot 提供商或手动填写 url/token/model）
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
# 复制核心（包目录）
cp -r "$RCMS_ROOT/rcms_core" "$PLUGIN_DIR/"
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

# ── 检测 AstrBot 可用的模型提供商 ──

CMD_CONFIG="$ASTRBOT_DIR/data/cmd_config.json"
PROVIDER_INFO=""

if [ -f "$CMD_CONFIG" ]; then
    PROVIDER_INFO=$(python3 -c "
import json
with open('$CMD_CONFIG', encoding='utf-8-sig') as f:
    cfg = json.load(f)
sources = cfg.get('provider_sources', [])
if not sources:
    print('none')
else:
    for s in sources:
        sid = s.get('id', '?')
        stype = s.get('type', '?')
        api_base = s.get('api_base', 'https://api.openai.com/v1')
        keys = s.get('key', [])
        has_key = 'yes' if (keys and keys[0]) else 'no'
        print(f'{sid}|{stype}|{api_base}|{has_key}')
" 2>/dev/null)
fi

if [ -n "$PROVIDER_INFO" ] && [ "$PROVIDER_INFO" != "none" ]; then
    echo ""
    info "检测到 AstrBot 已配置的模型提供商："
    echo "$PROVIDER_INFO" | while IFS='|' read -r sid stype base key; do
        echo "    - $sid ($stype, $base, API Key: $key)"
    done
    echo ""
    info "RCMS 默认使用 'source=astrbot'（自动匹配 AstrBot 当前启用的提供商）。"
    info "如需指定某提供商，安装后在插件设置中将 api.retrieval.astrbot_source_id"
    info "或 api.post_analysis.astrbot_source_id 设为对应的提供商 ID。"
fi

# ── 配置 API ──

CONFIG_FILE="$PLUGIN_DIR/config.json"
if [ ! -f "$CONFIG_FILE" ] && [ -f "$RCMS_ROOT/config.json" ]; then
    cp "$RCMS_ROOT/config.json" "$CONFIG_FILE"
fi

if [ -f "$CONFIG_FILE" ]; then
    HAS_ASTRBOT=$(python3 -c "
import json
c = json.load(open('$CONFIG_FILE'))
r = c.get('api', {}).get('retrieval', {})
p = c.get('api', {}).get('post_analysis', {})
# 检查是否有自定义配置或环境变量
import os
ek = r.get('custom_token', '') or os.environ.get('OPENAI_API_KEY', '')
lk = p.get('custom_token', '') or os.environ.get('OPENAI_API_KEY', '')
print('custom' if (ek or lk) else 'astrbot')
" 2>/dev/null || echo "unknown")

    if [ "$HAS_ASTRBOT" = "astrbot" ] && [ -n "$PROVIDER_INFO" ] && [ "$PROVIDER_INFO" != "none" ]; then
        echo ""
        info "当前使用 AstrBot 提供商，无需额外配置。"
    elif [ "$HAS_ASTRBOT" != "custom" ]; then
        echo ""
        warn "⚠ 未检测到 API Key 配置"
        echo ""
        echo "  RCMS 支持两种配置方式："
        echo "    1. 使用 AstrBot 现有提供商（默认，推荐）"
        echo "    2. 手动输入 API URL / Token / 模型名（自定义）"
        echo ""
        echo "  方式 1 无需额外操作，插件启动后自动读取 AstrBot 配置。"
        echo ""
        read -r -p "  是否现在配置自定义 API（用于 standalone 模式）？(y/N): " SETUP_CUSTOM
        if [ "$SETUP_CUSTOM" = "y" ] || [ "$SETUP_CUSTOM" = "Y" ]; then
            echo ""
            read -r -p "  Embedding API Token: " EMB_TOKEN
            read -r -p "  Embedding API URL (默认 https://api.openai.com/v1): " EMB_URL
            EMB_URL=${EMB_URL:-https://api.openai.com/v1}
            read -r -p "  Embedding 模型 (默认 text-embedding-3-small): " EMB_MODEL
            EMB_MODEL=${EMB_MODEL:-text-embedding-3-small}
            echo ""
            read -r -p "  ANALYSIS LLM Token (留空同 Embedding): " LLM_TOKEN
            LLM_TOKEN=${LLM_TOKEN:-$EMB_TOKEN}
            read -r -p "  ANALYSIS LLM URL (默认 https://api.openai.com/v1): " LLM_URL
            LLM_URL=${LLM_URL:-https://api.openai.com/v1}
            read -r -p "  ANALYSIS LLM 模型 (默认 gpt-4o-mini): " LLM_MODEL
            LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}

            python3 -c "
import json
with open('$CONFIG_FILE', encoding='utf-8') as f:
    c = json.load(f)
c.setdefault('api', {}).setdefault('retrieval', {})['source'] = 'custom'
c['api']['retrieval']['custom_token'] = '$EMB_TOKEN'
c['api']['retrieval']['custom_url'] = '$EMB_URL'
c['api']['retrieval']['custom_model'] = '$EMB_MODEL'
c.setdefault('api', {}).setdefault('post_analysis', {})['source'] = 'custom'
c['api']['post_analysis']['custom_token'] = '$LLM_TOKEN'
c['api']['post_analysis']['custom_url'] = '$LLM_URL'
c['api']['post_analysis']['custom_model'] = '$LLM_MODEL'
with open('$CONFIG_FILE', 'w', encoding='utf-8') as f:
    json.dump(c, f, ensure_ascii=False, indent=2)
" 2>&1 && ok "自定义 API 配置已保存" || warn "配置保存失败"
        fi
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
