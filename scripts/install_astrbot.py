"""
安装 RCMS 到 AstrBot 插件目录

自动设置按人格分离的记忆存储（不同人格使用独立数据库）。

用法:
    python install_astrbot.py                    # 自动查找 ~/.astrbot
    python install_astrbot.py --plugin-dir D:/path/to/plugins  # 手动指定
    python install_astrbot.py --pull             # 先 git pull 再安装（含 --force）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

# Windows GBK 兼容
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 源文件: RCMS 项目根目录
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PLUGIN = os.path.join(_HERE, "plugins", "rcms-astrbot")
_CORE_DIRS = ["rcms_core"]
_BACKEND_DIR = os.path.join(_HERE, "backends")

PLUGIN_DIR_NAME = "astrbot_plugin_rcms"


def _find_root_by_config() -> str | None:
    """扫描可能位置，查找含 data/cmd_config.json 的 AstrBot 根目录"""
    candidates = [
        os.path.expanduser("~/.astrbot"),
        _HERE,
    ]
    for p in candidates:
        if os.path.isfile(os.path.join(p, "data", "cmd_config.json")):
            return p

    # 未命中标准路径 → 扫描常见部署目录（/www/xxxAstrBotxxx/）
    for root in ["/www", "/var/www", "/opt", "/home"]:
        if not os.path.isdir(root):
            continue
        try:
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if os.path.isdir(full) and "AstrBot" in entry:
                    if os.path.isfile(os.path.join(full, "data", "cmd_config.json")):
                        return full
        except PermissionError:
            continue
    return None


def derive_astrbot_root(target_dir: str) -> str | None:
    """从插件目录向上推导 AstrBot 根目录"""
    for d in [target_dir, os.path.dirname(target_dir),
              os.path.dirname(os.path.dirname(target_dir))]:
        if os.path.isfile(os.path.join(d, "data", "cmd_config.json")):
            return d
    return None


def find_astrbot_plugin_dir() -> str | None:
    """扫描所有能找到的 AstrBot 插件目录"""
    # 1. 用 cmd_config.json 定位根目录
    root = _find_root_by_config()
    if root:
        for candidate in ["plugins", "plugin"]:
            p = os.path.join(root, "data", candidate)
            if os.path.isdir(p):
                return p

    # 2. dev 模式
    for candidate in ["plugins", "plugin"]:
        p = os.path.join(_HERE, "data", candidate)
        if os.path.isdir(p):
            return p

    return None


def scan_astrbot_personas(astrbot_root: str) -> list[str]:
    """扫描 AstrBot 已配置的人格列表"""
    config_paths = [
        os.path.join(astrbot_root, "data", "cmd_config.json"),
    ]
    personas = []
    for cfg_path in config_paths:
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            provider_settings = cfg.get("provider_settings", {})
            personalities = provider_settings.get("personalities", [])
            for p in personalities:
                if isinstance(p, dict) and p.get("name"):
                    personas.append(p["name"])
            # v3 人格
            personas_v3 = provider_settings.get("personality_v3", [])
            for p in personas_v3:
                if isinstance(p, dict) and p.get("name"):
                    personas.append(p["name"])
        except Exception as e:
            print(f"  [!] 扫描人格信息失败: {e}")
            continue
    return personas


def _forward_api_config(rcms_config_path: str, astrbot_root: str):
    """从 AstrBot cmd_config.json 读取 API 配置，写入 RCMS 的 analysis 段"""
    cmd_path = os.path.join(astrbot_root, "data", "cmd_config.json")
    if not os.path.exists(cmd_path):
        print("  [!] 未找到 AstrBot cmd_config.json，跳过 API 导入")
        return

    try:
        with open(cmd_path, encoding="utf-8-sig") as f:
            cmd_cfg = json.load(f)
    except Exception as e:
        print(f"  [!] 读取 AstrBot 配置失败: {e}")
        return

    sources = {s["id"]: s for s in cmd_cfg.get("provider_sources", [])}

    # ── LLM 提供商（默认启用的） ──
    llm_src_id = ""
    llm_model = "gpt-4o"
    providers = [p for p in cmd_cfg.get("provider", []) if p.get("enable", False)]
    if providers:
        default_id = cmd_cfg.get("provider_settings", {}).get("default_provider_id", "")
        target = next((p for p in providers if p["id"] == default_id), providers[0])
        llm_src_id = target.get("provider_source_id", "")
        llm_model = target.get("model", "gpt-4o")
    llm_src = sources.get(llm_src_id) if llm_src_id else None

    # ── Embedding 提供商（AstrBot v3 在 provider[] 里，旧版在顶层 embedding_provider） ──
    emb_key = None
    emb_url = None
    emb_model = "text-embedding-3-small"
    emb_providers = [p for p in cmd_cfg.get("provider", [])
                     if p.get("enable", False) and (
                         p.get("type") == "openai_embedding"
                         or p.get("provider_type") == "embedding")]
    if emb_providers:
        ep = emb_providers[0]
        emb_key = ep.get("embedding_api_key", "") or None
        emb_url = ep.get("embedding_api_base", "") or None
        emb_model = ep.get("embedding_model", "text-embedding-3-small")
    else:
        # 旧版: 顶层 embedding_provider 字段
        ec = cmd_cfg.get("embedding_provider", {}) or {}
        src_id = ec.get("provider_source_id", "")
        if src_id:
            src = sources.get(src_id)
            if src:
                keys = src.get("key", [])
                emb_key = (keys[0] if isinstance(keys, list) and keys else "") or None
                emb_url = src.get("api_base", "")
            emb_model = ec.get("model", "text-embedding-3-small")
    # fallback: 未找到 embedding 提供商时复用 LLM
    if not emb_key and llm_src:
        keys = llm_src.get("key", [])
        emb_key = (keys[0] if isinstance(keys, list) and keys else "") or None
        emb_url = llm_src.get("api_base", "")
        emb_model = llm_model
    emb_src = None  # embedding 不再依赖 provider_sources，直接使用 emb_key/emb_url

    # ── 构造新 analysis 配置 ──
    analysis_retrieval = {"source": "astrbot", "astrbot_source_id": ""}
    analysis_post = {"source": "astrbot", "astrbot_source_id": llm_src_id}

    # 如能找到具体提供商则填入 custom_* 备用
    # retrieval 可能来自 embedding provider（直接 key/url）或 fallback（从 sources 取 key）
    def _set_custom(entry, key, url, mdl, default_mdl):
        entry["custom_url"] = url or "https://api.openai.com/v1"
        entry["custom_token"] = key or ""
        entry["custom_model"] = mdl or default_mdl
    _set_custom(analysis_retrieval, emb_key, emb_url, emb_model, "text-embedding-3-small")
    if llm_src:
        keys = llm_src.get("key", [])
        llm_key = (keys[0] if isinstance(keys, list) and keys else "") or ""
        llm_url = llm_src.get("api_base", "https://api.openai.com/v1")
    _set_custom(analysis_post, llm_key, llm_url, llm_model, "gpt-4o-mini")

    # ── 写入 RCMS config.json ──
    try:
        with open(rcms_config_path, encoding="utf-8") as f:
            rcms_cfg = json.load(f)
    except Exception:
        rcms_cfg = {}

    rcms_cfg.setdefault("analysis", {})
    rcms_cfg["analysis"]["retrieval"] = analysis_retrieval
    rcms_cfg["analysis"]["post_analysis"] = analysis_post

    with open(rcms_config_path, "w", encoding="utf-8") as f:
        json.dump(rcms_cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("  [OK] API 配置已导入")
    print(f"      Embedding: url={analysis_retrieval['custom_url']} model={analysis_retrieval['custom_model']}")
    print(f"      LLM:       url={analysis_post['custom_url']} model={analysis_post['custom_model']}")
    if emb_key or llm_src_id:
        print("      source=astrbot（自动读取 AstrBot 提供商）")


def install(target_dir: str, force: bool = False, forward_api: bool = False):
    plugin_dir = os.path.join(target_dir, PLUGIN_DIR_NAME)

    if os.path.exists(plugin_dir) and not force:
        print(f"发现已有目录: {plugin_dir}")
        ans = input("覆盖代码文件，config.json 和数据库保留? (y/N): ").strip().lower()
        if ans != "y":
            print("取消")
            return

    os.makedirs(plugin_dir, exist_ok=True)

    # 复制适配器文件
    for f in ["main.py", "metadata.yaml"]:
        src = os.path.join(_SRC_PLUGIN, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(plugin_dir, f))
            print(f"  [+] {f}")

    # 复制 RCMS 核心 + backends
    for d in _CORE_DIRS + ["backends"]:
        src = os.path.join(_HERE, d)
        dst = os.path.join(plugin_dir, d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            print(f"  [+] {d}/")

    # 复制 config.json（仅首次安装，不覆盖已有）
    config_src = os.path.join(_HERE, "config.json")
    config_dst = os.path.join(plugin_dir, "config.json")
    if os.path.exists(config_src) and not os.path.exists(config_dst):
        shutil.copy2(config_src, config_dst)
        print("  [+] config.json")
    elif os.path.exists(config_dst):
        print("  [=] config.json 已存在，保留")

    # 保留已有数据库（含人格分离后的多库）
    existing_dbs = [f for f in os.listdir(plugin_dir) if f.startswith("rcms_memory") and f.endswith(".db")]
    if existing_dbs:
        print(f"  [=] 已有 {len(existing_dbs)} 个数据库文件 (已保留)")

    # 扫描并提示人格信息（AstrBot 数据目录）
    astrbot_root = derive_astrbot_root(target_dir) or os.path.expanduser("~/.astrbot")
    personas = scan_astrbot_personas(astrbot_root)
    if personas:
        print(f"  [i] 检测到 {len(personas)} 个人格: {', '.join(personas)}")
        print("  [i] RCMS 将按人格自动分离记忆存储")
    else:
        print("  [i] 未检测到已配置的人格，将使用默认记忆库")

    # ── 导入 AstrBot API 配置 ──
    if forward_api:
        rcms_config = os.path.join(plugin_dir, "config.json")
        _forward_api_config(rcms_config, astrbot_root)
    else:
        # 仅检测并展示提供商信息
        cmd_config_path = os.path.join(astrbot_root, "data", "cmd_config.json")
        sources_found = []
        if os.path.exists(cmd_config_path):
            try:
                with open(cmd_config_path, encoding="utf-8-sig") as f:
                    cmd_cfg = json.load(f)
                for s in cmd_cfg.get("provider_sources", []):
                    sid = s.get("id", "?")
                    stype = s.get("type", "?")
                    base = s.get("api_base", "https://api.openai.com/v1")
                    keys = s.get("key", [])
                    has_key = "Y" if (keys and keys[0]) else "N"
                    sources_found.append((sid, stype, base, has_key))
            except Exception as e:
                print(f"  [!] 读取提供商信息失败: {e}")
                pass

        if sources_found:
            print(f"\n  [i] 检测到 AstrBot 模型提供商 ({len(sources_found)} 个):")
            for sid, stype, base, has_key in sources_found:
                print(f"      {sid} ({stype}, {base}, API Key: {has_key})")

    print(f"\n安装完成: {plugin_dir}")
    print("重启 AstrBot 即可加载 RCMS 插件。")
    print("如需修改设置，请在 AstrBot 后台 -> 插件管理 -> RCMS 中配置。")
    if not forward_api:
        print("提示: 使用 --forward-api 可在安装时自动导入 AstrBot 的 API 配置。")
    print("API 配置说明:")
    print("  source=astrbot — 外部读取 AstrBot 已配置的提供商（默认）")
    print("  source=custom  — 在 analysis 中手动填写 url / token / model")


def main():
    parser = argparse.ArgumentParser(description="安装 RCMS 到 AstrBot")
    parser.add_argument(
        "--plugin-dir",
        help="AstrBot plugins 目录（默认自动查找）",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="覆盖安装，不提示确认",
    )
    parser.add_argument(
        "--forward-api",
        action="store_true",
        help="导入 AstrBot 的 API 配置（url / token / model）到 RCMS 的 analysis 段",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="先 git pull 拉取最新代码，再安装（自动 --force）",
    )
    args = parser.parse_args()

    if args.pull:
        print("拉取最新代码...")
        result = subprocess.run(["git", "pull"], cwd=_HERE, capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"git pull 失败: {result.stderr.strip()}")
            sys.exit(1)
        args.force = True

    target = args.plugin_dir or find_astrbot_plugin_dir()
    if not target:
        print("未找到 AstrBot 插件目录，请通过 --plugin-dir 指定")
        sys.exit(1)

    print(f"安装到: {target}")
    install(target, force=args.force, forward_api=args.forward_api)


if __name__ == "__main__":
    main()
