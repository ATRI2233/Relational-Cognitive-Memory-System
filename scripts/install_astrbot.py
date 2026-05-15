"""
安装 RCMS 到 AstrBot 插件目录

自动设置按人格分离的记忆存储（不同人格使用独立数据库）。

用法:
    python install_astrbot.py                    # 自动查找 ~/.astrbot
    python install_astrbot.py --plugin-dir D:/path/to/plugins  # 手动指定
"""
import argparse
import json
import os
import shutil
import sys

# 源文件: RCMS 项目根目录
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PLUGIN = os.path.join(_HERE, "plugins", "rcms-astrbot")
_CORE_FILES = ["minimal_rcms.py"]
_BACKEND_DIR = os.path.join(_HERE, "backends")

PLUGIN_DIR_NAME = "astrbot_plugin_rcms"


def find_astrbot_plugin_dir() -> str | None:
    """常见的 AstrBot 插件目录位置"""
    candidates = [
        os.path.expanduser("~/.astrbot/data/plugins"),
        os.path.join(_HERE, "data", "plugins"),  # dev 模式
    ]
    for p in candidates:
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
        except Exception:
            continue
    return personas


def install(target_dir: str, force: bool = False):
    plugin_dir = os.path.join(target_dir, PLUGIN_DIR_NAME)

    # 已存在则先提示
    if os.path.exists(plugin_dir) and not force:
        print(f"发现已有目录: {plugin_dir}")
        ans = input("数据库会保留，覆盖其他文件? (y/N): ").strip().lower()
        if ans != "y":
            print("取消安装")
            return

    os.makedirs(plugin_dir, exist_ok=True)

    # 复制适配器文件
    for f in ["main.py", "metadata.yaml", "_conf_schema.json"]:
        src = os.path.join(_SRC_PLUGIN, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(plugin_dir, f))
            print(f"  [+] {f}")

    # 复制 RCMS 核心
    for f in _CORE_FILES:
        src = os.path.join(_HERE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(plugin_dir, f))
            print(f"  [+] {f}")

    # 复制项目配置
    config_src = os.path.join(_HERE, "config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(plugin_dir, "config.json"))
        print(f"  [+] config.json")

    # 复制 backends
    dst_backends = os.path.join(plugin_dir, "backends")
    if os.path.exists(_BACKEND_DIR):
        if os.path.exists(dst_backends):
            shutil.rmtree(dst_backends)
        shutil.copytree(_BACKEND_DIR, dst_backends, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"  [+] backends/")

    # 保留已有数据库（含人格分离后的多库）
    existing_dbs = [f for f in os.listdir(plugin_dir) if f.startswith("rcms_memory") and f.endswith(".db")]
    if existing_dbs:
        print(f"  [=] 已有 {len(existing_dbs)} 个数据库文件 (已保留)")

    # 扫描并提示人格信息
    astrbot_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    personas = scan_astrbot_personas(astrbot_root)
    if personas:
        print(f"  [i] 检测到 {len(personas)} 个人格: {', '.join(personas)}")
        print(f"  [i] RCMS 将按人格自动分离记忆存储")
    else:
        print(f"  [i] 未检测到已配置的人格，将使用默认记忆库")

    print(f"\n安装完成: {plugin_dir}")
    print("重启 AstrBot 即可加载 RCMS 插件。")
    print("如需修改设置，请在 AstrBot 后台 -> 插件管理 -> RCMS 中配置。")


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
    args = parser.parse_args()

    target = args.plugin_dir or find_astrbot_plugin_dir()
    if not target:
        print("未找到 AstrBot 插件目录，请通过 --plugin-dir 指定")
        sys.exit(1)

    print(f"安装到: {target}")
    install(target, force=args.force)


if __name__ == "__main__":
    main()
