"""
安装 RCMS 到 AstrBot 插件目录

用法:
    python install_astrbot.py                    # 自动查找 ~/.astrbot
    python install_astrbot.py --plugin-dir D:/path/to/plugins  # 手动指定
"""
import argparse
import os
import shutil
import sys

# 源文件: RCMS 项目根目录
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PLUGIN = os.path.join(_HERE, "plugins", "rcms-astrbot")
_CORE_FILES = ["minimal_rcms.py"]
_BACKEND_DIR = os.path.join(_HERE, "backends")


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


def install(target_dir: str, force: bool = False):
    plugin_dir = os.path.join(target_dir, "rcms")

    # 已存在则先提示
    if os.path.exists(plugin_dir) and not force:
        print(f"发现已有目录: {plugin_dir}")
        ans = input("数据库会保留，覆盖其他文件? (y/N): ").strip().lower()
        if ans != "y":
            print("取消安装")
            return

    os.makedirs(plugin_dir, exist_ok=True)

    # 复制适配器 main.py + metadata.yaml
    for f in ["main.py", "metadata.yaml"]:
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

    # 复制 backends
    dst_backends = os.path.join(plugin_dir, "backends")
    if os.path.exists(_BACKEND_DIR):
        if os.path.exists(dst_backends):
            shutil.rmtree(dst_backends)
        shutil.copytree(_BACKEND_DIR, dst_backends, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"  [+] backends/")

    # 如果已有数据库，保留
    existing_db = os.path.join(plugin_dir, "rcms_memory.db")
    if os.path.exists(existing_db):
        print(f"  [=] 已有数据库 (已保留)")

    print(f"\n安装完成: {plugin_dir}")
    print("重启 AstrBot 即可加载 RCMS 插件。")


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
