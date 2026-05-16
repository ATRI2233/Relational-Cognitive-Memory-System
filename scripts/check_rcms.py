"""
RCMS 诊断脚本 — 检查安装、配置、数据库、日志

用法：
  python3 scripts/check_rcms.py                  # 全部检查
  python3 scripts/check_rcms.py --config         # 仅配置
  python3 scripts/check_rcms.py --db             # 仅数据库
  python3 scripts/check_rcms.py --log            # 仅输出日志
  python3 scripts/check_rcms.py --all-db         # 所有数据库内容
"""
import json
import os
import sqlite3
import sys
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)


def title(s):
    print(f"\n{'=' * 60}")
    print(f"  {s}")
    print(f"{'=' * 60}")


def check_config():
    title("配置检查 (config.json)")
    path = os.path.join(PROJECT, "config.json")
    if not os.path.exists(path):
        print("  ❌ config.json 不存在")
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"  ✅ config.json 已加载 ({len(json.dumps(cfg))} bytes)")

    # 逐段检查
    errors = []
    sections = {
        "general": ["enabled", "persona_separated", "user_id", "injection_method"],
        "memory": ["enable_auto_save", "max_memories_per_prompt"],
        "output_log": ["enabled", "max_size_mb", "path"],
        "debug": ["log_level"],
    }
    for sec, keys in sections.items():
        obj = cfg.get(sec, {})
        if not obj:
            errors.append(f"缺失段: {sec}")
            continue
        for k in keys:
            if k not in obj:
                errors.append(f"{sec} 缺失字段: {k}")
    print(f"  • general.enabled = {cfg.get('general',{}).get('enabled')}")
    print(f"  • general.injection_method = {cfg.get('general',{}).get('injection_method')}")

    # analysis 段
    analysis = cfg.get("analysis", {})
    if analysis:
        ret = analysis.get("retrieval", {})
        post = analysis.get("post_analysis", {})
        print(f"  • analysis.retrieval = enabled={ret.get('enabled')} model={ret.get('model')}")
        print(f"  • analysis.post_analysis = mode={post.get('mode')} sampling={post.get('sampling')}")
        # API key 安全检查
        ret_key = ret.get("api_key", "")
        post_key = post.get("api_key", "")
        if ret_key and len(ret_key) > 4:
            print(f"  • analysis.retrieval.api_key = {'****' + ret_key[-4:]}")
        if post_key and len(post_key) > 4:
            print(f"  • analysis.post_analysis.api_key = {'****' + post_key[-4:]}")
        if not ret_key and not post_key:
            print(f"  ⚠️  analysis 段未配置 api_key，将使用环境变量 OPENAI_API_KEY")
    else:
        print("  ⚠️  缺少 analysis 段（v2 新功能不可用）")

    if errors:
        for e in errors:
            print(f"  ⚠️  {e}")
    else:
        print(f"  ✅ 配置结构无缺失")

    # 环境变量
    env_key = os.environ.get("OPENAI_API_KEY", "")
    env_url = os.environ.get("OPENAI_BASE_URL", "")
    print(f"  • OPENAI_API_KEY = {'已设置' if env_key else '未设置'}")
    if env_url:
        print(f"  • OPENAI_BASE_URL = {env_url}")

    return cfg


def check_db():
    title("数据库检查")
    data_dir = os.path.join(PROJECT, "data")
    if not os.path.exists(data_dir):
        print(f"  ⚠️  data/ 目录不存在")
        return
    dbs = [f for f in os.listdir(data_dir) if f.endswith(".db")]
    if not dbs:
        print(f"  ⚠️  data/ 下无 .db 文件（尚未产生对话记录）")
        return
    for db_name in sorted(dbs):
        db_path = os.path.join(data_dir, db_name)
        size = os.path.getsize(db_path)
        print(f"\n  📁 {db_name} ({size / 1024:.1f} KB)")
        try:
            conn = sqlite3.connect(db_path)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            # 显示表结构
            for (tname,) in tables:
                cols = conn.execute(f"PRAGMA table_info({tname})").fetchall()
                col_str = ", ".join(f"{c[1]}:{c[2]}" for c in cols[:5])
                row_count = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                extra = "..." if len(cols) > 5 else ""
                print(f"    📋 {tname} ({row_count} 行) — {col_str}{extra}")

            # 检查新表数据
            emb_count = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
            ent_count = conn.execute("SELECT COUNT(*) FROM entity_relations").fetchone()[0]
            if emb_count:
                print(f"    🔤 memory_embeddings: {emb_count} 条向量")
            if ent_count:
                ents = conn.execute("SELECT entity_name, relation_type FROM entity_relations ORDER BY mention_count DESC LIMIT 5").fetchall()
                print(f"    👤 entity_relations: {ent_count} 条 — {[e[0] for e in ents]}")

            # 长期记忆统计
            ltm = conn.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()[0]
            chat = conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
            if ltm or chat:
                print(f"    💬 long_term_memory: {ltm} 条 | chat_history: {chat} 条")

            # 检查 identity 和 shared_context
            ident = conn.execute("SELECT traits FROM identity_memory LIMIT 1").fetchone()
            if ident and ident[0] and ident[0] != "[]":
                traits = json.loads(ident[0])
                print(f"    🧠 identity_memory.traits: {traits}")

            conn.close()
        except Exception as e:
            print(f"    ❌ 检查失败: {e}")


def check_log():
    title("输出日志检查")
    log_path = os.path.join(PROJECT, "plugins/rcms-astrbot", "rcms_output.jsonl")
    if not os.path.exists(log_path):
        # 检查项目根目录
        log_path = os.path.join(PROJECT, "rcms_output.jsonl")
    if not os.path.exists(log_path):
        print("  ⚠️  未找到输出日志文件")
        return

    size = os.path.getsize(log_path)
    print(f"  📄 {log_path} ({size / 1024:.1f} KB)")
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    # 显示最近 10 条
    show = lines[-10:] if len(lines) > 10 else lines
    print(f"\n  最近 {len(show)}/{len(lines)} 条:")
    for line in show:
        try:
            entry = json.loads(line.strip())
            ts = datetime.fromtimestamp(entry.get("t", 0)).strftime("%H:%M:%S")
            p = entry.get("p", "?")[:12]
            s = entry.get("s", "?")
            u = entry.get("u", "")[:40]
            r = entry.get("r", "")[:40]
            print(f"    [{ts}] [{p}] ({s}) u=\"{u}\" r=\"{r}\"")
        except Exception:
            print(f"    (解析失败) {line[:80]}")


def check_all_db():
    """显示所有数据库的全部内容（调试用）"""
    title("全量数据库内容")
    data_dir = os.path.join(PROJECT, "data")
    if not os.path.exists(data_dir):
        print("  data/ 不存在")
        return
    dbs = [f for f in os.listdir(data_dir) if f.endswith(".db")]
    if not dbs:
        print("  无数据库文件")
        return
    for db_name in sorted(dbs):
        db_path = os.path.join(data_dir, db_name)
        print(f"\n{'─' * 50}")
        print(f"  📁 {db_name}")
        print(f"{'─' * 50}")
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (tname,) in tables:
            rows = conn.execute(f"SELECT * FROM {tname} LIMIT 10").fetchall()
            if rows:
                cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tname})").fetchall()]
                print(f"\n  {tname} ({len(rows)} 行):")
                for row in rows:
                    parts = []
                    for i, val in enumerate(row):
                        if isinstance(val, bytes):
                            parts.append(f"{cols[i]}=<{len(val)} bytes>")
                        else:
                            s = str(val)[:50]
                            parts.append(f"{cols[i]}={s}")
                    print(f"    {', '.join(parts[:6])}")
            else:
                print(f"\n  {tname}: (空)")
        conn.close()


def main():
    args = set(sys.argv[1:]) if len(sys.argv) > 1 else {"--all"}

    if "--config" in args or "--all" in args:
        check_config()
    if "--db" in args or "--all" in args:
        check_db()
    if "--log" in args or "--all" in args:
        check_log()
    if "--all-db" in args:
        check_all_db()

    # 简短的安装检查总结
    if "--all" in args:
        title("安装检查总结")
        ok = True
        # 1. 核心文件存在
        core_path = os.path.join(PROJECT, "rcms_core.py")
        if os.path.exists(core_path):
            print("  ✅ rcms_core.py 存在")
        else:
            print("  ❌ rcms_core.py 缺失")
            ok = False

        # 2. 插件目录存在
        plug_path = os.path.join(PROJECT, "plugins/rcms-astrbot")
        if os.path.exists(plug_path):
            print("  ✅ plugins/rcms-astrbot/ 存在")
        else:
            print("  ❌ 插件目录缺失")
            ok = False

        # 3. 数据库目录存在
        data_path = os.path.join(PROJECT, "data")
        if os.path.exists(data_path):
            print("  ✅ data/ 目录存在")
        else:
            print("  ⚠️  data/ 目录不存在（首次运行会自动创建）")

        # 4. 依赖
        deps_ok = True
        for mod, name in [("numpy", "numpy"), ("openai", "openai")]:
            try:
                __import__(mod)
                print(f"  ✅ {name} 已安装")
            except ImportError:
                print(f"  ❌ {name} 未安装")
                deps_ok = False
                ok = False

        # 5. Plugin metacfg
        meta_path = os.path.join(plug_path, "metadata.yaml")
        schema_path = os.path.join(plug_path, "_conf_schema.json")
        if os.path.exists(meta_path):
            print(f"  ✅ metadata.yaml 存在")
        if os.path.exists(schema_path):
            print(f"  ✅ _conf_schema.json 存在")

        # 6. 环境变量
        if os.environ.get("OPENAI_API_KEY"):
            print("  ✅ OPENAI_API_KEY 已设置")
        else:
            print("  ⚠️  OPENAI_API_KEY 未设置（分析/embedding 需配置）")

        if ok:
            print("\n  🟢 安装检查通过")
        else:
            print(f"\n  🔴 存在问题需要修复")


if __name__ == "__main__":
    main()
