#!/usr/bin/env python3
"""
RCMS 数据分布分析脚本
========================
读取 SQLite 数据库，对每张表输出行数、空值分布、distinct 值、
时间跨度、数值统计，以及 cognitive_distill / session_state 的专项分析。

用法:
    python scripts/analyze_data_distribution.py
    python scripts/analyze_data_distribution.py --db-path data/rcms.db
    python scripts/analyze_data_distribution.py --output report.txt
"""

import argparse
import os
import sqlite3
import sys
from collections import OrderedDict
from datetime import datetime

# ============================================================
# 表名列名映射（中文输出）
# ============================================================
TABLE_CN = {
    "cognitive_distill": "认知蒸馏 (cognitive_distill)",
    "session_state": "会话状态 (session_state)",
    "chat_history": "聊天历史 (chat_history)",
    "identity_memory": "身份记忆 (identity_memory)",
    "memory_graph_nodes": "图节点 (memory_graph_nodes)",
    "memory_graph_edges": "图边 (memory_graph_edges)",
    "shared_context": "共享上下文 (shared_context)",
    "user_mappings": "用户映射 (user_mappings)",
    "analysis_raw": "原始分析 (analysis_raw)",
    "embedding_rebuild_queue": "嵌入重建队列 (embedding_rebuild_queue)",
}

# 中文列名映射
COLUMN_CN = {
    "id": "ID",
    "user_id": "用户ID",
    "session_id": "会话ID",
    "content": "内容",
    "keylabel": "关键词标签",
    "summary": "摘要",
    "mood": "情绪标签",
    "mood_intensity": "情绪强度",
    "importance": "重要性",
    "entities": "实体列表",
    "embedding": "嵌入向量",
    "embedding_dim": "嵌入维度",
    "turn_num": "轮次编号",
    "created_at": "创建时间",
    "expires_at": "过期时间",
    "stance": "立场",
    "turn_count": "对话轮次计数",
    "stance_turns": "立场持续轮次",
    "engagement_level": "参与度等级",
    "momentum_depth": "动量深度",
    "momentum_energy": "动量能量",
    "last_active": "最后活跃时间",
    "dangling_threads": "悬挂话题",
    "embedding_updated": "嵌入已更新",
    "last_distill_turn": "最后蒸馏轮次",
    "last_distill_at": "最后蒸馏时间",
    "role": "角色",
    "sender_name": "发送者名称",
    "traits": "特质",
    "preferences": "偏好",
    "self_identity": "自我认同",
    "boundaries": "边界",
    "updated_at": "更新时间",
    "node_id": "节点ID",
    "label": "标签",
    "node_type": "节点类型",
    "freq": "频率",
    "last_seen": "最后出现时间",
    "entity_type": "实体类型",
    "from_node_id": "起始节点ID",
    "to_node_id": "目标节点ID",
    "weight": "权重",
    "encounter_count": "相遇次数",
    "relation": "关系",
    "context_id": "上下文ID",
    "context_body": "上下文内容",
    "omission_count": "遗漏次数",
    "confirmed": "已确认",
    "source": "来源",
    "parsed": "已解析",
    "record_id": "记录ID",
    "reason": "原因",
}

# 时间列名集合（自动识别）
TIME_COLUMN_NAMES = {
    "created_at", "updated_at", "last_active", "last_seen",
    "last_distill_at", "expires_at",
}

# 数值列名集合（自动识别，整数和浮点）
NUMERIC_COLUMN_NAMES = {
    "id", "turn_count", "stance_turns", "momentum_depth",
    "momentum_energy", "embedding_updated", "last_distill_turn",
    "importance", "mood_intensity", "freq", "weight",
    "encounter_count", "omission_count", "confirmed", "turn_num",
    "embedding_dim", "parsed", "record_id", "node_id", "context_id",
}


def get_column_type(conn, table, col_name):
    """通过 PRAGMA table_info 获取列的声明类型"""
    rows = conn.execute(f"PRAGMA table_info(`{table}`)").fetchall()
    for row in rows:
        # row: (cid, name, type, notnull, dflt_value, pk)
        if row[1] == col_name:
            return row[2].upper() if row[2] else ""
    return ""


def is_time_column(col_name, col_type):
    """判断是否为时间列"""
    if col_name.lower() in TIME_COLUMN_NAMES:
        return True
    # TIMESTAMP 类型也视为时间列
    return "TIMESTAMP" in col_type.upper()


def is_numeric_column(col_name, col_type):
    """判断是否为数值列"""
    if col_name.lower() in NUMERIC_COLUMN_NAMES:
        return True
    ut = col_type.upper()
    if ut in ("INTEGER", "INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC"):
        return True
    # BLOB/PK 列跳过
    if "BLOB" in ut:
        return False
    return False


def is_text_column(col_name, col_type):
    """判断是否为文本/分类列（可用于 DISTINCT 分析）"""
    name_lower = col_name.lower()
    if name_lower in ("embedding",):
        return False
    ut = col_type.upper()
    if ut in ("TEXT", "VARCHAR", "CHAR", "CLOB"):
        return True
    if ut in ("INTEGER", "INT", "REAL", "FLOAT", "NUMERIC", "BLOB") or not ut:
        return False
    return True


def fmt(val):
    """格式化输出值，None → 'NULL'"""
    if val is None:
        return "NULL"
    return str(val)


def fmt_pct(ratio):
    """格式化百分比"""
    return f"{ratio * 100:.1f}%"


def fmt_span(seconds):
    """格式化时间跨度"""
    if seconds is None:
        return "N/A"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    if days > 0:
        return f"{days}天 {hours}小时"
    if hours > 0:
        mins = int((seconds % 3600) // 60)
        return f"{hours}小时 {mins}分"
    mins = int(seconds // 60)
    if mins > 0:
        return f"{mins}分"
    return f"{seconds:.0f}秒"


def safe_len(val):
    """安全获取字符串长度"""
    if val is None:
        return 0
    return len(str(val))


# ============================================================
# 分析函数
# ============================================================

def analyze_table(conn, table, lines):
    """对单张表执行通用分析，结果追加到 lines"""
    cn_table = TABLE_CN.get(table, table)
    lines.append(f"")
    lines.append(f"{'=' * 70}")
    lines.append(f"  表: {cn_table}")
    lines.append(f"{'=' * 70}")

    # --- 总行数 ---
    row = conn.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()
    total = row[0] if row else 0
    lines.append(f"  总行数: {total}")
    lines.append("")

    if total == 0:
        lines.append("  (空表)")
        return

    # --- 获取列信息 ---
    col_info = conn.execute(f"PRAGMA table_info(`{table}`)").fetchall()
    columns = []
    for c in col_info:
        cid, name, ctype, notnull, dflt, pk = c
        columns.append({
            "name": name,
            "type": ctype.upper() if ctype else "",
            "notnull": bool(notnull),
            "pk": bool(pk),
        })

    # --- 逐列统计 ---
    col_stats = []

    for col in columns:
        name = col["name"]
        ctype = col["type"]
        cn = COLUMN_CN.get(name, name)

        # NULL 计数
        null_count = conn.execute(
            f"SELECT COUNT(*) FROM `{table}` WHERE `{name}` IS NULL"
        ).fetchone()[0]
        null_ratio = null_count / total if total > 0 else 0.0

        stat = OrderedDict()
        stat["列名"] = cn
        stat["原始列名"] = name
        stat["类型"] = ctype if ctype else "TEXT"
        stat["非空约束"] = "是" if col["notnull"] else ""
        stat["主键"] = "是" if col["pk"] else ""
        stat["NULL数"] = null_count
        stat["NULL比例"] = fmt_pct(null_ratio)

        # 时间列分析
        if is_time_column(name, ctype):
            try:
                t_min = conn.execute(
                    f"SELECT MIN(`{name}`) FROM `{table}` WHERE `{name}` IS NOT NULL"
                ).fetchone()[0]
                t_max = conn.execute(
                    f"SELECT MAX(`{name}`) FROM `{table}` WHERE `{name}` IS NOT NULL"
                ).fetchone()[0]
                stat["最小值"] = fmt(t_min)
                stat["最大值"] = fmt(t_max)
                if t_min and t_max:
                    try:
                        fmt_t = "%Y-%m-%d %H:%M:%S"
                        dt_min = datetime.strptime(t_min[:19], fmt_t) if len(str(t_min)) >= 19 else datetime.strptime(str(t_min), fmt_t)
                        dt_max = datetime.strptime(t_max[:19], fmt_t) if len(str(t_max)) >= 19 else datetime.strptime(str(t_max), fmt_t)
                        span = (dt_max - dt_min).total_seconds()
                        stat["时间跨度"] = fmt_span(span)
                    except (ValueError, TypeError):
                        stat["时间跨度"] = "N/A"
                else:
                    stat["时间跨度"] = "N/A"
            except Exception:
                stat["最小值"] = "ERR"
                stat["最大值"] = "ERR"
                stat["时间跨度"] = "ERR"

        # 数值列分析
        elif is_numeric_column(name, ctype) and name.lower() not in ("embedding_updated",):
            try:
                row_stats = conn.execute(
                    f"SELECT MIN(`{name}`), MAX(`{name}`), AVG(`{name}`) FROM `{table}` WHERE `{name}` IS NOT NULL"
                ).fetchone()
                if row_stats and row_stats[0] is not None:
                    stat["最小值"] = round(row_stats[0], 4)
                    stat["最大值"] = round(row_stats[1], 4)
                    stat["均值"] = round(row_stats[2], 4)
                else:
                    stat["最小值"] = "N/A"
                    stat["最大值"] = "N/A"
                    stat["均值"] = "N/A"
            except Exception:
                stat["最小值"] = "ERR"
                stat["最大值"] = "ERR"
                stat["均值"] = "ERR"

        # 文本列 DISTINCT
        if is_text_column(name, ctype):
            try:
                distinct_count = conn.execute(
                    f"SELECT COUNT(DISTINCT `{name}`) FROM `{table}`"
                ).fetchone()[0]
                stat["DISTINCT值"] = distinct_count
                if total > 0:
                    stat["唯一率"] = fmt_pct(distinct_count / total)
            except Exception:
                stat["DISTINCT值"] = "ERR"

        col_stats.append(stat)

    # --- 输出列统计表格 ---
    # 确定表头（取所有 stat 的并集）
    all_keys = OrderedDict()
    for s in col_stats:
        for k in s:
            all_keys[k] = True

    # 控制台友好的列宽
    headers = list(all_keys.keys())
    # 对于某些列做宽度限制
    col_widths = {}
    for h in headers:
        max_w = len(h)
        for s in col_stats:
            val = str(s.get(h, ""))
            max_w = max(max_w, safe_len(val))
        col_widths[h] = min(max_w + 2, 48)

    # 表头行
    header_line = "  "
    for h in headers:
        w = col_widths[h]
        header_line += h.ljust(w)
    lines.append(header_line)
    lines.append("  " + "-" * (sum(col_widths.values()) + 2))

    # 数据行
    for s in col_stats:
        line = "  "
        for h in headers:
            w = col_widths[h]
            val = str(s.get(h, ""))
            line += val.ljust(w)
        lines.append(line)

    lines.append("")


def analyze_cognitive_distill_extra(conn, lines):
    """cognitive_distill 专项分析"""
    table = "cognitive_distill"
    total = conn.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()[0]
    if total == 0:
        lines.append("  [cognitive_distill 为空表，跳过专项分析]")
        lines.append("")
        return

    lines.append(f"  --- cognitive_distill 专项分析 ---")
    lines.append("")

    # 1. mood 值分布（TOP 10）
    lines.append("  [情绪标签分布 TOP 10]")
    mood_rows = conn.execute(f"""
        SELECT COALESCE(NULLIF(mood, ''), '(空值)') AS mood_val,
               COUNT(*) AS cnt
        FROM cognitive_distill
        GROUP BY mood_val
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    if mood_rows:
        max_cnt = max(r[1] for r in mood_rows) if mood_rows else 0
        for mood_val, cnt in mood_rows:
            bar = "█" * int(cnt / max_cnt * 40) if max_cnt > 0 else ""
            lines.append(f"    {mood_val:<20s}  {cnt:>6d} ({cnt/total*100:5.1f}%)  {bar}")
    else:
        lines.append("    (无数据)")
    lines.append("")

    # 2. mood_intensity 分布
    lines.append("  [情绪强度分布]")
    mi_stats = conn.execute("""
        SELECT COUNT(*),
               ROUND(AVG(mood_intensity), 4),
               ROUND(MIN(mood_intensity), 4),
               ROUND(MAX(mood_intensity), 4)
        FROM cognitive_distill WHERE mood_intensity IS NOT NULL
    """).fetchone()
    if mi_stats and mi_stats[0] > 0:
        lines.append(f"    总记录: {mi_stats[0]}")
        lines.append(f"    均值:   {mi_stats[1]}")
        lines.append(f"    最小值: {mi_stats[2]}")
        lines.append(f"    最大值: {mi_stats[3]}")
        # 分段
        segments = [
            ("= 0.0", "mood_intensity = 0.0"),
            ("(0, 0.3]", "mood_intensity > 0.0 AND mood_intensity <= 0.3"),
            ("(0.3, 0.5]", "mood_intensity > 0.3 AND mood_intensity <= 0.5"),
            ("(0.5, 0.8]", "mood_intensity > 0.5 AND mood_intensity <= 0.8"),
            ("(0.8, 1.0]", "mood_intensity > 0.8 AND mood_intensity <= 1.0"),
            ("> 1.0", "mood_intensity > 1.0"),
        ]
        for seg_name, seg_cond in segments:
            seg_cnt = conn.execute(
                f"SELECT COUNT(*) FROM cognitive_distill WHERE {seg_cond}"
            ).fetchone()[0]
            if seg_cnt > 0:
                lines.append(f"    {seg_name:<15s}  {seg_cnt:>6d} ({seg_cnt/total*100:5.1f}%)")
    else:
        lines.append("    (无数据)")
    lines.append("")

    # 3. importance 分布分段
    lines.append("  [重要性分布分段]")
    imp_segments = [
        ("< 0.3", "importance < 0.3"),
        ("0.3 - 0.5", "importance >= 0.3 AND importance < 0.5"),
        ("0.5 - 0.8", "importance >= 0.5 AND importance < 0.8"),
        (">= 0.8", "importance >= 0.8"),
    ]
    for seg_name, seg_cond in imp_segments:
        seg_cnt = conn.execute(
            f"SELECT COUNT(*) FROM cognitive_distill WHERE {seg_cond}"
        ).fetchone()[0]
        bar = "█" * int(seg_cnt / max(total, 1) * 60)
        lines.append(f"    {seg_name:<15s}  {seg_cnt:>6d} ({seg_cnt/total*100:5.1f}%)  {bar}")

    imp_null = conn.execute(
        "SELECT COUNT(*) FROM cognitive_distill WHERE importance IS NULL"
    ).fetchone()[0]
    if imp_null > 0:
        lines.append(f"    {'NULL':<15s}  {imp_null:>6d} ({imp_null/total*100:5.1f}%)")
    lines.append("")

    # 4. embedding IS NULL 统计
    emb_null = conn.execute(
        "SELECT COUNT(*) FROM cognitive_distill WHERE embedding IS NULL"
    ).fetchone()[0]
    emb_not_null = total - emb_null
    lines.append("  [嵌入向量统计]")
    lines.append(f"    有嵌入向量:  {emb_not_null:>8d} ({emb_not_null/total*100:5.1f}%)")
    lines.append(f"    无嵌入向量:  {emb_null:>8d} ({emb_null/total*100:5.1f}%)")
    lines.append("")


def analyze_session_state_extra(conn, lines):
    """session_state 专项分析"""
    table = "session_state"
    total = conn.execute(f"SELECT COUNT(*) FROM `{table}`").fetchone()[0]
    if total == 0:
        lines.append("  [session_state 为空表，跳过专项分析]")
        lines.append("")
        return

    lines.append(f"  --- session_state 专项分析 ---")
    lines.append("")

    # 1. stance 值分布
    lines.append("  [立场分布]")
    stance_rows = conn.execute("""
        SELECT COALESCE(NULLIF(stance, ''), '(空值)') AS stance_val,
               COUNT(*) AS cnt
        FROM session_state
        GROUP BY stance_val
        ORDER BY cnt DESC
    """).fetchall()
    if stance_rows:
        max_s = max(r[1] for r in stance_rows) if stance_rows else 0
        for s_val, cnt in stance_rows:
            bar = "█" * int(cnt / max_s * 40) if max_s > 0 else ""
            lines.append(f"    {s_val:<20s}  {cnt:>4d} ({cnt/total*100:5.1f}%)  {bar}")
    else:
        lines.append("    (无数据)")
    lines.append("")

    # 2. engagement_level 分布
    lines.append("  [参与度等级分布]")
    eng_rows = conn.execute("""
        SELECT COALESCE(NULLIF(engagement_level, ''), '(空值)') AS eng_val,
               COUNT(*) AS cnt
        FROM session_state
        GROUP BY eng_val
        ORDER BY cnt DESC
    """).fetchall()
    if eng_rows:
        max_e = max(r[1] for r in eng_rows) if eng_rows else 0
        for e_val, cnt in eng_rows:
            bar = "█" * int(cnt / max_e * 40) if max_e > 0 else ""
            lines.append(f"    {e_val:<20s}  {cnt:>4d} ({cnt/total*100:5.1f}%)  {bar}")
    else:
        lines.append("    (无数据)")
    lines.append("")


def get_all_tables(conn):
    """获取数据库中所有用户表（排除 sqlite_* 系统表）"""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def find_db_files(root_dir):
    """递归查找 .db 文件"""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".db"):
                matches.append(os.path.join(dirpath, fn))
    return matches


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RCMS 数据分布分析脚本"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite 数据库路径。未指定时自动搜索当前目录下的 .db 文件。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径。未指定时仅输出到 stdout。",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # ---- 定位数据库文件 ----
    if args.db_path:
        db_path = args.db_path
        if not os.path.isfile(db_path):
            print(f"错误: 文件不存在 --db-path={db_path}", file=sys.stderr)
            sys.exit(1)
    else:
        db_files = find_db_files(os.path.abspath("."))
        if not db_files:
            print("错误: 未找到 .db 文件，请通过 --db-path 指定", file=sys.stderr)
            sys.exit(1)
        if len(db_files) > 1:
            print("找到多个 .db 文件，请通过 --db-path 指定其中一个:")
            for f in db_files:
                print(f"  {f}")
            sys.exit(1)
        db_path = db_files[0]

    # ---- 打开数据库 ----
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=OFF")  # 只读查询
    except sqlite3.Error as e:
        print(f"错误: 无法打开数据库 {db_path}: {e}", file=sys.stderr)
        sys.exit(1)

    lines = []
    lines.append(f"RCMS 数据分布分析报告")
    lines.append(f"数据库: {os.path.abspath(db_path)}")
    lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ---- 遍历所有表 ----
    tables = get_all_tables(conn)
    if not tables:
        lines.append("数据库中未找到任何表。")

    for table in tables:
        analyze_table(conn, table, lines)
        # 专项分析
        if table == "cognitive_distill":
            analyze_cognitive_distill_extra(conn, lines)
        elif table == "session_state":
            analyze_session_state_extra(conn, lines)

    conn.close()

    # ---- 输出 ----
    output = "\n".join(lines)
    sys.stdout.write(output)
    sys.stdout.write("\n")

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
                f.write("\n")
            print(f"\n报告已写入: {os.path.abspath(args.output)}", file=sys.stderr)
        except IOError as e:
            print(f"错误: 写入文件失败 {args.output}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
