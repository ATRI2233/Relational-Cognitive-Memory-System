"""
data_integrity_check.py — RCMS 数据库完整性检查脚本

逐表检查以下项目：
  1. 行数概览
  2. 主键是否存在（INTEGER / TEXT PRIMARY KEY）
  3. NOT NULL 约束列是否有 NULL 值
  4. UNIQUE 索引是否有重复
  5. 隐式外键关系的 orphan 记录
  6. user_mappings.label 完整性
  7. session_state.turn_count 与实际 chat_history 行数的一致性

依赖：仅标准库（sqlite3, sys, os, glob, argparse）

用法：
    python scripts/data_integrity_check.py --db-path data/rcms.db
    python scripts/data_integrity_check.py              # 自动搜索 *.db
"""

import argparse
import re
import sqlite3
import sys
import os
import glob


_SAFE_IDENTIFIER = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _safe_quote(name: str) -> str:
    """Quote a SQL identifier, raising ValueError if it contains unsafe characters."""
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


# ======================================================================
# 隐式外键关系对
# 从应用逻辑推断的外键关系（SQLite 未显式声明 FOREIGN KEY）。
# 每个元素: (源表, 源列, 目标表, 目标列, 可读描述)
# ======================================================================
FOREIGN_KEY_PAIRS = [
    ("cognitive_distill", "session_id", "session_state", "session_id",
     "cognitive_distill.session_id → session_state.session_id"),
    ("chat_history", "session_id", "session_state", "session_id",
     "chat_history.session_id → session_state.session_id"),
    ("memory_graph_edges", "from_node_id", "memory_graph_nodes", "node_id",
     "memory_graph_edges.from_node_id → memory_graph_nodes.node_id"),
    ("memory_graph_edges", "to_node_id", "memory_graph_nodes", "node_id",
     "memory_graph_edges.to_node_id → memory_graph_nodes.node_id"),
    ("embedding_rebuild_queue", "record_id", "cognitive_distill", "id",
     "embedding_rebuild_queue.record_id → cognitive_distill.id"),
    ("memory_links", "from_memory_id", "cognitive_distill", "id",
     "memory_links.from_memory_id → cognitive_distill.id"),
    ("memory_links", "to_memory_id", "cognitive_distill", "id",
     "memory_links.to_memory_id → cognitive_distill.id"),
]


def _get_user_tables(conn):
    """获取数据库中所有用户定义的表名，排除 sqlite_sequence 等内部表"""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows if r[0] != "sqlite_sequence"]


def _table_exists(conn, table):
    """检查表是否存在"""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ======================================================================
# 1. 行数检查
# ======================================================================
def check_row_count(conn, table):
    """返回表中的行数"""
    return conn.execute(f'SELECT COUNT(*) FROM {_safe_quote(table)}').fetchone()[0]


# ======================================================================
# 2. 主键存在性检查
# ======================================================================
def check_primary_key(conn, table):
    """
    检查表是否有主键，返回主键列信息。

    通过 PRAGMA table_info 读取每列的 pk 标记（pk > 0 表示是主键的一部分）。
    INTEGER PRIMARY KEY 在 SQLite 中自带自增行为。
    """
    cols = conn.execute(f'PRAGMA table_info({_safe_quote(table)})').fetchall()
    pk_cols = []
    for col in cols:
        cid, name, ctype, notnull, dflt, pk = col
        if pk > 0:
            pk_cols.append((name, ctype, pk))
    if not pk_cols:
        return None  # 无主键
    return pk_cols


# ======================================================================
# 3. NOT NULL 约束检查
# ======================================================================
def check_not_null_violations(conn, table):
    """
    检查表中标记了 NOT NULL 的列是否存在 NULL 值。

    从 PRAGMA table_info 获取列的 notnull 标记（1=NOT NULL），
    然后逐列统计 NULL 行数。
    """
    cols = conn.execute(f'PRAGMA table_info({_safe_quote(table)})').fetchall()
    violations = []
    for col in cols:
        cid, name, ctype, notnull, dflt, pk = col
        if not notnull:
            continue
        count = conn.execute(
            f'SELECT COUNT(*) FROM {_safe_quote(table)} WHERE {_safe_quote(name)} IS NULL'
        ).fetchone()[0]
        if count > 0:
            violations.append((name, count))
    return violations


# ======================================================================
# 4. UNIQUE 约束重复检查
# ======================================================================
def check_unique_violations(conn, table):
    """
    检查表中所有 UNIQUE 索引是否存在重复值。

    通过 PRAGMA index_list 获取所有索引，筛选 unique=1 的索引，
    再用 PRAGMA index_info 获取索引包含的列，
    最后 GROUP BY 统计重复行数。

    SQLite 中 UNIQUE 索引允许多个 NULL 值被视为不重复，
    因此 WHERE 条件排除 NULL 行。
    """
    indexes = conn.execute(f'PRAGMA index_list({_safe_quote(table)})').fetchall()
    violations = []
    for idx in indexes:
        seq, idx_name, unique, origin, partial = idx
        if not unique:
            continue
        # 获取索引包含的列（按 seqno 排序）
        cols = conn.execute(f'PRAGMA index_info({_safe_quote(idx_name)})').fetchall()
        col_names = [c[2] for c in sorted(cols, key=lambda x: x[1])]
        if not col_names:
            continue

        quoted = [_safe_quote(c) for c in col_names]
        # 排除 NULL 行（SQLite UNIQUE 允许多个 NULL 共存）
        not_null_conds = [f'{_safe_quote(c)} IS NOT NULL' for c in col_names]
        where_clause = " AND ".join(not_null_conds)

        group_expr = ", ".join(quoted)
        dups = conn.execute(
            f'SELECT {group_expr}, COUNT(*) AS cnt '
            f'FROM {_safe_quote(table)} WHERE {where_clause} '
            f'GROUP BY {group_expr} HAVING COUNT(*) > 1'
        ).fetchall()
        if dups:
            total_extra = sum(row[-1] - 1 for row in dups)
            violations.append((col_names, total_extra, idx_name))
    return violations


# ======================================================================
# 5. 外键孤记录检查
# ======================================================================
def check_orphans(conn, src_table, src_col, dst_table, dst_col):
    """
    检查 src_table 中 src_col 指向 dst_table.dst_col 时，
    是否存在目标不存在的孤记录。

    使用 LEFT JOIN 找出源列非空但目标列为空的行数。
    """
    return conn.execute(
        f'SELECT COUNT(*) FROM {_safe_quote(src_table)} s '
        f'LEFT JOIN {_safe_quote(dst_table)} d ON s.{_safe_quote(src_col)} = d.{_safe_quote(dst_col)} '
        f'WHERE s.{_safe_quote(src_col)} IS NOT NULL AND d.{_safe_quote(dst_col)} IS NULL'
    ).fetchone()[0]


# ======================================================================
# 6. user_mappings label 完整性检查
# ======================================================================
def check_label_integrity(conn):
    """
    检查 user_mappings.label 列的完整性：
      - 是否为 NULL
      - 是否为空字符串
      - 是否仅包含空白字符
    """
    if not _table_exists(conn, "user_mappings"):
        return {"exists": False}

    null_count = conn.execute(
        "SELECT COUNT(*) FROM user_mappings WHERE label IS NULL"
    ).fetchone()[0]

    empty_count = conn.execute(
        "SELECT COUNT(*) FROM user_mappings WHERE label = ''"
    ).fetchone()[0]

    # 非空字符串但全是空白字符
    blank_count = conn.execute(
        "SELECT COUNT(*) FROM user_mappings WHERE label != '' AND trim(label) = ''"
    ).fetchone()[0]

    return {
        "exists": True,
        "null_count": null_count,
        "empty_count": empty_count,
        "blank_count": blank_count,
    }


# ======================================================================
# 7. turn_count 一致性检查
# ======================================================================
def check_turn_count_consistency(conn):
    """
    检查 session_state.turn_count 是否与 chat_history 中
    对应 session_id 的实际行数一致。

    若不匹配，返回不一致的 session 列表。
    """
    if not _table_exists(conn, "session_state") or \
       not _table_exists(conn, "chat_history"):
        return None

    mismatches = conn.execute("""
        SELECT ss.session_id, ss.turn_count, COUNT(ch.id) AS actual_count
        FROM session_state ss
        LEFT JOIN chat_history ch ON ss.session_id = ch.session_id
        GROUP BY ss.session_id
        HAVING ss.turn_count != COUNT(ch.id)
    """).fetchall()

    return mismatches


# ======================================================================
# 运行所有检查
# ======================================================================
def run_checks(conn):
    """
    串联所有检查项，返回结构化的结果列表。
    每条结果包含: section(检查类别), table(表名/标签),
    check(检查名), status(PASS/FAIL/SKIP/INFO), detail(详情)
    """
    results = []
    tables = _get_user_tables(conn)

    # --- 1. 行数概览 ---
    for t in tables:
        count = check_row_count(conn, t)
        results.append({
            "section": "table_overview",
            "table": t,
            "check": "行数",
            "status": "INFO",
            "detail": f"{count} 行",
        })

    # --- 2. 主键存在性检查 ---
    for t in tables:
        pk_info = check_primary_key(conn, t)
        if pk_info is None:
            results.append({
                "section": "primary_key",
                "table": t,
                "check": "主键",
                "status": "FAIL",
                "detail": "无主键定义",
            })
        else:
            parts = [f"{name}({ctype})" for name, ctype, _ in pk_info]
            results.append({
                "section": "primary_key",
                "table": t,
                "check": "主键",
                "status": "PASS",
                "detail": f"主键列: {', '.join(parts)}",
            })

    # --- 3. NOT NULL 约束检查 ---
    for t in tables:
        violations = check_not_null_violations(conn, t)
        if not violations:
            results.append({
                "section": "not_null",
                "table": t,
                "check": "NOT NULL",
                "status": "PASS",
                "detail": "无 NULL 违规",
            })
        else:
            for col_name, cnt in violations:
                results.append({
                    "section": "not_null",
                    "table": t,
                    "check": f"NOT NULL({col_name})",
                    "status": "FAIL",
                    "detail": f"{cnt} 个 NULL 值",
                })

    # --- 4. UNIQUE 约束检查 ---
    for t in tables:
        violations = check_unique_violations(conn, t)
        if not violations:
            results.append({
                "section": "unique",
                "table": t,
                "check": "UNIQUE",
                "status": "PASS",
                "detail": "无重复",
            })
        else:
            for col_names, dup_cnt, idx_name in violations:
                col_str = ", ".join(col_names)
                results.append({
                    "section": "unique",
                    "table": t,
                    "check": f"UNIQUE({col_str})",
                    "status": "FAIL",
                    "detail": f"{dup_cnt} 条重复记录 (索引: {idx_name})",
                })

    # --- 5. 外键孤记录检查 ---
    for src_table, src_col, dst_table, dst_col, desc in FOREIGN_KEY_PAIRS:
        if src_table not in tables:
            results.append({
                "section": "foreign_key",
                "table": desc,
                "check": "外键孤记录",
                "status": "SKIP",
                "detail": f"源表 {src_table} 不存在",
            })
            continue
        if dst_table not in tables:
            results.append({
                "section": "foreign_key",
                "table": desc,
                "check": "外键孤记录",
                "status": "SKIP",
                "detail": f"目标表 {dst_table} 不存在",
            })
            continue
        orphan_count = check_orphans(conn, src_table, src_col, dst_table, dst_col)
        if orphan_count == 0:
            results.append({
                "section": "foreign_key",
                "table": desc,
                "check": "外键孤记录",
                "status": "PASS",
                "detail": "无孤记录",
            })
        else:
            results.append({
                "section": "foreign_key",
                "table": desc,
                "check": "外键孤记录",
                "status": "FAIL",
                "detail": f"{orphan_count} 条孤记录（目标不存在）",
            })

    # --- 6. label 完整性检查 ---
    label_result = check_label_integrity(conn)
    if not label_result["exists"]:
        results.append({
            "section": "label_integrity",
            "table": "user_mappings",
            "check": "label 完整性",
            "status": "SKIP",
            "detail": "user_mappings 表不存在",
        })
    else:
        total_issues = (
            label_result["null_count"]
            + label_result["empty_count"]
            + label_result["blank_count"]
        )
        if total_issues == 0:
            results.append({
                "section": "label_integrity",
                "table": "user_mappings",
                "check": "label 完整性",
                "status": "PASS",
                "detail": "所有 label 有效",
            })
        else:
            issue_parts = []
            if label_result["null_count"]:
                issue_parts.append(f"NULL={label_result['null_count']}")
            if label_result["empty_count"]:
                issue_parts.append(f"空字符串={label_result['empty_count']}")
            if label_result["blank_count"]:
                issue_parts.append(f"仅空白={label_result['blank_count']}")
            results.append({
                "section": "label_integrity",
                "table": "user_mappings",
                "check": "label 完整性",
                "status": "FAIL",
                "detail": " | ".join(issue_parts),
            })

    # --- 7. turn_count 一致性检查 ---
    mismatches = check_turn_count_consistency(conn)
    if mismatches is None:
        results.append({
            "section": "consistency",
            "table": "session_state",
            "check": "turn_count 一致性",
            "status": "SKIP",
            "detail": "session_state 或 chat_history 表不存在",
        })
    elif not mismatches:
        results.append({
            "section": "consistency",
            "table": "session_state",
            "check": "turn_count 一致性",
            "status": "PASS",
            "detail": "所有 session 的 turn_count 与 chat_history 条数一致",
        })
    else:
        detail_parts = []
        for r in mismatches:
            sid, recorded, actual = r
            detail_parts.append(f"session {sid}: turn_count={recorded}, 实际={actual}")
        results.append({
            "section": "consistency",
            "table": "session_state",
            "check": "turn_count 一致性",
            "status": "FAIL",
            "detail": "; ".join(detail_parts),
        })

    return results


# ======================================================================
# 结果打印
# ======================================================================
def print_results(results):
    """
    将检查结果格式化为表格输出。

    分组按 section 顺序打印，每行标记 [PASS]/[FAIL]/[SKIP]/[INFO]，
    末尾汇总 PASS/FAIL/SKIP 数量。
    """
    # 定义分组及打印顺序
    sections = [
        ("table_overview", "=== 表结构概览 ==="),
        ("primary_key", "\n=== 主键检查 ==="),
        ("not_null", "\n=== NOT NULL 约束检查 ==="),
        ("unique", "\n=== UNIQUE 约束检查 ==="),
        ("foreign_key", "\n=== 外键孤记录检查 ==="),
        ("label_integrity", "\n=== user_mappings.label 完整性检查 ==="),
        ("consistency", "\n=== turn_count 一致性检查 ==="),
    ]

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for section_key, section_title in sections:
        group = [r for r in results if r["section"] == section_key]
        if not group:
            continue
        print(section_title)
        for r in group:
            status_padded = f"[{r['status']}]".ljust(7)
            print(f"  {status_padded} {r['check']:35s} {r['detail']}")
            if r["status"] == "PASS":
                pass_count += 1
            elif r["status"] == "FAIL":
                fail_count += 1
            elif r["status"] == "SKIP":
                skip_count += 1

    # 汇总
    total = pass_count + fail_count + skip_count
    print("\n" + "=" * 70)
    print(f"总计: {pass_count} PASS, {fail_count} FAIL, {skip_count} SKIP  (共 {total} 项)")
    print("exit code: 0（FAIL 仅报告，非断言）")


# ======================================================================
# 入口
# ======================================================================
def main():
    """
    命令行入口：
      1. 解析参数（--db-path）
      2. 未指定时自动搜索当前目录下 *.db
      3. 连接数据库并运行全部检查
      4. 打印结果，以 exit code 0 退出
    """
    parser = argparse.ArgumentParser(
        description="RCMS 数据库完整性检查脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/data_integrity_check.py --db-path data/rcms.db
  python scripts/data_integrity_check.py
        """,
    )
    parser.add_argument(
        "--db-path",
        help="SQLite 数据库文件路径；若不指定则自动搜索当前目录下的 *.db 文件",
    )
    args = parser.parse_args()

    # 确定数据库路径
    if args.db_path:
        db_path = args.db_path
    else:
        # 搜索当前工作目录下所有 .db 文件
        db_files = glob.glob(os.path.join(os.getcwd(), "*.db"))
        if not db_files:
            print("错误: 未指定 --db-path 且当前目录下未找到 *.db 文件")
            sys.exit(1)
        if len(db_files) > 1:
            print("错误: 找到多个 .db 文件，请使用 --db-path 指定其中一个:")
            for f in db_files:
                print(f"  {f}")
            sys.exit(1)
        db_path = db_files[0]
        print(f"自动检测到数据库: {db_path}")

    # 验证文件存在
    if not os.path.isfile(db_path):
        print(f"错误: 文件不存在: {db_path}")
        sys.exit(1)

    print(f"数据库: {db_path}\n")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        results = run_checks(conn)
        print_results(results)
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()

    # FAIL 仅报告非断言，始终返回 0
    sys.exit(0)


if __name__ == "__main__":
    main()
