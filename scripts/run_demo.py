"""
RCMS 真实对话演示

用法:
    # AstrBot 后端（自动读取配置）
    python run_demo.py --backend astrbot

    # 直接 OpenAI 兼容 API
    python run_demo.py --backend openai --api-key sk-xxx --base-url https://api.example.com/v1 --model gpt-4o
"""
import asyncio
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from minimal_rcms import MinimalRCMS


async def interactive_loop(rcms, backend, user_id, session_id):
    """交互式对话"""
    print("\n" + "="*60)
    print("RCMS 对话开始（输入 /quit 退出，输入 /status 查看状态）")
    print("="*60)

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/status":
            state = rcms.conn.execute(
                "SELECT * FROM session_state WHERE session_id = ?", (session_id,)
            ).fetchone()
            print(f"[状态] turn={state[5]}, stance={state[2]}, mood={state[3]}")
            cnt = rcms.conn.execute(
                "SELECT COUNT(*) FROM long_term_memory"
            ).fetchone()[0]
            print(f"[记忆] 长期记忆 {cnt} 条")
            continue

        reply = await rcms.chat(user_id, session_id, user_input, backend)
        print(f"RCMS: {reply}")


async def auto_demo(rcms, backend, user_id, session_id):
    """自动对话序列"""
    inputs = [
        "你好",
        "今天天气不错",
        "最近好累，工作压力太大了，天天加班到半夜",
        "你觉得我该怎么办",
        "其实以前我也经历过类似的低谷期",
        "吃了吗",
        "我在想要不要换个工作",
    ]
    print("\n" + "="*60)
    print("自动对话演示 (7 轮)")
    print("="*60)
    for msg in inputs:
        print(f"\n你: {msg}")
        reply = await rcms.chat(user_id, session_id, msg, backend)
        print(f"RCMS: {reply}")

    state = rcms.conn.execute(
        "SELECT turn_count, stance FROM session_state WHERE session_id = ?",
        (session_id,)
    ).fetchone()
    mem_cnt = rcms.conn.execute(
        "SELECT COUNT(*) FROM long_term_memory"
    ).fetchone()[0]
    print(f"\n--- 统计 ---")
    print(f"总轮数: {state[0]}")
    print(f"最后 stance: {state[1]}")
    print(f"长期记忆: {mem_cnt} 条")


async def main():
    parser = argparse.ArgumentParser(description="RCMS 对话演示")
    parser.add_argument("--backend", choices=["mock", "astrbot", "openai"], default="astrbot")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--model", default="gpt-4o", help="模型名称")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--db", default=os.path.join(_ROOT, "data", "demo_memory.db"), help="数据库路径")
    args = parser.parse_args()

    # 初始化 RCMS
    rcms = MinimalRCMS(db_path=args.db)

    # 创建后端
    if args.backend == "mock":
        from backends import MockBackend
        backend = MockBackend()
    elif args.backend == "astrbot":
        from backends.astrbot_backend import AstrBotBackend
        backend = AstrBotBackend()
        print(f"模型: {backend.get_model_name()}")
    elif args.backend == "openai":
        if not args.api_key:
            print("错误: openai 后端需要 --api-key")
            return
        from backends.openai_backend import OpenAIBackend
        backend = OpenAIBackend(api_key=args.api_key, base_url=args.base_url, model=args.model)

    try:
        if args.interactive:
            await interactive_loop(rcms, backend, "demo_user", "demo_session")
        else:
            await auto_demo(rcms, backend, "demo_user", "demo_session")
    finally:
        await backend.close()
        rcms.close()


if __name__ == "__main__":
    asyncio.run(main())
