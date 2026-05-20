"""
70 轮模拟 — 真实 AstrBot API，全链路验证
验证点: 日志 / DB / emb 向量 / 三通道融合 / 图谱 BFS / narrative_context
"""
import asyncio, io, json, logging, os, sys, tempfile, traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -- 1. 从 AstrBot cmd_config.json 读取 API 凭证 --
astrbot_cfg_path = os.path.expanduser("~/.astrbot/data/cmd_config.json")
with open(astrbot_cfg_path, encoding="utf-8-sig") as f:
    astrbot_cfg = json.load(f)
sources = {s["id"]: s for s in astrbot_cfg.get("provider_sources", [])}
providers = [p for p in astrbot_cfg.get("provider", []) if p.get("enable", False)]
default_id = astrbot_cfg.get("provider_settings", {}).get("default_provider_id", "")
target = next((p for p in providers if p["id"] == default_id), providers[0])
src_id = target.get("provider_source_id", "")
llm_model = target.get("model", "gpt-4o-mini")
src = sources.get(src_id)
llm_key = (src.get("key", [""])[0] if isinstance(src.get("key"), list) else src.get("key", "")) or None
llm_url = src.get("api_base", "https://api.openai.com/v1")
emb_providers = [p for p in providers if p.get("type") == "openai_embedding" or p.get("provider_type") == "embedding"]
if emb_providers:
    ep = emb_providers[0]
    emb_key = ep.get("embedding_api_key", "") or None
    emb_url = ep.get("embedding_api_base", "https://api.openai.com/v1")
    emb_model = ep.get("embedding_model", "text-embedding-3-small")
else:
    emb_key, emb_url, emb_model = llm_key, llm_url, "text-embedding-3-small"
logger.info(f"LLM: model={llm_model} url={llm_url} key_set={bool(llm_key)}")
logger.info(f"Emb: model={emb_model} url={emb_url} key_set={bool(emb_key)}")

from openai import AsyncOpenAI
from rcms_core import MinimalRCMS

_llm_client = AsyncOpenAI(api_key=llm_key, base_url=llm_url)
async def real_llm_call(prompt, model=""):
    m = model or llm_model
    resp = await _llm_client.chat.completions.create(
        model=m, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, max_tokens=2048)
    c = resp.choices[0].message.content or "{}"
    logger.info(f"REAL_LLM: model={m} tokens={resp.usage.total_tokens if resp.usage else '?'}")
    return c

_emb_client = AsyncOpenAI(api_key=emb_key, base_url=emb_url)
async def real_embed_call(text):
    resp = await _emb_client.embeddings.create(model=emb_model, input=text.replace("\n", " "))
    vec = resp.data[0].embedding
    logger.info(f"REAL_EMB: dim={len(vec)} front={[round(v,4) for v in vec[:3]]}")
    return vec

analysis_config = {
    "retrieval": {
        "embedding_enabled": True, "source": "custom",
        "custom_api_key": emb_key, "custom_base_url": emb_url, "custom_model": emb_model,
        "total_cap": 5, "channel_min": [1, 1, 1],
        "time_decay_halflife": 30, "emotional_resonance_bonus": 0.15,
    },
    "post_analysis": {
        "source": "custom",
        "custom_api_key": llm_key, "custom_base_url": llm_url, "custom_model": llm_model,
        "max_turns": 30, "max_minutes": 60, "dangling_expire_turns": 15,
    }
}

USER_INPUTS = [
    "今天好累啊，上了一天班",
    "最近项目压力有点大，经理又改需求了",
    "不过好在明天周五了，周末打算去爬山放松一下",
    "最近在学 Rust，感觉这个语言的所有权系统很有意思",
    "用 Rust 写出来的程序很稳，我打算用 Rust 写个小工具",
    "周末爬山计划取消了，下雨，只好在家看纪录片",
    "看了一部关于深海探索的纪录片，突然对海洋生物感兴趣了",
    "想养鱼，但怕养不好，以前养死过一条金鱼",
    "下周那个项目要截止了，得抓紧时间赶工",
    "今天加班到很晚，同事小王帮我调了个bug，他技术很厉害",
    "改天请小王吃个饭，最近又胖了，打算从明天开始跑步",
    "但每次跑步都坚持不下来，可能得找个跑友",
    "算了，先忙完这阵子再说，今天就到这吧，晚安",
    "早上好，昨晚睡得很好，今天打算把核心模块写完",
    "Rust 那个小工具有进展了，感觉渐入佳境",
    "中午吃了碗牛肉面，下午继续写代码",
    "小王说周末一起去看电影，好久没去电影院了",
    "最近在看美剧，剧情挺有意思，推荐",
    "周末电影看完了，一般般，不如预期",
    "这周项目终于上线了，松了一口气",
    "客户反馈还不错，没有重大bug",
    "打算学一下Go语言，听说并发编程很简洁",
    "Rust 的异步编程也挺有意思，tokio 生态越来越完善了",
    "周末准备去图书馆待一天，好久没安静看书了",
    "最近读《程序员修炼之道》，很经典",
    "书里讲了很多软件设计的哲学，值得反复读",
    "打算把书里的原则用到实际项目中",
    "小王也读过这本书，我们聊了很多",
    "其实有个想法，想做一个开源项目",
    "用 Rust 写一个轻量级的Web框架，应该挺有挑战的",
    "不过先把手头的事情忙完再说，不能好高骛远",
    "今天状态不错，写了500行代码，效率很高",
    "晚上做了个梦，梦见自己会飞，感觉很自由",
    "梦里的场景特别真实，醒来还回味了好久",
    "最近开始冥想，感觉对专注力有帮助",
    "每天冥想15分钟，坚持了一周了",
    "朋友推荐了一个冥想App，挺好用的",
    "感觉整个人平和了很多，没那么焦虑了",
    "工作上遇到了一个新挑战，要用机器学习",
    "虽然不太懂ML，但觉得可以学一下试试",
    "在网上找了个ML入门课程，开始啃了",
    "课程里讲的线性回归还挺好理解的",
    "打算用 Python 做几个小项目练手",
    "同时还在继续写 Rust 小工具，两边一起学",
    "小王也对ML感兴趣，我们可以组队学习",
    "周末约了小王一起看书，顺便聊聊技术",
    "最近睡眠质量不错，可能跟冥想有关系",
    "今天公司开了个技术分享会，学到了很多",
    "有个同事分享了K8s的经验，感觉很实用",
    "打算把现在的服务容器化，用 Docker 部署",
    "Docker 学起来比想象中简单，主要是概念要理解",
    "用 Docker Compose 编排了好几个服务，跑起来了",
    "下一步打算学一下 Kubernetes，虽然有点复杂",
    "周末去图书馆看了本关于分布式的书，挺有启发",
    "CAP 定理和一致性协议这些概念挺烧脑的",
    "慢慢消化吧，不着急一下子全搞懂",
    "Rust 小工具终于写完了，已经上传到 GitHub",
    "虽然功能很简单，但算是第一个完整的 Rust 项目",
    "发给了小王看，他给了很多改进建议",
    "打算根据建议重构一下代码结构",
    "重构完了，代码清晰了很多，Rust 的所有权系统确实好用",
    "ML 课程学到了神经网络，感觉打开了新世界",
    "虽然数学推导有点难，但直觉上理解了反向传播",
    "打算做个图像分类的小项目练手",
    "用 Rust 写了一个简单的图片处理库，结合 ML 用",
    "公司那边容器化也推进顺利，测试环境已经跑起来了",
    "这周事情好多但很充实，感觉自己在成长",
    "回顾这阵子，从 Rust 到 ML 到容器化，学了好多新东西",
    "最大的收获是学会了怎么同时推进多个学习项目",
    "继续保持这个节奏，打算年底前把 K8s 也拿下",
]

async def main():
    db_fd, db_path = tempfile.mkstemp(suffix=".db", prefix="rcms_live_")
    os.close(db_fd)
    logger.info(f"DB: {db_path}")
    rcms = MinimalRCMS(db_path=db_path, analysis_config=analysis_config,
                        llm_call=real_llm_call, embed_call=real_embed_call)
    UID, SID = "test_user", "test_session_001"

    print("=" * 80)
    print("70 轮对话 - 真实 API (deepseek-v4-flash + qwen3-embedding-8b)")
    print("=" * 80)
    print(f"{'t':>3} {'ch':>3} {'cd':>3} {'nd':>3} {'eg(s/c)':>9} {'sl':>3} {'emb(hi)':>9} {'ld':>3}")
    print("-" * 80)

    distill_turns = []
    for i, ui in enumerate(USER_INPUTS):
        turn = i + 1
        reply = f"你说的这个确实有意思，{ui[:20]}我理解了。"
        rcms.save_turn(SID, ui, reply, user_id=UID)
        await rcms.post_update_rules(UID, SID, ui, "open", reply)

        triggered, last_turn, tc, snap, senders = rcms.check_distill_needed(SID)
        if triggered:
            logger.info(f">>> DISTILL_TRIGGER: turn={tc} last={last_turn}")
            lt = rcms._load_long_term_context(UID)
            try:
                await rcms._run_distill_analysis(UID, SID, snap, lt, last_turn, tc, senders=senders)
                distill_turns.append(tc)
            except Exception as e:
                logger.error(f">>> DISTILL_FAILED: {e}")
                traceback.print_exc()

        c = rcms.conn
        cd = c.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]
        ch = c.execute("SELECT count(*) FROM chat_history").fetchone()[0]
        nd = c.execute("SELECT count(*) FROM memory_graph_nodes").fetchone()[0]
        eg = c.execute("SELECT count(*) FROM memory_graph_edges").fetchone()[0]
        sl = c.execute("SELECT count(*) FROM memory_graph_edges WHERE from_node_id=to_node_id").fetchone()[0]
        se = c.execute("SELECT count(*) FROM memory_graph_edges WHERE relation!=''").fetchone()[0]
        em = c.execute("SELECT count(*) FROM cognitive_distill WHERE embedding IS NOT NULL AND embedding != ''").fetchone()[0]
        eh = c.execute("SELECT count(*) FROM cognitive_distill WHERE importance>=0.5 AND embedding IS NOT NULL AND embedding != ''").fetchone()[0]
        ss = c.execute("SELECT last_distill_turn FROM session_state WHERE session_id=?", (SID,)).fetchone()
        ld = ss[0] if ss else 0
        print(f"  t={turn:2d} ch={ch:3d} cd={cd:2d} nd={nd:3d} eg={eg:3d}(s={se}/c={eg-se}) sl={sl} emb={em}(hi={eh}) ld={ld}")

    # ═══════════ 最终校验 ═══════════
    c = rcms.conn
    ch_t = c.execute("SELECT count(*) FROM chat_history").fetchone()[0]
    cd_t = c.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]
    cd_03 = c.execute("SELECT count(*) FROM cognitive_distill WHERE importance=0.3").fetchone()[0]
    cd_05 = c.execute("SELECT count(*) FROM cognitive_distill WHERE importance>=0.5").fetchone()[0]
    cd_ex = c.execute("SELECT count(*) FROM cognitive_distill WHERE expires_at IS NOT NULL").fetchone()[0]
    cd_em = c.execute("SELECT count(*) FROM cognitive_distill WHERE embedding IS NOT NULL AND embedding != ''").fetchone()[0]
    nd_t = c.execute("SELECT count(*) FROM memory_graph_nodes").fetchone()[0]
    eg_t = c.execute("SELECT count(*) FROM memory_graph_edges").fetchone()[0]
    sl_t = c.execute("SELECT count(*) FROM memory_graph_edges WHERE from_node_id=to_node_id").fetchone()[0]
    se_t = c.execute("SELECT count(*) FROM memory_graph_edges WHERE relation!=''").fetchone()[0]
    id_t = c.execute("SELECT traits FROM identity_memory WHERE user_id=?", (UID,)).fetchone()
    ss_t = c.execute("SELECT turn_count,last_distill_turn,dangling_threads FROM session_state WHERE session_id=?", (SID,)).fetchone()

    print("\n" + "=" * 80)
    print("最终校验")
    print("=" * 80)
    print(f"chat_history      : {ch_t}")
    print(f"cognitive_distill : {cd_t} (0.3:{cd_03} >=0.5:{cd_05} expires:{cd_ex} emb:{cd_em})")
    print(f"图谱               : {nd_t} nodes / {eg_t} edges (semantic={se_t} co-occur={eg_t-se_t})")
    print(f"自环               : {sl_t}")
    print(f"蒸馏触发           : {len(distill_turns)} 次 (turns: {distill_turns})")
    print(f"session            : turn={ss_t[0]} last_distill={ss_t[1]}")
    print(f"  悬案             : {ss_t[2]}")
    try:
        traits = json.loads(id_t[0]) if id_t and id_t[0] else []
        print(f"identity traits    : {len(traits)}")
    except:
        pass

    # ─── 三通道融合 ───
    print("\n--- 三通道融合 retrieval ---")
    for query, label in [
        ("最近在学什么新技术", "无空格长句"),
        ("Rust 小王 冥想 学习", "空格分开词"),
    ]:
        fused = await rcms.retrieve_memories(UID, query, "open", total_cap=8, session_id=SID)
        channels = {}
        for item in fused:
            channels[item[1]] = channels.get(item[1], 0) + 1
        print(f"  query={query!r} ({label}) -> {len(fused)} 条, 通道分布: {channels}")
        for content, tag in fused:
            print(f"    [{tag}] {content[:70]}")

    # ─── 图谱 BFS ───
    print("\n--- 图谱 BFS 激活扩散 ---")
    for seed in ["Rust", "小王"]:
        bfs = rcms._graph_activation_diffusion(UID, [seed])
        labels = [x[0] for x in bfs] if bfs else []
        print(f"  {seed} -> {labels}")

    # ─── Embedding 检索 ───
    print("\n--- Embedding 向量检索 ---")
    for q in ["今天工作好累压力大", "Rust 编程", "小王", "冥想和跑步"]:
        results, source = await rcms.retrieve_by_embedding(UID, q, limit=3)
        if results:
            print(f"  {q!r} ({source}) -> {len(results)} hits")
            for content, score in results:
                print(f"    [{score:.3f}] {content[:50]}")
        else:
            print(f"  {q!r} -> ({source})")

    # ─── narrative_context ───
    print("\n--- RCMS narrative_context (含三通道记忆) ---")
    lt = rcms._load_long_term_context(UID)
    memories = await rcms.retrieve_memories(UID, "最近在忙什么", "open", total_cap=5, session_id=SID)
    print(f"  [retrieve_memories -> {len(memories)} 条, 通道: {dict(zip(*[iter(sum(([tag],[tag]) for _,tag in memories),[])])) if len(set(tag for _,tag in memories))==1 else {tag:sum(1 for _,t in memories if t==tag) for tag in set(t for _,t in memories) if True}}")
    ctx = rcms.narrative_context("engaged", session_id=SID, user_id=UID,
                                 user_input="最近在忙什么", long_term=lt, memories=memories)
    print(ctx)

    # ─── 蒸馏摘要 ───
    if cd_05 > 0:
        print("\n蒸馏摘要 (imp>=0.5):")
        rows = c.execute("SELECT id,keylabel,importance,expires_at FROM cognitive_distill WHERE importance>=0.5 AND keylabel IS NOT NULL ORDER BY id").fetchall()
        for r in rows:
            expires = " [时效]" if r[3] else ""
            print(f"  [{r[0]}] imp={r[2]}{expires} | {(r[1] or '')[:60]}")

    # ─── PASS/FAIL ───
    errs = []
    if ch_t != 140: errs.append(f"chat_history {ch_t}")
    if sl_t > 0: errs.append(f"self-loops {sl_t}")
    if len(distill_turns) < 2: errs.append(f"distill {len(distill_turns)}")
    if cd_05 < 5: errs.append(f"hi-count {cd_05}")
    if cd_em < 10: errs.append(f"emb {cd_em}")
    if nd_t < 50: errs.append(f"nodes {nd_t}")
    if se_t < 2: errs.append(f"semantic edges {se_t}")
    if errs:
        print(f"\n  FAIL: {'; '.join(errs)}")
    else:
        print("\n  PASS all checks")

    rcms.close()
    os.unlink(db_path)
    print(f"DB cleaned: {db_path}")

asyncio.run(main())
