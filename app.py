"""
CourseRepoFinder - 面向高校课程作业的 GitHub 仓库检索与推荐系统
Streamlit 前端应用
"""

import streamlit as st
from dotenv import load_dotenv

from src.query_parser import parse_query
from src.search_engine import get_search_engine
from src.search_planner import build_search_plan
from src.course_kb import get_course_kb

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="CourseHunt",
    page_icon="",
    layout="wide"
)


@st.cache_resource
def get_engine():
    return get_search_engine()


def main():
    st.title("CourseHunt")
    st.caption("Hunt the right repo for your course — 面向高校课程作业的 GitHub 仓库智能检索系统")

    # 侧边栏配置
    with st.sidebar:
        st.header("配置")

        # 搜索模式
        mode = st.radio(
            "搜索模式",
            ["fast", "deep"],
            format_func=lambda x: "快速模式" if x == "fast" else "深度模式",
            help="快速模式：4路召回，分析前20个仓库；深度模式：8路召回，分析前50个仓库"
        )

        # 结果数量
        max_results = st.slider("显示结果数", 5, 30, 10)

        st.markdown("---")
        st.markdown("### 关于")
        st.markdown("""
        本系统针对 GitHub 原生搜索的不足，提供：
        - **查询扩展**：自动处理中英文课程名
        - **学校识别**：识别仓库所属高校
        - **作业识别**：识别课程作业特征
        - **用户画像**：通过作者其他仓库推断
        - **多路召回**：RRF 融合多路结果
        - **可解释推荐**：展示推荐理由和证据
        """)
        course_count = len(get_course_kb().courses)
        st.caption(f"当前课程知识库覆盖 {course_count} 门/组课程，已加入南开计科 2024 培养方案核心课程。")
        st.link_button("NKUCS.ICU 课程经验站", "https://nkucs.icu/#/")
        st.link_button("课程讨论 Issues", "https://github.com/NKUCS-ICU/NKUCS.ICU/issues")

        # 缓存统计
        st.markdown("---")
        st.markdown("### 缓存统计")
        engine = get_engine()
        cache_stats = engine.github.get_cache_stats()
        st.text(f"内存缓存: {cache_stats.get('memory_count', 0)} 条")
        st.text(f"磁盘缓存: {cache_stats.get('disk_count', 0)} 条")

        if st.button("清除缓存"):
            engine.github.clear_cache()
            st.success("缓存已清除")

    # 主界面
    col1, col2 = st.columns([3, 1])

    with col1:
        course_input = st.text_input(
            "输入课程名称或查询",
            placeholder="例如：南开 并行程序设计 lab2"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("搜索", type="primary", use_container_width=True)

    # 预设课程快捷按钮
    st.markdown("**快捷搜索：**")
    preset_cols = st.columns(4)
    preset_queries = [
        "南开 并行程序设计",
        "南开 操作系统",
        "南开 编译原理",
        "南开 数据库",
        "南开 计算机网络",
        "南开 数据结构",
        "南开 信息检索系统原理",
        "南开 机器学习及应用"
    ]

    for i, query in enumerate(preset_queries):
        with preset_cols[i % len(preset_cols)]:
            if st.button(query, use_container_width=True):
                course_input = query
                search_button = True

    # 查询解析预览
    if course_input:
        intent = parse_query(course_input)
        with st.expander("查询解析结果", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**学校:** {intent.school or '未识别'}")
                st.markdown(f"**课程:** {intent.course or '未识别'}")
            with col2:
                st.markdown(f"**作业类型:** {intent.assignment_type or '未识别'}")
                st.markdown(f"**作业编号:** {intent.assignment_number or '未识别'}")
            with col3:
                st.markdown(f"**意图类型:** {intent.intent_type}")
                st.markdown(f"**解析置信度:** {intent.confidence:.0%}")
        with st.expander("检索计划预览", expanded=False):
            preview_plan = build_search_plan(intent, budget=4 if mode == "fast" else 8)
            for task in preview_plan.tasks:
                st.markdown(f"- `{task.route}` `{task.query}`")

    # 执行搜索
    if search_button and course_input:
        perform_search(course_input, mode, max_results)


def perform_search(query: str, mode: str, max_results: int):
    """执行搜索"""
    engine = get_engine()

    st.markdown("---")
    st.markdown("### 搜索结果")

    # 显示进度
    with st.spinner("正在搜索..."):
        results = engine.search(query, mode)

    if not results:
        st.warning("未找到相关仓库，请尝试其他关键词")
        return

    st.success(f"找到 {len(results)} 个候选仓库")

    # 耗时明细
    _show_trace(engine.last_trace)

    # 展示结果
    for i, result in enumerate(results[:max_results]):
        render_result_card(i + 1, result)


def _show_trace(trace):
    """展示各步骤耗时明细。"""
    if not trace or trace.t_total <= 0:
        return

    steps = [
        ("查询解析",                trace.t_query_parse),
        ("构建检索计划",            trace.t_build_plan),
        ("GitHub API 多路搜索",     trace.t_execute_api),
        ("种子仓库拉取",            trace.t_seed_repos),
        ("RRF 第一轮融合",          trace.t_rrf_first),
        (f"丰富候选 README/目录树",  trace.t_enrich),
        ("写入本地语料库",          trace.t_persist),
        ("自建 BM25 检索",          trace.t_bm25),
        ("BGE-M3 Embedding 补算",   trace.t_dense_embed),
        ("BGE-M3 向量搜索",          trace.t_dense_search),
        ("RRF 三路融合",            trace.t_rrf_final),
        ("候选合并",                trace.t_merge),
        ("用户画像分析",            trace.t_owner_profile),
        ("证据构建",                trace.t_evidence),
        ("混合排序",                trace.t_ranking),
        ("构建最终结果",            trace.t_build_results),
    ]

    # 过滤掉为 0 的步骤（未执行）
    active_steps = [(name, t) for name, t in steps if t > 0.001]

    with st.expander(f"各步骤耗时明细（总计 {trace.t_total:.2f}s）", expanded=False):
        # 条形图
        max_t = max(t for _, t in active_steps) if active_steps else 1
        for name, t in active_steps:
            pct = t / trace.t_total * 100
            bar_len = int(t / max_t * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = "red" if t > 5 else ("orange" if t > 1 else "green")
            st.markdown(
                f":{color}[{bar}] {name}: **{t:.2f}s** ({pct:.0f}%)"
            )

        # 网络/CPU 概要
        api_total = trace.t_execute_api + trace.t_seed_repos + trace.t_enrich + trace.t_owner_profile + trace.t_dense_embed + trace.t_dense_search
        cpu_total = trace.t_total - api_total
        st.markdown(f"---\nI/O 耗时: **{api_total:.2f}s** / 计算耗时: **{cpu_total:.2f}s**")


def render_result_card(rank: int, result):
    """渲染结果卡片"""
    repo_data = result.repo_data
    evidence = result.evidence

    # 仓库基本信息
    full_name = result.repo
    repo_url = repo_data.get("html_url", f"https://github.com/{full_name}")
    stars = repo_data.get("stargazers_count", 0)
    forks = repo_data.get("forks_count", 0)
    language = repo_data.get("language", "N/A")
    description = repo_data.get("description", "")
    updated_at = repo_data.get("pushed_at", "")[:10]

    with st.container():
        # 标题行
        col1, col2 = st.columns([4, 1])

        with col1:
            # 置信度标签
            confidence_color = {
                "high": "green",
                "medium": "orange",
                "low": "red"
            }.get(result.confidence, "gray")

            st.markdown(
                f"### #{rank} [{full_name}]({repo_url}) "
                f":{confidence_color}[{result.confidence}]"
            )
            if description:
                st.markdown(f"*{description}*")

        with col2:
            score_display = f"{result.score:.0%}"
            st.markdown(f"<h1 style='text-align: center;'>{score_display}</h1>",
                       unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>综合评分</p>",
                       unsafe_allow_html=True)

        # 元数据行
        meta_cols = st.columns(5)
        with meta_cols[0]:
            st.metric("Stars", stars)
        with meta_cols[1]:
            st.metric("Forks", forks)
        with meta_cols[2]:
            st.metric("Language", language)
        with meta_cols[3]:
            st.metric("Updated", updated_at)
        with meta_cols[4]:
            st.metric("RRF Score", f"{result.rrf_score:.3f}")

        # 推荐理由
        if result.reasons:
            st.markdown("**推荐理由：**")
            for reason in result.reasons[:5]:
                st.markdown(f"- {reason}")

        # 风险提示
        if result.risks:
            st.markdown("**风险提示：**")
            for risk in result.risks:
                st.markdown(f"- :orange[{risk}]")

        # 详细证据（可展开）
        with st.expander("查看详细证据"):
            # 学校证据
            if evidence.school_evidence:
                st.markdown("**学校证据：**")
                for ev in evidence.school_evidence[:3]:
                    st.text(f"  [{ev.field}] {ev.text} (置信度: {ev.confidence:.0%})")

            # 课程证据
            if evidence.course_evidence:
                st.markdown("**课程证据：**")
                for ev in evidence.course_evidence[:3]:
                    st.text(f"  [{ev.field}] {ev.text} (置信度: {ev.confidence:.0%})")

            # 作业证据
            if evidence.assignment_evidence:
                st.markdown("**作业证据：**")
                for ev in evidence.assignment_evidence[:3]:
                    st.text(f"  [{ev.field}] {ev.text} (置信度: {ev.confidence:.0%})")

            # 用户画像证据
            if evidence.owner_evidence:
                st.markdown("**用户画像证据：**")
                for ev in evidence.owner_evidence[:3]:
                    st.text(f"  [{ev.field}] {ev.text}")

        # 用户画像（可展开）
        owner_context = result.owner_context
        if owner_context.get("school_confidence", 0) > 0.3:
            with st.expander("查看用户画像分析"):
                st.markdown(f"**作者：** {owner_context.get('login', 'Unknown')}")
                st.markdown(f"**公开仓库数：** {owner_context.get('total_repos', 0)}")
                st.markdown(f"**学校置信度：** {owner_context.get('school_confidence', 0):.0%}")

                school_repos = owner_context.get("school_repo_examples", [])
                if school_repos:
                    st.markdown(f"**学校相关仓库：** {', '.join(school_repos[:3])}")

                course_repos = owner_context.get("course_repo_examples", [])
                if course_repos:
                    st.markdown(f"**课程相关仓库：** {', '.join(course_repos[:3])}")

        st.markdown("---")


if __name__ == "__main__":
    main()
