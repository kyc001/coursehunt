"""
模块测试脚本
验证各个模块的基本功能
"""

from src.query_parser import parse_query
from src.search_planner import build_search_plan
from src.school_kb import get_school_kb
from src.course_kb import get_course_kb
from src.evidence_builder import get_evidence_builder
from src.matching import contains_signal
from src.search_engine import get_search_engine
from src.query_expander import expand_query


def test_school_kb():
    """测试学校知识库"""
    print("=" * 50)
    print("测试学校知识库")
    print("=" * 50)

    kb = get_school_kb()

    # 测试学校查询
    school = kb.get_school("nankai")
    print(f"\n学校: {school['canonical_name']}")
    print(f"别名: {kb.get_all_aliases('nankai')[:5]}")
    print(f"域名: {kb.get_domains('nankai')}")
    print(f"GitHub Org: {kb.get_github_orgs('nankai')}")

    # 测试信号检测
    test_texts = [
        "NKU Parallel Programming Homework",
        "南开大学操作系统实验",
        "Nankai University Database Lab",
        "清华大学编译原理"  # 应该不匹配
    ]

    for text in test_texts:
        result = kb.check_school_signal(text, "nankai")
        print(f"\n文本: {text}")
        print(f"  发现: {result['found']}, 信号: {result['signals']}, 置信度: {result['confidence']:.2f}")

    print("\n[OK] 学校知识库测试通过\n")


def test_course_kb():
    """测试课程知识库"""
    print("=" * 50)
    print("测试课程知识库")
    print("=" * 50)

    kb = get_course_kb()

    # 测试课程查询
    course = kb.get_course("parallel_programming")
    print(f"\n课程: {course['canonical_name']}")
    print(f"别名: {kb.get_all_aliases('parallel_programming')[:5]}")
    print(f"技术栈: {kb.get_technologies('parallel_programming')}")
    print(f"课程代码: {kb.get_course_codes('parallel_programming')}")

    # 测试信号检测
    test_texts = [
        "MPI CUDA OpenMP Parallel Programming",
        "并行程序设计 实验1",
        "HPC homework lab2",
        "操作系统实验",  # 不应该匹配并行课程
        "nankai university"  # 不应该因为 nankai 里的 ai 匹配人工智能
    ]

    for text in test_texts:
        result = kb.check_course_signal(text, "parallel_programming")
        print(f"\n文本: {text}")
        print(f"  发现: {result['found']}, 信号: {result['signals']}, 技术: {result['tech_signals']}")

    ai_false_positive = kb.check_course_signal("nankai university", "artificial_intelligence")
    assert not ai_false_positive["found"], "短别名 AI 不应命中 nankai"

    print("\n[OK] 课程知识库测试通过\n")


def test_query_parser():
    """测试查询解析"""
    print("=" * 50)
    print("测试查询解析")
    print("=" * 50)

    test_queries = [
        "南开 并行程序设计 lab2",
        "南开大学 操作系统 实验3",
        "NKU 编译原理 homework",
        "数据库 作业",
        "并行程序设计",
        "南开大学 课程作业",
        "COSC3000 lab1",
        "Nankai computer network",
        "NKU AI lab1"
    ]

    for query in test_queries:
        intent = parse_query(query)
        print(f"\n查询: {query}")
        print(f"  学校: {intent.school}")
        print(f"  课程: {intent.course}")
        print(f"  作业类型: {intent.assignment_type}")
        print(f"  作业编号: {intent.assignment_number}")
        print(f"  意图类型: {intent.intent_type}")
        print(f"  置信度: {intent.confidence:.0%}")

    assert parse_query("Nankai computer network").course == "computer_network"
    assert parse_query("Nankai computer network").language is None
    assert parse_query("NKU AI lab1").course == "artificial_intelligence"
    assert parse_query("南开 并行程序设计 lab2").assignment_type == "lab"
    assert parse_query("南开大学 高数资料").course == "advanced_mathematics"
    assert parse_query("南开大学 高数资料").resource_type == "materials"

    print("\n[OK] 查询解析测试通过\n")


def test_search_planner():
    """测试检索计划器"""
    print("=" * 50)
    print("测试检索计划器")
    print("=" * 50)

    test_queries = [
        "南开 并行程序设计 lab2",
        "南开 操作系统",
        "编译原理 homework",
    ]

    for query in test_queries:
        intent = parse_query(query)
        plan = build_search_plan(intent, budget=6)

        print(f"\n查询: {query}")
        print(f"意图类型: {intent.intent_type}")
        print(f"搜索任务数: {len(plan.tasks)}")

        for task in plan.tasks:
            print(f"  [{task.route}] {task.query}")
            print(f"    精度: {task.expected_precision}, {task.description}")

    print("\n[OK] 检索计划器测试通过\n")


def test_query_expansion():
    """测试通用查询扩展和剪枝"""
    print("=" * 50)
    print("测试查询扩展")
    print("=" * 50)

    intent = parse_query("南开大学 高数资料")
    expansion = expand_query(intent, max_queries=6)
    selected_texts = [q.text for q in expansion.selected]
    print(f"学校别名: {expansion.school_aliases[:5]}")
    print(f"课程别名: {expansion.course_aliases[:6]}")
    print(f"资源词: {expansion.resource_aliases[:6]}")
    for text in selected_texts:
        print(f"  {text}")

    assert any("高数" in text or "高等数学" in text for text in selected_texts)
    assert any("资料" in text or "笔记" in text or "notes" in text for text in selected_texts)
    assert len(selected_texts) <= 6

    reverse_intent = parse_query("高数 南开大学")
    reverse_expansion = expand_query(reverse_intent, max_queries=6)
    assert reverse_intent.course == intent.course
    assert reverse_intent.school == intent.school
    assert any("南开" in q.text and ("高数" in q.text or "高等数学" in q.text)
               for q in reverse_expansion.selected)

    print("\n[OK] 查询扩展测试通过\n")


def test_null_safe_evidence():
    """测试 GitHub 返回空字段时证据抽取不崩溃"""
    print("=" * 50)
    print("测试空值安全")
    print("=" * 50)

    assert not contains_signal(None, "NKU")
    assert not contains_signal("NKU", None)

    builder = get_evidence_builder()
    repo_data = {
        "full_name": None,
        "name": None,
        "description": None,
        "topics": [None, "nku"],
        "readme_text": None,
    }
    evidence = builder.build_evidence(repo_data, {}, school_id="nankai")
    print(f"仓库: {evidence.repo!r}, 学校证据: {[e.text for e in evidence.school_evidence]}")
    print("\n[OK] 空值安全测试通过\n")


def test_collection_repo_evidence():
    """测试多课程合集仓库可以通过目录路径形成课程证据"""
    print("=" * 50)
    print("测试合集仓库路径证据")
    print("=" * 50)

    builder = get_evidence_builder()
    repo_data = {
        "full_name": "example/NKU-share",
        "name": "NKU-share",
        "description": "NKU计网学院各个课程作业",
        "topics": [],
        "readme_text": "",
        "known_collection": True,
        "collection_reason": "南开课程资料合集种子仓库",
        "tree_paths": ["操作系统/lab1", "数据库系统/project", "编译原理"],
    }
    evidence = builder.build_evidence(
        repo_data, {}, school_id="nankai", course_id="operating_system"
    )
    assert any(ev.field == "path" for ev in evidence.course_evidence)
    assert any("合集" in reason for reason in evidence.reasons)
    print(f"推荐理由: {evidence.reasons[:2]}")
    print("\n[OK] 合集仓库路径证据测试通过\n")


def test_seed_repo_loading():
    """测试南开种子仓库配置能进入召回候选"""
    print("=" * 50)
    print("测试种子仓库召回")
    print("=" * 50)

    engine = get_search_engine()
    intent = parse_query("Nankai operating system")
    route_results = {}
    engine._add_seed_repositories(route_results, intent)
    repos = {repo.get("full_name") for repo in route_results.get("seed_collections", [])}
    assert "Absurdaaa/NKU_CS_courses" in repos
    assert "fscdc/NKU-CS-HELP" in repos
    assert "Luhaozhhhhe/NKU_Finance_Course" not in repos
    print(f"种子仓库数: {len(repos)}")
    print("\n[OK] 种子仓库召回测试通过\n")


def main():
    print("\n[CourseRepoFinder] 模块测试\n")

    test_school_kb()
    test_course_kb()
    test_query_parser()
    test_search_planner()
    test_query_expansion()
    test_null_safe_evidence()
    test_collection_repo_evidence()
    test_seed_repo_loading()

    print("=" * 50)
    print("[OK] 所有测试通过！")
    print("=" * 50)
    print("\n运行 'pixi run start' 启动应用")


if __name__ == "__main__":
    main()
