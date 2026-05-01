"""
课程别名与查询扩展模块
管理课程名称、学校名称的同义词映射
"""

# 学校别名词典
SCHOOL_ALIASES = {
    "南开大学": [
        "南开大学", "南开", "NKU", "Nankai", "Nankai University",
        "nankai", "nku"
    ],
    "清华大学": [
        "清华大学", "清华", "Tsinghua", "Tsinghua University",
        "THU", "thu", "tsinghua"
    ],
    "北京大学": [
        "北京大学", "北大", "PKU", "Peking University",
        "peking", "pku"
    ],
}

# 课程别名词典 (以南开大学计算机课程为例)
COURSE_ALIASES = {
    # 并行程序设计
    "并行程序设计": [
        "并行程序设计", "并行计算", "并行编程",
        "Parallel Programming", "parallel programming", "parallel",
        "HPC", "high performance computing", "High Performance Computing",
        "COSCXXX"  # 需要替换为实际课程代码
    ],

    # 操作系统
    "操作系统": [
        "操作系统", "OS实验", "操作系统实验",
        "Operating System", "operating system", "OS", "os",
        "COSCXXX"
    ],

    # 编译原理
    "编译原理": [
        "编译原理", "编译器", "编译原理实验",
        "Compiler", "compiler", "Compiler Principles",
        "COSCXXX"
    ],

    # 数据库系统
    "数据库系统": [
        "数据库", "数据库系统", "数据库实验",
        "Database", "database", "Database System", "DB",
        "COSCXXX"
    ],

    # 计算机网络
    "计算机网络": [
        "计算机网络", "网络编程", "网络实验",
        "Computer Network", "computer network", "Networking",
        "COSCXXX"
    ],

    # 数据结构
    "数据结构": [
        "数据结构", "数据结构与算法",
        "Data Structure", "data structure", "Data Structures",
        "COSCXXX"
    ],

    # 人工智能
    "人工智能": [
        "人工智能", "AI实验", "机器学习",
        "Artificial Intelligence", "artificial intelligence", "AI", "ai",
        "Machine Learning", "machine learning", "ML",
        "COSCXXX"
    ],

    # 软件工程
    "软件工程": [
        "软件工程", "软件工程实验",
        "Software Engineering", "software engineering", "SE",
        "COSCXXX"
    ],
}

# 作业相关关键词
ASSIGNMENT_KEYWORDS = [
    "homework", "hw", "lab", "experiment", "实验", "作业",
    "project", "assignment", "report", "报告", "大作业",
    "课程设计", "course design", "final project"
]

# 课程代码模式 (匹配类似 COSC1001 的格式)
COURSE_CODE_PATTERN = r'[A-Z]{2,4}\d{3,4}'


def expand_query(course_name: str, school_name: str = "南开大学") -> dict:
    """
    扩展用户查询，生成多路召回的关键词组合

    Args:
        course_name: 用户输入的课程名
        school_name: 学校名称，默认南开大学

    Returns:
        扩展后的查询字典，包含多路召回关键词
    """
    # 获取课程别名
    course_aliases = []
    for key, aliases in COURSE_ALIASES.items():
        if course_name in aliases or course_name.lower() in [a.lower() for a in aliases]:
            course_aliases = aliases
            break

    if not course_aliases:
        # 如果没找到预设别名，使用原始输入
        course_aliases = [course_name]

    # 获取学校别名
    school_aliases = SCHOOL_ALIASES.get(school_name, [school_name])

    return {
        "course_aliases": course_aliases,
        "school_aliases": school_aliases,
        "assignment_keywords": ASSIGNMENT_KEYWORDS,
        "original_course": course_name,
        "original_school": school_name
    }


def generate_search_queries(expanded: dict) -> list:
    """
    根据扩展后的查询生成多路召回的 GitHub 搜索查询

    Args:
        expanded: expand_query 返回的扩展查询字典

    Returns:
        搜索查询列表，每个元素为 (query_string, search_type, description)
    """
    queries = []
    course_aliases = expanded["course_aliases"]
    school_aliases = expanded["school_aliases"]
    assignment_keywords = expanded["assignment_keywords"]

    # 召回策略1: 学校 + 课程名 (in:name,description,readme)
    for school in school_aliases[:3]:  # 取前3个学校别名
        for course in course_aliases[:3]:  # 取前3个课程别名
            query = f"{school} {course} in:name,description,readme"
            queries.append((query, "repo", f"学校({school}) + 课程({course})"))

    # 召回策略2: 课程名 + 作业关键词
    for course in course_aliases[:2]:
        for assign in assignment_keywords[:3]:
            query = f"{course} {assign} in:readme"
            queries.append((query, "repo", f"课程({course}) + 作业({assign})"))

    # 召回策略3: 课程代码搜索
    for course in course_aliases:
        if len(course) >= 5 and course[:4].isalpha() and course[4:].isdigit():
            query = f"{course} in:name,readme"
            queries.append((query, "repo", f"课程代码({course})"))

    # 召回策略4: 纯课程名搜索 (README)
    for course in course_aliases[:2]:
        query = f"{course} in:readme"
        queries.append((query, "repo", f"课程名({course}) README"))

    return queries


def get_course_display_name(course_name: str) -> str:
    """获取课程的显示名称"""
    for key, aliases in COURSE_ALIASES.items():
        if course_name in aliases:
            return key
    return course_name
