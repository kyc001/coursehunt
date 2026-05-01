"""
查询解析模块
识别用户查询中的学校、课程、作业类型等意图
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .school_kb import get_school_kb
from .course_kb import get_course_kb
from .matching import contains_signal


@dataclass
class QueryIntent:
    """查询意图"""
    raw_query: str
    school: Optional[str] = None           # 学校 ID
    school_aliases: List[str] = field(default_factory=list)  # 学校别名
    course: Optional[str] = None           # 课程 ID
    course_aliases: List[str] = field(default_factory=list)  # 课程别名
    assignment_type: Optional[str] = None  # 作业类型: lab/homework/project/report
    assignment_number: Optional[str] = None  # 作业编号: lab2, hw1
    resource_type: Optional[str] = None    # 资源类型: materials/notes/exam/courseware
    resource_aliases: List[str] = field(default_factory=list)  # 资源类型别名
    language: Optional[str] = None         # 编程语言
    technologies: List[str] = field(default_factory=list)  # 技术栈
    intent_type: str = "generic"           # 意图类型
    confidence: float = 0.0                # 解析置信度

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "school": self.school,
            "course": self.course,
            "assignment_type": self.assignment_type,
            "assignment_number": self.assignment_number,
            "resource_type": self.resource_type,
            "resource_aliases": self.resource_aliases,
            "language": self.language,
            "technologies": self.technologies,
            "intent_type": self.intent_type,
            "confidence": self.confidence
        }


# 作业类型关键词映射
ASSIGNMENT_TYPE_MAP = {
    "lab": ["lab", "实验", "experiment"],
    "homework": ["homework", "hw", "作业"],
    "project": ["project", "大作业", "课程设计", "course design"],
    "report": ["report", "报告", "实验报告"],
}

# 资料/资源类型关键词映射
RESOURCE_TYPE_MAP = {
    "materials": ["资料", "资源", "course material", "materials", "resource"],
    "notes": ["笔记", "note", "notes", "讲义", "summary"],
    "courseware": ["课件", "ppt", "slides", "slide", "courseware"],
    "exam": ["考试", "试卷", "真题", "期末", "期中", "复习", "exam", "final", "midterm", "review", "quiz"],
    "homework": ["作业", "homework", "hw", "assignment"],
    "lab": ["实验", "lab", "experiment", "report", "实验报告"],
}

# 编程语言关键词
LANGUAGE_MAP = {
    "c": ["c", "clang"],
    "cpp": ["c++", "cpp", "cxx"],
    "java": ["java"],
    "python": ["python", "py"],
    "cuda": ["cuda"],
    "verilog": ["verilog", "vhdl"],
    "assembly": ["assembly", "asm", "汇编"],
}

# 作业编号模式
ASSIGNMENT_NUMBER_PATTERN = r'(lab|hw|homework|experiment|实验|作业|project)\s*(\d+|[一-龥]{1,3})'

# 中文数字映射
CHINESE_NUM_MAP = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
}


class QueryParser:
    """查询解析器"""

    def __init__(self):
        self.school_kb = get_school_kb()
        self.course_kb = get_course_kb()

    def parse(self, query: str) -> QueryIntent:
        """
        解析用户查询

        Args:
            query: 用户输入的查询

        Returns:
            QueryIntent 对象
        """
        intent = QueryIntent(raw_query=query)
        query_lower = query.lower()

        # 1. 识别学校
        self._parse_school(query_lower, intent)

        # 2. 识别课程
        self._parse_course(query_lower, intent)

        # 3. 识别作业类型和编号
        self._parse_assignment(query_lower, intent)

        # 4. 识别资料/资源类型
        self._parse_resource_type(query_lower, intent)

        # 5. 识别编程语言
        self._parse_language(query_lower, intent)

        # 6. 识别技术栈
        self._parse_technologies(query_lower, intent)

        # 7. 判断意图类型
        self._classify_intent(intent)

        # 8. 计算置信度
        self._calculate_confidence(intent)

        return intent

    def _parse_school(self, query: str, intent: QueryIntent):
        """识别学校"""
        for school_id, info in self.school_kb.schools.items():
            # 检查所有别名
            for alias in self.school_kb.get_all_aliases(school_id):
                if contains_signal(query, alias):
                    intent.school = school_id
                    intent.school_aliases = self.school_kb.get_all_aliases(school_id)
                    return

    def _parse_course(self, query: str, intent: QueryIntent):
        """识别课程"""
        # 先尝试通过课程代码匹配
        code_pattern = r'[A-Z]{2,4}\d{3,4}'
        codes = re.findall(code_pattern, query.upper())
        for code in codes:
            course_id = self.course_kb.find_course_by_code(code)
            if course_id:
                intent.course = course_id
                intent.course_aliases = self.course_kb.get_all_aliases(course_id)
                return

        # 再通过别名精确匹配
        for course_id, info in self.course_kb.courses.items():
            for alias in self.course_kb.get_all_aliases(course_id):
                if contains_signal(query, alias):
                    intent.course = course_id
                    intent.course_aliases = self.course_kb.get_all_aliases(course_id)
                    return

        # 如果没有精确匹配，尝试从查询中提取课程关键词
        # 移除学校名、作业关键词、作业编号等，剩下的可能是课程名
        course_keywords = self._extract_course_keywords(query)
        if course_keywords:
            intent.course_aliases = [course_keywords]
            intent.course = None  # 未知课程，但有关键词

    def _extract_course_keywords(self, query: str) -> str:
        """从查询中提取可能的课程关键词"""
        # 移除学校相关词
        school_words = [
            "南开", "南开大学", "nku", "nankai", "nankai university",
            "清华", "北大", "tsinghua", "peking"
        ]

        # 移除作业相关词
        assignment_words = [
            "homework", "hw", "lab", "experiment", "project", "assignment",
            "report", "实验", "作业", "大作业", "课程设计", "实验报告",
            "资料", "资源", "笔记", "课件", "试卷", "复习", "考试",
            "notes", "slides", "exam", "review", "materials"
        ]

        # 移除常见停用词
        stop_words = ["的", "了", "和", "与", "课程", "原理", "系统"]

        words = query.split()
        filtered = []

        for word in words:
            word_lower = word.lower()
            # 跳过学校词
            if any(s in word_lower for s in school_words):
                continue
            # 跳过作业词
            if any(a in word_lower for a in assignment_words):
                continue
            # 跳过作业编号
            if re.match(r'^(lab|hw|homework|experiment)\s*\d+$', word_lower):
                continue
            # 跳过停用词
            if word in stop_words:
                continue
            # 跳过纯数字
            if word.isdigit():
                continue
            filtered.append(word)

        return " ".join(filtered) if filtered else ""

    def _parse_assignment(self, query: str, intent: QueryIntent):
        """识别作业类型和编号"""
        # 识别作业类型
        for assign_type, keywords in ASSIGNMENT_TYPE_MAP.items():
            for kw in keywords:
                if contains_signal(query, kw):
                    intent.assignment_type = assign_type
                    break
            if intent.assignment_type:
                break

        # 识别作业编号
        match = re.search(ASSIGNMENT_NUMBER_PATTERN, query, re.IGNORECASE)
        if match:
            prefix = match.group(1).lower()
            num_str = match.group(2)
            # 转换中文数字
            if num_str in CHINESE_NUM_MAP:
                num_str = CHINESE_NUM_MAP[num_str]
            intent.assignment_number = num_str
            if not intent.assignment_type:
                for assign_type, keywords in ASSIGNMENT_TYPE_MAP.items():
                    if prefix in [kw.lower() for kw in keywords]:
                        intent.assignment_type = assign_type
                        break

    def _parse_resource_type(self, query: str, intent: QueryIntent):
        """识别资料/资源类型"""
        matched_aliases = []
        for resource_type, keywords in RESOURCE_TYPE_MAP.items():
            for kw in keywords:
                if contains_signal(query, kw):
                    intent.resource_type = resource_type
                    matched_aliases = keywords
                    break
            if intent.resource_type:
                break

        if intent.assignment_type and not intent.resource_type:
            intent.resource_type = intent.assignment_type
            matched_aliases = ASSIGNMENT_TYPE_MAP.get(intent.assignment_type, [])

        intent.resource_aliases = matched_aliases

    def _parse_language(self, query: str, intent: QueryIntent):
        """识别编程语言"""
        for lang, keywords in LANGUAGE_MAP.items():
            for kw in keywords:
                if contains_signal(query, kw):
                    intent.language = lang
                    return

    def _parse_technologies(self, query: str, intent: QueryIntent):
        """识别技术栈"""
        tech_keywords = [
            "mpi", "openmp", "cuda", "pthread", "simd",
            "xv6", "nachos", "ucore", "rcore",
            "llvm", "flex", "bison", "antlr",
            "mysql", "postgresql", "sqlite",
            "pytorch", "tensorflow", "keras",
            "docker", "kubernetes", "git"
        ]

        for tech in tech_keywords:
            if contains_signal(query, tech):
                intent.technologies.append(tech)

    def _classify_intent(self, intent: QueryIntent):
        """判断意图类型"""
        has_school = intent.school is not None
        has_course = intent.course is not None
        has_course_keyword = len(intent.course_aliases) > 0  # 有提取的课程关键词
        has_assignment = intent.assignment_type is not None or intent.assignment_number is not None

        if has_school and has_course and has_assignment:
            intent.intent_type = "specific_school_course_assignment"
        elif has_school and has_course:
            intent.intent_type = "school_course"
        elif has_school and has_course_keyword and has_assignment:
            # 有学校、课程关键词（非预设）、作业
            intent.intent_type = "specific_school_course_assignment"
        elif has_school and has_course_keyword:
            # 有学校、课程关键词（非预设）
            intent.intent_type = "school_course"
        elif has_school and has_assignment:
            intent.intent_type = "school_assignment"
        elif has_course and has_assignment:
            intent.intent_type = "course_assignment"
        elif has_school:
            intent.intent_type = "school_only"
        elif has_course:
            intent.intent_type = "course_only"
        else:
            intent.intent_type = "generic"

    def _calculate_confidence(self, intent: QueryIntent):
        """计算解析置信度"""
        score = 0.0

        if intent.school:
            score += 0.35
        if intent.course:
            score += 0.35
        if intent.assignment_type:
            score += 0.15
        if intent.assignment_number:
            score += 0.10
        if intent.resource_type:
            score += 0.10
        if intent.language:
            score += 0.05

        intent.confidence = min(score, 1.0)


# 全局实例
_parser = None


def get_query_parser() -> QueryParser:
    """获取查询解析器单例"""
    global _parser
    if _parser is None:
        _parser = QueryParser()
    return _parser


def parse_query(query: str) -> QueryIntent:
    """解析查询的便捷函数"""
    return get_query_parser().parse(query)
