"""
检索计划器
根据查询意图生成多路召回策略
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .query_parser import QueryIntent
from .school_kb import get_school_kb
from .course_kb import get_course_kb
from .query_expander import QueryExpansion, expand_query


@dataclass
class SearchTask:
    """单个搜索任务"""
    name: str                # 任务名称
    query: str               # 搜索查询
    route: str               # 路由类型: exact/broad/code/readme/owner
    priority: int            # 优先级 (1=最高)
    expected_precision: str  # 预期精度: high/medium/low
    description: str = ""    # 描述


@dataclass
class SearchPlan:
    """检索计划"""
    intent: QueryIntent
    tasks: List[SearchTask] = field(default_factory=list)
    total_budget: int = 8  # 最大搜索请求数
    expansion: Optional[QueryExpansion] = None  # 查询扩展调试信息


class SearchPlanner:
    """检索计划器"""

    def __init__(self):
        self.school_kb = get_school_kb()
        self.course_kb = get_course_kb()

    def build_plan(self, intent: QueryIntent, budget: int = 8) -> SearchPlan:
        """
        根据查询意图生成检索计划

        Args:
            intent: 查询意图
            budget: API 预算

        Returns:
            SearchPlan 对象
        """
        plan = SearchPlan(intent=intent, total_budget=budget)
        plan.expansion = expand_query(intent, max_queries=budget)
        for idx, expanded in enumerate(plan.expansion.selected, start=1):
            precision = (
                "high" if expanded.expected_precision >= 0.8
                else "medium" if expanded.expected_precision >= 0.55
                else "low"
            )
            plan.tasks.append(SearchTask(
                name=f"expanded_{idx}_{expanded.route}",
                query=expanded.text,
                route=expanded.route,
                priority=expanded.priority,
                expected_precision=precision,
                description=expanded.reason or "查询扩展"
            ))

        # 根据意图类型生成不同的搜索策略
        if intent.intent_type == "specific_school_course_assignment":
            self._build_specific_plan(intent, plan)
        elif intent.intent_type == "school_course":
            self._build_school_course_plan(intent, plan)
        elif intent.intent_type == "course_assignment":
            self._build_course_assignment_plan(intent, plan)
        elif intent.intent_type == "course_only":
            self._build_course_only_plan(intent, plan)
        elif intent.intent_type == "school_only":
            self._build_school_only_plan(intent, plan)
        else:
            self._build_generic_plan(intent, plan)

        # 去重并按优先级限制任务数量
        deduped = []
        seen_queries = set()
        for task in sorted(plan.tasks, key=lambda t: (t.priority + (0 if t.name.startswith("expanded_") else 3), t.priority)):
            key = (task.route, task.query)
            if key in seen_queries:
                continue
            seen_queries.add(key)
            deduped.append(task)
        plan.tasks = deduped[:budget]

        return plan

    def _build_specific_plan(self, intent: QueryIntent, plan: SearchPlan):
        """学校 + 课程 + 作业 的精确查询"""
        school_aliases = intent.school_aliases[:3]
        course_aliases = intent.course_aliases[:3]
        assign_num = intent.assignment_number or ""
        assign_type = intent.assignment_type or "lab"

        # 高精度查询
        for school in school_aliases[:2]:
            for course in course_aliases[:2]:
                query = f'"{school}" "{course}"'
                if assign_num:
                    query += f' {assign_type}{assign_num}'
                query += ' in:readme'
                plan.tasks.append(SearchTask(
                    name=f"exact_{school}_{course}",
                    query=query,
                    route="exact",
                    priority=1,
                    expected_precision="high",
                    description=f"精确匹配: {school} + {course}"
                ))

        # 作业文件路径查询
        if assign_num:
            for course in course_aliases[:2]:
                query = f'"{course}" {assign_type}{assign_num} in:name,description,readme'
                plan.tasks.append(SearchTask(
                    name=f"path_{course}_{assign_type}{assign_num}",
                    query=query,
                    route="path",
                    priority=2,
                    expected_precision="high",
                    description=f"作业路径: {course} + {assign_type}{assign_num}"
                ))

        # 技术栈查询
        if intent.technologies:
            for school in school_aliases[:1]:
                for tech in intent.technologies[:2]:
                    query = f'"{school}" {tech}'
                    if assign_num:
                        query += f' {assign_type}{assign_num}'
                    query += ' in:readme'
                    plan.tasks.append(SearchTask(
                        name=f"tech_{school}_{tech}",
                        query=query,
                        route="tech",
                        priority=3,
                        expected_precision="medium",
                        description=f"技术栈: {school} + {tech}"
                    ))

        # 课程代码查询
        course_codes = self.course_kb.get_course_codes(intent.course)
        if course_codes:
            for code in course_codes[:2]:
                query = f'{code} in:name,readme'
                if assign_num:
                    query += f' {assign_type}{assign_num}'
                plan.tasks.append(SearchTask(
                    name=f"code_{code}",
                    query=query,
                    route="code",
                    priority=3,
                    expected_precision="medium",
                    description=f"课程代码: {code}"
                ))

        # 目录/文件路径查询，用于召回多课程合集仓库中的课程子文件夹
        for course in course_aliases[:2]:
            plan.tasks.append(SearchTask(
                name=f"code_path_{course}",
                query=f'"{course}" in:path',
                route="code_path",
                priority=3,
                expected_precision="medium",
                description=f"路径命中: {course}"
            ))

        # 泛化查询
        for course in course_aliases[:1]:
            query = f'"{course}" {assign_type} in:name,description,readme'
            plan.tasks.append(SearchTask(
                name=f"broad_{course}",
                query=query,
                route="broad",
                priority=4,
                expected_precision="medium",
                description=f"泛化: {course} + {assign_type}"
            ))

    def _build_school_course_plan(self, intent: QueryIntent, plan: SearchPlan):
        """学校 + 课程 查询"""
        school_aliases = intent.school_aliases[:3]
        course_aliases = intent.course_aliases[:3]

        # 高精度查询
        for school in school_aliases[:2]:
            for course in course_aliases[:2]:
                query = f'"{school}" "{course}" in:name,description,readme'
                plan.tasks.append(SearchTask(
                    name=f"exact_{school}_{course}",
                    query=query,
                    route="exact",
                    priority=1,
                    expected_precision="high",
                    description=f"精确匹配: {school} + {course}"
                ))

        # 课程代码查询
        course_codes = self.course_kb.get_course_codes(intent.course)
        for code in course_codes[:2]:
            for school in school_aliases[:1]:
                query = f'{code} "{school}" in:name,readme'
                plan.tasks.append(SearchTask(
                    name=f"code_{code}_{school}",
                    query=query,
                    route="code",
                    priority=2,
                    expected_precision="high",
                    description=f"课程代码: {code} + {school}"
                ))

        # 目录/文件路径查询，用于召回课程合集仓库
        for course in course_aliases[:2]:
            plan.tasks.append(SearchTask(
                name=f"code_path_{course}",
                query=f'"{course}" in:path',
                route="code_path",
                priority=2,
                expected_precision="medium",
                description=f"路径命中: {course}"
            ))

        # 技术栈查询
        techs = self.course_kb.get_technologies(intent.course)
        for school in school_aliases[:1]:
            for tech in techs[:3]:
                query = f'"{school}" {tech} in:readme'
                plan.tasks.append(SearchTask(
                    name=f"tech_{school}_{tech}",
                    query=query,
                    route="tech",
                    priority=3,
                    expected_precision="medium",
                    description=f"技术栈: {school} + {tech}"
                ))

        # 泛化查询
        for course in course_aliases[:2]:
            query = f'"{course}" homework OR lab OR project in:name,description,readme'
            plan.tasks.append(SearchTask(
                name=f"broad_{course}_assignment",
                query=query,
                route="broad",
                priority=4,
                expected_precision="medium",
                description=f"泛化: {course} + 作业关键词"
            ))

    def _build_course_assignment_plan(self, intent: QueryIntent, plan: SearchPlan):
        """课程 + 作业 查询"""
        course_aliases = intent.course_aliases[:3]
        assign_type = intent.assignment_type or "lab"
        assign_num = intent.assignment_number or ""

        # 课程 + 作业精确查询
        for course in course_aliases[:3]:
            query = f'"{course}" {assign_type}'
            if assign_num:
                query += assign_num
            query += ' in:name,description,readme'
            plan.tasks.append(SearchTask(
                name=f"exact_{course}_{assign_type}",
                query=query,
                route="exact",
                priority=1,
                expected_precision="high",
                description=f"精确: {course} + {assign_type}"
            ))

        # 技术栈查询
        techs = self.course_kb.get_technologies(intent.course) if intent.course else []
        for tech in techs[:3]:
            query = f'{tech} {assign_type}'
            if assign_num:
                query += assign_num
            query += ' in:readme'
            plan.tasks.append(SearchTask(
                name=f"tech_{tech}_{assign_type}",
                query=query,
                route="tech",
                priority=2,
                expected_precision="medium",
                description=f"技术栈: {tech} + {assign_type}"
            ))

        # 目录/文件路径查询
        for course in course_aliases[:2]:
            query = f'"{course}" in:path'
            if assign_num:
                query += f' {assign_type}{assign_num}'
            plan.tasks.append(SearchTask(
                name=f"code_path_{course}_{assign_type}",
                query=query,
                route="code_path",
                priority=2,
                expected_precision="medium",
                description=f"路径命中: {course} + {assign_type}"
            ))

    def _build_course_only_plan(self, intent: QueryIntent, plan: SearchPlan):
        """只有课程的查询"""
        course_aliases = intent.course_aliases[:4]

        # 课程名查询
        for course in course_aliases[:2]:
            query = f'"{course}" in:name,description,readme'
            plan.tasks.append(SearchTask(
                name=f"course_{course}",
                query=query,
                route="exact",
                priority=1,
                expected_precision="medium",
                description=f"课程名: {course}"
            ))

        # 课程 + 作业关键词
        assign_keywords = ["homework", "lab", "project", "实验", "作业"]
        for course in course_aliases[:2]:
            for kw in assign_keywords[:3]:
                query = f'"{course}" {kw} in:readme'
                plan.tasks.append(SearchTask(
                    name=f"course_{course}_{kw}",
                    query=query,
                    route="broad",
                    priority=2,
                    expected_precision="medium",
                    description=f"课程 + 作业: {course} + {kw}"
                ))

        # 课程代码查询
        if intent.course:
            course_codes = self.course_kb.get_course_codes(intent.course)
            for code in course_codes[:2]:
                query = f'{code} in:name,readme'
                plan.tasks.append(SearchTask(
                    name=f"code_{code}",
                    query=query,
                    route="code",
                    priority=3,
                    expected_precision="medium",
                    description=f"课程代码: {code}"
                ))

        for course in course_aliases[:2]:
            plan.tasks.append(SearchTask(
                name=f"code_path_{course}",
                query=f'"{course}" in:path',
                route="code_path",
                priority=3,
                expected_precision="medium",
                description=f"路径命中: {course}"
            ))

    def _build_school_only_plan(self, intent: QueryIntent, plan: SearchPlan):
        """只有学校的查询"""
        school_aliases = intent.school_aliases[:3]

        # 如果有从查询中提取的课程关键词，优先使用
        if intent.course_aliases:
            course_keyword = intent.course_aliases[0]  # 提取的关键词
            for school in school_aliases[:2]:
                query = f'"{school}" "{course_keyword}" in:name,description,readme'
                plan.tasks.append(SearchTask(
                    name=f"school_{school}_extracted_{course_keyword}",
                    query=query,
                    route="exact",
                    priority=1,
                    expected_precision="high",
                    description=f"学校 + 课程关键词: {school} + {course_keyword}"
                ))

            # 加作业关键词
            assign_keywords = ["homework", "lab", "project", "实验", "作业"]
            for school in school_aliases[:1]:
                for kw in assign_keywords[:2]:
                    query = f'"{school}" "{course_keyword}" {kw} in:readme'
                    plan.tasks.append(SearchTask(
                        name=f"school_{school}_{course_keyword}_{kw}",
                        query=query,
                        route="broad",
                        priority=2,
                        expected_precision="medium",
                        description=f"学校 + 课程 + 作业: {school} + {course_keyword} + {kw}"
                    ))
            return

        # 学校 + 课程作业关键词
        assign_keywords = ["homework", "lab", "project", "课程", "作业", "实验"]
        for school in school_aliases[:2]:
            for kw in assign_keywords[:3]:
                query = f'"{school}" {kw} in:name,description,readme'
                plan.tasks.append(SearchTask(
                    name=f"school_{school}_{kw}",
                    query=query,
                    route="broad",
                    priority=1,
                    expected_precision="medium",
                    description=f"学校 + 作业: {school} + {kw}"
                ))

        # 学校 + 热门课程
        hot_courses = ["操作系统", "编译原理", "数据结构", "计算机网络", "数据库", "并行程序设计"]
        for school in school_aliases[:1]:
            for course in hot_courses[:3]:
                query = f'"{school}" "{course}" in:readme'
                plan.tasks.append(SearchTask(
                    name=f"school_{school}_course_{course}",
                    query=query,
                    route="exact",
                    priority=2,
                    expected_precision="high",
                    description=f"学校 + 课程: {school} + {course}"
                ))

    def _build_generic_plan(self, intent: QueryIntent, plan: SearchPlan):
        """通用查询"""
        query = intent.raw_query

        # 直接搜索
        plan.tasks.append(SearchTask(
            name="generic_exact",
            query=f'{query} in:name,description,readme',
            route="exact",
            priority=1,
            expected_precision="medium",
            description=f"精确搜索: {query}"
        ))

        # 加作业关键词
        assign_keywords = ["homework", "lab", "project", "实验", "作业"]
        for kw in assign_keywords[:3]:
            plan.tasks.append(SearchTask(
                name=f"generic_{kw}",
                query=f'{query} {kw} in:readme',
                route="broad",
                priority=2,
                expected_precision="low",
                description=f"泛化: {query} + {kw}"
            ))


# 全局实例
_planner = None


def get_search_planner() -> SearchPlanner:
    """获取检索计划器单例"""
    global _planner
    if _planner is None:
        _planner = SearchPlanner()
    return _planner


def build_search_plan(intent: QueryIntent, budget: int = 8) -> SearchPlan:
    """构建检索计划的便捷函数"""
    return get_search_planner().build_plan(intent, budget)
