"""
课程知识库模块
加载和查询课程信息
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .matching import contains_signal


class CourseKB:
    """课程知识库"""

    def __init__(self, yaml_path: str = None):
        self.courses = {}
        self.alias_index = {}  # alias -> course_id
        self.code_index = {}   # course_code -> course_id

        self._load(yaml_path or Path(__file__).parent / "data" / "courses.yaml")

    def _load(self, yaml_path: str):
        """加载 YAML 文件"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"课程知识库文件不存在: {yaml_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        for course_id, info in data.items():
            self.courses[course_id] = info

            # 建立别名索引
            canonical = info.get('canonical_name', '')
            if canonical:
                self.alias_index[canonical.lower()] = course_id

            for alias in info.get('aliases', []):
                self.alias_index[alias.lower()] = course_id

            for name in info.get('english_names', []):
                self.alias_index[name.lower()] = course_id

            # 建立课程代码索引
            for code in info.get('course_codes', []):
                self.code_index[code.upper()] = course_id

    def get_course(self, course_id: str) -> Optional[Dict]:
        """获取课程信息"""
        return self.courses.get(course_id)

    def find_course_by_alias(self, alias: str) -> Optional[str]:
        """通过别名查找课程 ID"""
        return self.alias_index.get(alias.lower())

    def find_course_by_code(self, code: str) -> Optional[str]:
        """通过课程代码查找课程 ID"""
        return self.code_index.get(code.upper())

    def get_all_aliases(self, course_id: str) -> List[str]:
        """获取课程的所有别名"""
        course = self.courses.get(course_id)
        if not course:
            return []

        aliases = [course.get('canonical_name', '')]
        aliases.extend(course.get('aliases', []))
        aliases.extend(course.get('english_names', []))
        return _dedupe_keep_order([a for a in aliases if a])

    def get_technologies(self, course_id: str) -> List[str]:
        """获取课程相关技术栈"""
        course = self.courses.get(course_id)
        if not course:
            return []
        return course.get('technologies', [])

    def get_assignment_keywords(self, course_id: str) -> List[str]:
        """获取课程作业关键词"""
        course = self.courses.get(course_id)
        if not course:
            return []
        return course.get('assignment_keywords', [])

    def get_file_patterns(self, course_id: str) -> List[str]:
        """获取课程常见文件模式"""
        course = self.courses.get(course_id)
        if not course:
            return []
        return course.get('common_file_patterns', [])

    def get_negative_keywords(self, course_id: str) -> List[str]:
        """获取课程负向关键词"""
        course = self.courses.get(course_id)
        if not course:
            return []
        return course.get('negative_keywords', [])

    def get_course_codes(self, course_id: str) -> List[str]:
        """获取课程代码列表"""
        course = self.courses.get(course_id)
        if not course:
            return []
        return course.get('course_codes', [])

    def check_course_signal(self, text: str, course_id: str) -> Dict:
        """
        检查文本中的课程信号

        Returns:
            {
                "found": bool,
                "signals": list,
                "tech_signals": list,
                "confidence": float
            }
        """
        signals = []
        tech_signals = []

        # 检查课程别名
        for alias in self.get_all_aliases(course_id):
            if contains_signal(text, alias):
                signals.append(alias)

        # 检查课程代码
        for code in self.get_course_codes(course_id):
            if contains_signal(text, code):
                signals.append(code)

        # 检查技术栈
        for tech in self.get_technologies(course_id):
            if contains_signal(text, tech):
                tech_signals.append(tech)

        signals = _dedupe_keep_order(signals)
        tech_signals = _dedupe_keep_order(tech_signals)

        confidence = 0.0
        if signals:
            confidence = min(len(signals) * 0.25, 0.7)
        if tech_signals:
            confidence = min(confidence + len(tech_signals) * 0.1, 1.0)

        return {
            "found": len(signals) > 0 or len(tech_signals) > 0,
            "signals": signals,
            "tech_signals": tech_signals,
            "confidence": confidence
        }

    def search_courses(self, query: str) -> List[str]:
        """搜索匹配的课程 ID"""
        matched = []

        for course_id, info in self.courses.items():
            # 检查所有别名
            all_aliases = self.get_all_aliases(course_id)
            for alias in all_aliases:
                if contains_signal(alias, query) or contains_signal(query, alias):
                    matched.append(course_id)
                    break

        return matched


# 全局实例
_course_kb = None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
            if item is None:
                continue
            key = str(item).lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(str(item))
    return output


def get_course_kb() -> CourseKB:
    """获取课程知识库单例"""
    global _course_kb
    if _course_kb is None:
        _course_kb = CourseKB()
    return _course_kb
