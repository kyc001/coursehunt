"""
学校知识库模块
加载和查询学校信息
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .matching import contains_signal


class SchoolKB:
    """学校知识库"""

    def __init__(self, yaml_path: str = None):
        self.schools = {}
        self.alias_index = {}  # alias -> school_id
        self.domain_index = {}  # domain -> school_id
        self.org_index = {}  # org -> school_id

        self._load(yaml_path or Path(__file__).parent / "data" / "schools.yaml")

    def _load(self, yaml_path: str):
        """加载 YAML 文件"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"学校知识库文件不存在: {yaml_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        for school_id, info in data.items():
            self.schools[school_id] = info

            # 建立别名索引
            canonical = info.get('canonical_name', '')
            if canonical:
                self.alias_index[canonical.lower()] = school_id

            for alias in info.get('aliases', []):
                self.alias_index[alias.lower()] = school_id

            for name in info.get('english_names', []):
                self.alias_index[name.lower()] = school_id

            # 建立域名索引
            for domain in info.get('domains', []):
                self.domain_index[domain.lower()] = school_id

            # 建立 org 索引
            for org in info.get('github_orgs', []):
                self.org_index[org.lower()] = school_id

    def get_school(self, school_id: str) -> Optional[Dict]:
        """获取学校信息"""
        return self.schools.get(school_id)

    def find_school_by_alias(self, alias: str) -> Optional[str]:
        """通过别名查找学校 ID"""
        return self.alias_index.get(alias.lower())

    def find_school_by_domain(self, domain: str) -> Optional[str]:
        """通过邮箱域名查找学校"""
        return self.domain_index.get(domain.lower())

    def find_school_by_org(self, org: str) -> Optional[str]:
        """通过 GitHub org 查找学校"""
        return self.org_index.get(org.lower())

    def get_all_aliases(self, school_id: str) -> List[str]:
        """获取学校的所有别名"""
        school = self.schools.get(school_id)
        if not school:
            return []

        aliases = [school.get('canonical_name', '')]
        aliases.extend(school.get('aliases', []))
        aliases.extend(school.get('english_names', []))
        return _dedupe_keep_order([a for a in aliases if a])

    def get_negative_aliases(self, school_id: str) -> List[str]:
        """获取学校的负向别名（歧义词）"""
        school = self.schools.get(school_id)
        if not school:
            return []
        return school.get('negative_aliases', [])

    def get_domains(self, school_id: str) -> List[str]:
        """获取学校的邮箱域名"""
        school = self.schools.get(school_id)
        if not school:
            return []
        return school.get('domains', [])

    def get_github_orgs(self, school_id: str) -> List[str]:
        """获取学校的 GitHub org"""
        school = self.schools.get(school_id)
        if not school:
            return []
        return school.get('github_orgs', [])

    def get_course_code_prefixes(self, school_id: str) -> List[str]:
        """获取学校的课程代码前缀"""
        school = self.schools.get(school_id)
        if not school:
            return []
        return school.get('course_code_prefix', [])

    def get_seed_repositories(self, school_id: str) -> List[str]:
        """获取人工维护的学校课程资料种子仓库"""
        school = self.schools.get(school_id)
        if not school:
            return []
        return school.get('seed_repositories', [])

    def get_seed_users(self, school_id: str) -> List[str]:
        """获取人工维护的学校课程资料种子用户"""
        school = self.schools.get(school_id)
        if not school:
            return []
        return school.get('seed_users', [])

    def get_seed_repository_paths(self, school_id: str, full_name: str) -> List[str]:
        """获取人工维护的种子仓库课程目录提示"""
        school = self.schools.get(school_id)
        if not school:
            return []
        path_map = school.get('seed_repository_paths', {})
        return path_map.get(full_name, [])

    def check_school_signal(self, text: str, school_id: str) -> Dict:
        """
        检查文本中的学校信号

        Returns:
            {
                "found": bool,
                "signals": list,
                "confidence": float,
                "is_ambiguous": bool
            }
        """
        signals = []
        negative_hits = []

        # 检查正向信号
        for alias in self.get_all_aliases(school_id):
            if contains_signal(text, alias):
                signals.append(alias)

        # 检查负向信号（歧义）
        for neg in self.get_negative_aliases(school_id):
            if contains_signal(text, neg):
                negative_hits.append(neg)

        # 检查域名
        for domain in self.get_domains(school_id):
            if contains_signal(text, domain):
                signals.append(domain)

        # 检查 org
        for org in self.get_github_orgs(school_id):
            if contains_signal(text, org):
                signals.append(org)

        signals = _dedupe_keep_order(signals)
        negative_hits = _dedupe_keep_order(negative_hits)

        confidence = 0.0
        if signals:
            confidence = min(len(signals) * 0.3, 1.0)
            if negative_hits:
                confidence *= 0.5  # 有歧义词，降低置信度

        return {
            "found": len(signals) > 0,
            "signals": signals,
            "confidence": confidence,
            "is_ambiguous": len(negative_hits) > 0,
            "negative_hits": negative_hits
        }


# 全局实例
_school_kb = None


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


def get_school_kb() -> SchoolKB:
    """获取学校知识库单例"""
    global _school_kb
    if _school_kb is None:
        _school_kb = SchoolKB()
    return _school_kb
