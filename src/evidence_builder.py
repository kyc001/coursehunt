"""
证据抽取与推荐理由生成模块
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .school_kb import get_school_kb
from .course_kb import get_course_kb
from .matching import contains_signal


@dataclass
class Evidence:
    """单条证据"""
    field: str        # 来源字段: readme/repo_name/description/path/owner_profile/owner_repos
    signal_type: str  # 信号类型: school/course/assignment/technology
    text: str         # 命中文本
    confidence: float = 1.0  # 置信度


@dataclass
class RepoEvidence:
    """仓库证据集合"""
    repo: str
    score: float
    confidence: str  # high/medium/low
    school_evidence: List[Evidence] = field(default_factory=list)
    course_evidence: List[Evidence] = field(default_factory=list)
    assignment_evidence: List[Evidence] = field(default_factory=list)
    technology_evidence: List[Evidence] = field(default_factory=list)
    owner_evidence: List[Evidence] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class EvidenceBuilder:
    """证据构建器"""

    def __init__(self):
        self.school_kb = get_school_kb()
        self.course_kb = get_course_kb()

    def build_evidence(self, repo_data: dict, owner_context: dict,
                       school_id: str = None, course_id: str = None) -> RepoEvidence:
        """
        构建仓库证据

        Args:
            repo_data: 仓库数据
            owner_context: 用户画像数据
            school_id: 目标学校 ID
            course_id: 目标课程 ID

        Returns:
            RepoEvidence 对象
        """
        full_name = repo_data.get("full_name") or ""
        evidence = RepoEvidence(repo=full_name, score=0.0, confidence="low")

        # 提取文本 - 确保不是 None
        name = repo_data.get("name") or ""
        description = repo_data.get("description") or ""
        readme = repo_data.get("readme_text") or ""
        topics = [str(topic) for topic in (repo_data.get("topics") or []) if topic]
        code_path_matches = repo_data.get("code_path_matches") or []
        matched_paths = [
            str(match.get("path") or "")
            for match in code_path_matches
            if match.get("path")
        ]
        tree_paths = [str(path) for path in (repo_data.get("tree_paths") or []) if path]
        path_text = " ".join(matched_paths + tree_paths[:5000])
        all_text = f"{name} {description} {' '.join(topics)} {path_text} {readme}"

        # 1. 学校证据
        if school_id:
            self._extract_school_evidence(evidence, all_text, name, description,
                                          readme, path_text, owner_context, school_id)

        # 2. 课程证据
        if course_id:
            self._extract_course_evidence(evidence, all_text, name, description,
                                          readme, path_text, topics, course_id)

        # 3. 作业证据
        self._extract_assignment_evidence(evidence, all_text, name, readme)

        # 4. 技术栈证据
        if course_id:
            self._extract_technology_evidence(evidence, all_text, readme, course_id)

        # 5. 用户画像证据
        self._extract_owner_evidence(evidence, owner_context, school_id)

        # 6. 生成推荐理由
        self._generate_reasons(evidence, repo_data)

        # 7. 生成风险提示
        self._generate_risks(evidence, repo_data, owner_context)

        # 8. 计算总分和置信度
        self._calculate_confidence(evidence)

        return evidence

    def _extract_school_evidence(self, evidence: RepoEvidence, all_text: str,
                                 name: str, description: str, readme: str,
                                 path_text: str, owner_context: dict, school_id: str):
        """提取学校证据"""
        # 仓库自身信号
        school_check = self.school_kb.check_school_signal(all_text, school_id)
        for signal in school_check["signals"]:
            # 判断来源字段
            if contains_signal(name, signal):
                field = "repo_name"
            elif description and contains_signal(description, signal):
                field = "description"
            elif contains_signal(path_text, signal):
                field = "path"
            elif contains_signal(readme, signal):
                field = "readme"
            else:
                field = "other"

            evidence.school_evidence.append(Evidence(
                field=field,
                signal_type="school",
                text=signal,
                confidence=0.8 if field in ["repo_name", "readme"] else 0.6
            ))

        # 域名信号
        for domain in self.school_kb.get_domains(school_id):
            if contains_signal(all_text, domain):
                evidence.school_evidence.append(Evidence(
                    field="readme",
                    signal_type="school_domain",
                    text=domain,
                    confidence=0.9
                ))

    def _extract_course_evidence(self, evidence: RepoEvidence, all_text: str,
                                 name: str, description: str, readme: str,
                                 path_text: str, topics: list, course_id: str):
        """提取课程证据"""
        course_check = self.course_kb.check_course_signal(all_text, course_id)

        for signal in course_check["signals"]:
            if contains_signal(name, signal):
                field = "repo_name"
            elif description and contains_signal(description, signal):
                field = "description"
            elif contains_signal(path_text, signal):
                field = "path"
            elif contains_signal(readme, signal):
                field = "readme"
            elif any(contains_signal(t, signal) for t in topics):
                field = "topics"
            else:
                field = "other"

            evidence.course_evidence.append(Evidence(
                field=field,
                signal_type="course",
                text=signal,
                confidence=0.9 if field == "repo_name" else 0.7
            ))

        for tech in course_check["tech_signals"]:
            evidence.technology_evidence.append(Evidence(
                field="readme",
                signal_type="technology",
                text=tech,
                confidence=0.6
            ))

    def _extract_assignment_evidence(self, evidence: RepoEvidence, all_text: str,
                                     name: str, readme: str):
        """提取作业证据"""
        import re

        # 作业关键词
        assignment_keywords = {
            "homework": 0.8, "hw": 0.8, "lab": 0.9, "experiment": 0.7,
            "project": 0.6, "assignment": 0.7, "report": 0.7,
            "实验": 0.8, "作业": 0.8, "大作业": 0.7, "课程设计": 0.7
        }

        for keyword, conf in assignment_keywords.items():
            if contains_signal(all_text, keyword):
                field = "repo_name" if contains_signal(name, keyword) else "readme"
                evidence.assignment_evidence.append(Evidence(
                    field=field,
                    signal_type="assignment",
                    text=keyword,
                    confidence=conf
                ))

        # 作业编号模式
        patterns = [
            r'lab\s*(\d+)', r'hw\s*(\d+)', r'homework\s*(\d+)',
            r'experiment\s*(\d+)', r'实验\s*(\d+)', r'作业\s*(\d+)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, all_text.lower())
            for match in matches:
                evidence.assignment_evidence.append(Evidence(
                    field="readme",
                    signal_type="assignment_number",
                    text=f"lab/hw {match}",
                    confidence=0.8
                ))

    def _extract_technology_evidence(self, evidence: RepoEvidence, all_text: str,
                                     readme: str, course_id: str):
        """提取技术栈证据"""
        techs = self.course_kb.get_technologies(course_id)
        for tech in techs:
            if contains_signal(all_text, tech):
                evidence.technology_evidence.append(Evidence(
                    field="readme",
                    signal_type="technology",
                    text=tech,
                    confidence=0.7
                ))

    def _extract_owner_evidence(self, evidence: RepoEvidence, owner_context: dict,
                                school_id: str):
        """提取用户画像证据"""
        if not owner_context:
            return

        school_confidence = owner_context.get("school_confidence", 0)
        if school_confidence > 0.3:
            evidence.owner_evidence.append(Evidence(
                field="owner_profile",
                signal_type="school",
                text=f"作者学校置信度: {school_confidence:.0%}",
                confidence=school_confidence
            ))

        # 用户的其他学校相关仓库
        school_repos = owner_context.get("school_repo_examples", [])
        if school_repos:
            evidence.owner_evidence.append(Evidence(
                field="owner_repos",
                signal_type="school",
                text=f"作者其他学校仓库: {', '.join(school_repos[:3])}",
                confidence=0.7
            ))

    def _generate_reasons(self, evidence: RepoEvidence, repo_data: dict):
        """生成推荐理由"""
        reasons = []

        if repo_data.get("known_collection"):
            reasons.append(repo_data.get("collection_reason") or "已知课程资料合集仓库")

        # 学校理由
        if evidence.school_evidence:
            signals = [e.text for e in evidence.school_evidence[:3]]
            reasons.append(f"仓库中出现学校相关关键词: {', '.join(signals)}")

        # 课程理由
        if evidence.course_evidence:
            signals = [e.text for e in evidence.course_evidence[:3]]
            if any(e.field == "path" for e in evidence.course_evidence):
                reasons.append(f"目录/文件路径命中课程关键词: {', '.join(signals)}")
            else:
                reasons.append(f"匹配课程相关关键词: {', '.join(signals)}")

        # 作业理由
        if evidence.assignment_evidence:
            signals = list(set(e.text for e in evidence.assignment_evidence[:3]))
            reasons.append(f"包含作业相关标识: {', '.join(signals)}")

        # 技术栈理由
        if evidence.technology_evidence:
            signals = [e.text for e in evidence.technology_evidence[:3]]
            reasons.append(f"使用相关技术栈: {', '.join(signals)}")

        # 用户画像理由
        if evidence.owner_evidence:
            for e in evidence.owner_evidence:
                reasons.append(e.text)

        evidence.reasons = reasons if reasons else ["搜索结果中匹配的仓库"]

    def _generate_risks(self, evidence: RepoEvidence, repo_data: dict,
                        owner_context: dict):
        """生成风险提示"""
        risks = []

        # 学校信号来自用户画像而非仓库本身
        has_repo_school = any(e.field != "owner_repos" for e in evidence.school_evidence)
        has_owner_school = any(e.field in ["owner_repos", "owner_profile"] for e in evidence.owner_evidence)

        if not has_repo_school and has_owner_school:
            risks.append("学校信号来自作者其他仓库，非当前仓库")

        # 星数过高可能是通用项目
        stars = repo_data.get("stargazers_count", 0)
        if stars > 100:
            risks.append(f"仓库 star 数较高({stars})，可能是通用开源项目而非课程作业")

        # 单一关键词命中
        if len(evidence.course_evidence) == 1 and len(evidence.school_evidence) == 0:
            risks.append("仅命中单一课程关键词，相关性可能较低")

        evidence.risks = risks

    def _calculate_confidence(self, evidence: RepoEvidence):
        """计算置信度"""
        score = 0.0

        # 学校证据得分
        school_score = sum(e.confidence for e in evidence.school_evidence) * 0.25
        score += min(school_score, 0.25)

        # 课程证据得分
        course_score = sum(e.confidence for e in evidence.course_evidence) * 0.25
        score += min(course_score, 0.25)

        # 作业证据得分
        assignment_score = sum(e.confidence for e in evidence.assignment_evidence) * 0.15
        score += min(assignment_score, 0.15)

        # 用户画像得分
        owner_score = sum(e.confidence for e in evidence.owner_evidence) * 0.20
        score += min(owner_score, 0.20)

        # 技术栈得分
        tech_score = sum(e.confidence for e in evidence.technology_evidence) * 0.10
        score += min(tech_score, 0.10)

        evidence.score = round(score, 3)

        # 置信度等级
        if score >= 0.6:
            evidence.confidence = "high"
        elif score >= 0.3:
            evidence.confidence = "medium"
        else:
            evidence.confidence = "low"


# 全局实例
_evidence_builder = None


def get_evidence_builder() -> EvidenceBuilder:
    """获取证据构建器单例"""
    global _evidence_builder
    if _evidence_builder is None:
        _evidence_builder = EvidenceBuilder()
    return _evidence_builder
