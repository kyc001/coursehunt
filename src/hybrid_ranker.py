"""
混合排序模块
整合多种信号进行排序
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .query_parser import QueryIntent
from .evidence_builder import RepoEvidence
from .matching import contains_signal


@dataclass
class RankerConfig:
    """排序器配置"""
    # 不同意图类型的权重配置
    intent_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "specific_school_course_assignment": {
            "course": 0.20,
            "school": 0.25,
            "assignment": 0.20,
            "owner_school": 0.15,
            "evidence": 0.10,
            "quality": 0.05,
            "freshness": 0.03,
            "rrf": 0.02
        },
        "school_course": {
            "course": 0.25,
            "school": 0.25,
            "assignment": 0.10,
            "owner_school": 0.20,
            "evidence": 0.10,
            "quality": 0.05,
            "freshness": 0.03,
            "rrf": 0.02
        },
        "course_assignment": {
            "course": 0.30,
            "school": 0.05,
            "assignment": 0.25,
            "owner_school": 0.05,
            "evidence": 0.15,
            "quality": 0.10,
            "freshness": 0.05,
            "rrf": 0.05
        },
        "course_only": {
            "course": 0.35,
            "school": 0.03,
            "assignment": 0.15,
            "owner_school": 0.02,
            "evidence": 0.15,
            "quality": 0.15,
            "freshness": 0.10,
            "rrf": 0.05
        },
        "school_only": {
            "course": 0.10,
            "school": 0.30,
            "assignment": 0.10,
            "owner_school": 0.25,
            "evidence": 0.10,
            "quality": 0.08,
            "freshness": 0.05,
            "rrf": 0.02
        },
        "generic": {
            "course": 0.15,
            "school": 0.10,
            "assignment": 0.15,
            "owner_school": 0.10,
            "evidence": 0.15,
            "quality": 0.15,
            "freshness": 0.10,
            "rrf": 0.10
        }
    })

    # 负向特征惩罚
    negative_penalties: Dict[str, float] = field(default_factory=lambda: {
        "generic_tutorial": 0.25,
        "unrelated_popular": 0.20,
        "weak_single_hit": 0.15,
        "school_ambiguity": 0.15,
        "fork_no_change": 0.10
    })


class HybridRanker:
    """混合排序器"""

    def __init__(self, config: RankerConfig = None):
        self.config = config or RankerConfig()

    def rank(self, candidates: List[dict], intent: QueryIntent) -> List[dict]:
        """
        对候选仓库进行排序

        Args:
            candidates: 候选仓库列表，每个元素包含:
                - repo_data: 仓库原始数据
                - evidence: RepoEvidence 对象
                - rrf_score: RRF 分数
                - owner_context: 用户画像
            intent: 查询意图

        Returns:
            排序后的候选列表
        """
        # 获取当前意图的权重配置
        weights = self.config.intent_weights.get(
            intent.intent_type,
            self.config.intent_weights["generic"]
        )

        # 计算每个候选的最终分数
        for candidate in candidates:
            score = self._calculate_score(candidate, intent, weights)
            candidate["final_score"] = score

        # 按最终分数降序排序
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return candidates

    def _calculate_score(self, candidate: dict, intent: QueryIntent,
                         weights: Dict[str, float]) -> float:
        """计算候选的最终分数"""
        evidence = candidate.get("evidence")
        repo_data = candidate.get("repo_data", {})
        owner_context = candidate.get("owner_context", {})
        rrf_score = candidate.get("rrf_score", 0)

        # 1. 课程匹配分数
        course_score = self._calc_course_score(evidence, intent)

        # 2. 学校匹配分数
        school_score = self._calc_school_score(evidence, owner_context, intent)

        # 3. 作业匹配分数
        assignment_score = self._calc_assignment_score(evidence, intent)

        # 4. 用户画像分数
        owner_score = self._calc_owner_score(owner_context, intent)

        # 5. 证据质量分数
        evidence_score = self._calc_evidence_score(evidence)

        # 6. 仓库质量分数
        quality_score = self._calc_quality_score(repo_data)

        # 7. 新鲜度分数
        freshness_score = self._calc_freshness_score(repo_data)

        # 8. RRF 分数 (归一化)
        rrf_normalized = min(rrf_score * 10, 1.0) if rrf_score else 0

        # 加权求和
        total = (
            weights["course"] * course_score +
            weights["school"] * school_score +
            weights["assignment"] * assignment_score +
            weights["owner_school"] * owner_score +
            weights["evidence"] * evidence_score +
            weights["quality"] * quality_score +
            weights["freshness"] * freshness_score +
            weights["rrf"] * rrf_normalized
        )

        # 减去负向惩罚
        penalty = self._calculate_penalty(candidate, intent)
        total = max(0, total - penalty)

        return round(total, 4)

    def _calc_course_score(self, evidence: RepoEvidence, intent: QueryIntent) -> float:
        """计算课程匹配分数"""
        if not evidence:
            return 0.0

        score = 0.0
        course_ev = evidence.course_evidence

        if course_ev:
            # 有课程证据
            max_conf = max(e.confidence for e in course_ev)
            score = max_conf

            # 如果查询有课程，检查是否匹配
            if intent.course:
                for ev in course_ev:
                    if any(contains_signal(alias, ev.text) or contains_signal(ev.text, alias)
                           for alias in intent.course_aliases):
                        score = min(score + 0.2, 1.0)
                        break

        return min(score, 1.0)

    def _calc_school_score(self, evidence: RepoEvidence, owner_context: dict,
                           intent: QueryIntent) -> float:
        """计算学校匹配分数"""
        if not evidence:
            return 0.0

        score = 0.0

        # 仓库自身的学校信号
        school_ev = evidence.school_evidence
        if school_ev:
            max_conf = max(e.confidence for e in school_ev)
            score = max_conf * 0.7

        # 用户画像的学校信号
        owner_conf = owner_context.get("school_confidence", 0)
        if owner_conf > 0.5:
            score = max(score, owner_conf * 0.5)

        return min(score, 1.0)

    def _calc_assignment_score(self, evidence: RepoEvidence, intent: QueryIntent) -> float:
        """计算作业匹配分数"""
        if not evidence:
            return 0.0

        score = 0.0
        assign_ev = evidence.assignment_evidence

        if assign_ev:
            max_conf = max(e.confidence for e in assign_ev)
            score = max_conf

            # 如果查询有作业编号，检查是否匹配
            if intent.assignment_number:
                for ev in assign_ev:
                    if intent.assignment_number in ev.text:
                        score = min(score + 0.2, 1.0)
                        break

        return min(score, 1.0)

    def _calc_owner_score(self, owner_context: dict, intent: QueryIntent) -> float:
        """计算用户画像分数"""
        if not owner_context:
            return 0.0

        school_conf = owner_context.get("school_confidence", 0)
        return min(school_conf, 1.0)

    def _calc_evidence_score(self, evidence: RepoEvidence) -> float:
        """计算证据质量分数"""
        if not evidence:
            return 0.0

        # 证据多样性
        evidence_types = 0
        if evidence.school_evidence:
            evidence_types += 1
        if evidence.course_evidence:
            evidence_types += 1
        if evidence.assignment_evidence:
            evidence_types += 1
        if evidence.technology_evidence:
            evidence_types += 1
        if evidence.owner_evidence:
            evidence_types += 1

        # 多样性得分
        diversity_score = min(evidence_types / 4.0, 1.0)

        # 平均置信度
        all_evidence = (
            evidence.school_evidence +
            evidence.course_evidence +
            evidence.assignment_evidence +
            evidence.technology_evidence +
            evidence.owner_evidence
        )
        avg_confidence = (
            sum(e.confidence for e in all_evidence) / len(all_evidence)
            if all_evidence else 0
        )

        return (diversity_score * 0.4 + avg_confidence * 0.6)

    def _calc_quality_score(self, repo_data: dict) -> float:
        """计算仓库质量分数"""
        score = 0.0

        # Star
        stars = repo_data.get("stargazers_count", 0)
        if stars >= 20:
            score += 0.3
        elif stars >= 10:
            score += 0.2
        elif stars >= 5:
            score += 0.15
        elif stars >= 1:
            score += 0.1

        # Fork
        forks = repo_data.get("forks_count", 0)
        if forks >= 10:
            score += 0.2
        elif forks >= 5:
            score += 0.15
        elif forks >= 1:
            score += 0.1

        # README 长度
        readme_len = len(repo_data.get("readme_text", ""))
        if readme_len > 2000:
            score += 0.3
        elif readme_len > 500:
            score += 0.2
        elif readme_len > 100:
            score += 0.1

        # Topics
        topics = repo_data.get("topics", [])
        if len(topics) >= 3:
            score += 0.15
        elif len(topics) >= 1:
            score += 0.05

        return min(score, 1.0)

    def _calc_freshness_score(self, repo_data: dict) -> float:
        """计算新鲜度分数"""
        from datetime import datetime

        pushed_at = repo_data.get("pushed_at", "")
        if not pushed_at:
            return 0.3

        try:
            push_time = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
            now = datetime.now(push_time.tzinfo)
            days_old = (now - push_time).days

            if days_old < 180:
                return 1.0
            elif days_old < 365:
                return 0.8
            elif days_old < 730:
                return 0.6
            elif days_old < 1095:
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.3

    def _calculate_penalty(self, candidate: dict, intent: QueryIntent) -> float:
        """计算负向惩罚"""
        penalty = 0.0
        repo_data = candidate.get("repo_data", {})
        evidence = candidate.get("evidence")

        if not evidence:
            return 0.0

        # 1. 通用教程惩罚
        name = repo_data.get("name", "").lower()
        desc = (repo_data.get("description") or "").lower()
        tutorial_words = ["tutorial", "guide", "notes", "awesome", "learning", "教程", "笔记"]
        if any(w in name or w in desc for w in tutorial_words):
            penalty += self.config.negative_penalties["generic_tutorial"]

        # 2. 高 star 通用项目惩罚
        stars = repo_data.get("stargazers_count", 0)
        if stars > 100 and len(evidence.course_evidence) <= 1:
            penalty += self.config.negative_penalties["unrelated_popular"]

        # 3. 单一关键词命中惩罚
        total_evidence = (
            len(evidence.school_evidence) +
            len(evidence.course_evidence) +
            len(evidence.assignment_evidence)
        )
        if total_evidence <= 1:
            penalty += self.config.negative_penalties["weak_single_hit"]

        # 4. 学校歧义惩罚
        if evidence.school_evidence:
            for ev in evidence.school_evidence:
                if ev.confidence < 0.5:
                    penalty += self.config.negative_penalties["school_ambiguity"]
                    break

        # 5. Fork 无修改惩罚
        if repo_data.get("fork", False):
            parent = repo_data.get("parent", {})
            if parent:
                # 比较更新时间判断是否有实质修改
                penalty += self.config.negative_penalties["fork_no_change"] * 0.5

        return penalty


# 全局实例
_ranker = None


def get_hybrid_ranker() -> HybridRanker:
    """获取混合排序器单例"""
    global _ranker
    if _ranker is None:
        _ranker = HybridRanker()
    return _ranker
