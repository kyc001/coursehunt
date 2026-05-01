"""
课程相关性评分模块
综合多维度信号计算仓库的课程相关性分数
"""

from typing import Dict, List


class RelevanceScorer:
    """课程相关性评分器"""

    # 各维度权重
    WEIGHTS = {
        "course": 0.25,      # 课程匹配度
        "school": 0.20,      # 学校匹配度
        "assignment": 0.15,  # 作业属性
        "owner": 0.20,       # 用户画像
        "quality": 0.10,     # 质量
        "freshness": 0.10    # 新鲜度
    }

    def calculate_relevance_score(self, repo_analysis: dict, owner_context: dict,
                                   query_course: str, query_school: str) -> dict:
        """
        计算仓库的综合相关性分数

        Args:
            repo_analysis: RepoAnalyzer.analyze_repo 的返回结果
            owner_context: RepoAnalyzer.analyze_owner_context 的返回结果
            query_course: 用户查询的课程名
            query_school: 用户查询的学校名

        Returns:
            包含总分和各维度分数的字典
        """
        scores = repo_analysis.get("scores", {})
        signals = repo_analysis.get("signals", {})

        # 课程匹配分数
        course_score = self._calculate_course_match_score(
            scores.get("course_score", 0),
            signals.get("course_signals", []),
            query_course
        )

        # 学校匹配分数
        school_score = self._calculate_school_match_score(
            scores.get("school_score", 0),
            signals.get("school_signals", []),
            owner_context.get("school_confidence", 0)
        )

        # 作业属性分数
        assignment_score = min(scores.get("assignment_score", 0), 1.0)

        # 用户画像分数
        owner_score = owner_context.get("school_confidence", 0)

        # 质量分数
        quality_score = scores.get("quality_score", 0)

        # 新鲜度分数
        freshness_score = scores.get("freshness_score", 0)

        # 计算加权总分
        total_score = (
            self.WEIGHTS["course"] * course_score +
            self.WEIGHTS["school"] * school_score +
            self.WEIGHTS["assignment"] * assignment_score +
            self.WEIGHTS["owner"] * owner_score +
            self.WEIGHTS["quality"] * quality_score +
            self.WEIGHTS["freshness"] * freshness_score
        )

        return {
            "total_score": round(total_score, 3),
            "breakdown": {
                "course": round(course_score, 3),
                "school": round(school_score, 3),
                "assignment": round(assignment_score, 3),
                "owner": round(owner_score, 3),
                "quality": round(quality_score, 3),
                "freshness": round(freshness_score, 3)
            }
        }

    def _calculate_course_match_score(self, base_score: float,
                                       course_signals: List[str],
                                       query_course: str) -> float:
        """计算课程匹配分数"""
        score = base_score

        # 检查查询课程名是否在信号中
        query_lower = query_course.lower()
        for signal in course_signals:
            if query_lower in signal.lower() or signal.lower() in query_lower:
                score = min(score + 0.3, 1.0)
                break

        return min(score, 1.0)

    def _calculate_school_match_score(self, base_score: float,
                                       school_signals: List[str],
                                       owner_confidence: float) -> float:
        """计算学校匹配分数"""
        # 如果仓库本身有学校信号
        if school_signals:
            return min(base_score + 0.3, 1.0)

        # 如果 owner 有学校背景
        if owner_confidence > 0.5:
            return owner_confidence * 0.8

        return base_score

    def generate_recommendation_reasons(self, repo_analysis: dict,
                                         owner_context: dict,
                                         relevance_scores: dict) -> List[str]:
        """
        生成推荐理由

        Args:
            repo_analysis: 仓库分析结果
            owner_context: 用户画像分析结果
            relevance_scores: 相关性分数

        Returns:
            推荐理由列表
        """
        reasons = []
        signals = repo_analysis.get("signals", {})
        breakdown = relevance_scores.get("breakdown", {})

        # 学校信号理由
        school_signals = signals.get("school_signals", [])
        if school_signals:
            reasons.append(f"仓库中出现学校相关关键词: {', '.join(school_signals[:3])}")

        # 课程信号理由
        course_signals = signals.get("course_signals", [])
        if course_signals:
            reasons.append(f"匹配课程相关关键词: {', '.join(course_signals[:3])}")

        # 作业属性理由
        assignment_signals = signals.get("assignment_signals", [])
        if assignment_signals:
            reasons.append(f"包含作业相关标识: {', '.join(assignment_signals[:3])}")

        # 用户画像理由
        if owner_context.get("school_confidence", 0) > 0.5:
            examples = owner_context.get("school_repo_examples", [])
            if examples:
                reasons.append(f"仓库作者的其他仓库包含学校相关项目: {', '.join(examples[:2])}")

        # 结构提示理由
        structure_hints = signals.get("structure_hints", [])
        if structure_hints:
            reasons.append(f"目录结构包含: {', '.join(structure_hints[:3])}")

        # 质量理由
        stars = repo_analysis.get("stars", 0)
        if stars >= 10:
            reasons.append(f"获得 {stars} 个 star，质量较高")

        # 报告理由
        if signals.get("has_report"):
            reasons.append("包含实验报告或文档")

        # 没有明确信号时的通用理由
        if not reasons:
            if breakdown.get("quality", 0) > 0.3:
                reasons.append("仓库质量较好，README 完整")
            if breakdown.get("freshness", 0) > 0.5:
                reasons.append("仓库更新较新")

        return reasons if reasons else ["搜索结果中匹配的仓库"]

    def get_confidence_level(self, total_score: float) -> str:
        """获取置信度等级"""
        if total_score >= 0.7:
            return "高"
        elif total_score >= 0.4:
            return "中"
        else:
            return "低"

    def format_score_display(self, scores: dict) -> dict:
        """格式化分数用于显示"""
        breakdown = scores.get("breakdown", {})
        return {
            "总分": f"{scores.get('total_score', 0):.1%}",
            "课程匹配": f"{breakdown.get('course', 0):.0%}",
            "学校匹配": f"{breakdown.get('school', 0):.0%}",
            "作业属性": f"{breakdown.get('assignment', 0):.0%}",
            "用户画像": f"{breakdown.get('owner', 0):.0%}",
            "质量": f"{breakdown.get('quality', 0):.0%}",
            "新鲜度": f"{breakdown.get('freshness', 0):.0%}"
        }
