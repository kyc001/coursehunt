"""
仓库内容分析模块
从仓库元数据和 README 中抽取课程相关信号
"""

import re
from typing import Dict, List, Set

from .matching import contains_signal


class RepoAnalyzer:
    """仓库内容分析器"""

    def __init__(self):
        # 学校关键词
        self.school_terms = {
            "nku", "nankai", "南开", "南开大学", "nankai university",
            "tsinghua", "清华", "pku", "北大", "peking"
        }

        # 作业关键词
        self.assignment_terms = {
            "homework", "hw", "lab", "experiment", "实验", "作业",
            "project", "assignment", "report", "报告", "大作业",
            "课程设计", "course design", "final project", "cw",
            "coursework", "tutorial", "exercise"
        }

        # 学生/课程相关关键词
        self.course_terms = {
            "course", "class", "student", "学生", "课程", "班级",
            "学号", "学期", "semester", "fall", "spring", "autumn",
            "2020", "2021", "2022", "2023", "2024", "2025", "2026"
        }

        # 代码文件扩展名
        self.code_extensions = {
            ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".js", ".ts",
            ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt"
        }

        # 报告文件扩展名
        self.report_extensions = {".pdf", ".doc", ".docx", ".md", ".tex"}

    def analyze_repo(self, repo_data: dict, readme_content: str = "") -> dict:
        """
        分析仓库，抽取课程相关信号

        Args:
            repo_data: GitHub API 返回的仓库数据
            readme_content: README 内容

        Returns:
            分析结果字典
        """
        # 提取基本信息
        name = repo_data.get("name", "").lower()
        full_name = repo_data.get("full_name", "")
        description = (repo_data.get("description") or "").lower()
        topics = [t.lower() for t in repo_data.get("topics", [])]
        language = repo_data.get("language", "")
        stars = repo_data.get("stargazers_count", 0)
        forks = repo_data.get("forks_count", 0)
        updated_at = repo_data.get("updated_at", "")
        owner = repo_data.get("owner", {}).get("login", "")

        # README 内容小写化
        readme_lower = readme_content.lower() if readme_content else ""

        # 合并所有文本用于分析
        all_text = f"{name} {description} {' '.join(topics)} {readme_lower}"

        # 抽取信号
        signals = {
            "school_signals": self._extract_school_signals(all_text),
            "assignment_signals": self._extract_assignment_signals(all_text),
            "course_signals": self._extract_course_signals(all_text),
            "has_report": self._check_has_report(readme_content, name),
            "has_code": self._check_has_code(language),
            "structure_hints": self._extract_structure_hints(readme_content)
        }

        # 计算各项分数
        scores = {
            "school_score": len(signals["school_signals"]) / 3.0,  # 归一化
            "assignment_score": len(signals["assignment_signals"]) / 4.0,
            "course_score": len(signals["course_signals"]) / 3.0,
            "quality_score": self._calculate_quality_score(stars, forks, readme_content),
            "freshness_score": self._calculate_freshness_score(updated_at)
        }

        return {
            "repo": full_name,
            "name": name,
            "description": repo_data.get("description", ""),
            "language": language,
            "stars": stars,
            "forks": forks,
            "updated_at": updated_at,
            "owner": owner,
            "topics": topics,
            "signals": signals,
            "scores": scores,
            "readme_preview": readme_content[:500] if readme_content else ""
        }

    def _extract_school_signals(self, text: str) -> List[str]:
        """提取学校相关信号"""
        found = []
        for term in self.school_terms:
            if contains_signal(text, term):
                found.append(term)
        return found

    def _extract_assignment_signals(self, text: str) -> List[str]:
        """提取作业相关信号"""
        found = []
        for term in self.assignment_terms:
            if contains_signal(text, term):
                found.append(term)
        return found

    def _extract_course_signals(self, text: str) -> List[str]:
        """提取课程相关信号"""
        found = []
        for term in self.course_terms:
            if contains_signal(text, term):
                found.append(term)

        # 提取课程代码 (如 COSC1001)
        code_pattern = r'[a-z]{2,4}\d{3,4}'
        codes = re.findall(code_pattern, text)
        found.extend(codes[:3])  # 最多取3个

        return found

    def _check_has_report(self, readme: str, name: str) -> bool:
        """检查是否包含报告"""
        report_indicators = ["report", "报告", "pdf", ".doc", "实验报告"]
        text = f"{readme} {name}".lower()
        return any(ind in text for ind in report_indicators)

    def _check_has_code(self, language: str) -> bool:
        """检查是否包含代码"""
        return bool(language)

    def _extract_structure_hints(self, readme: str) -> List[str]:
        """从 README 提取目录结构提示"""
        hints = []
        if not readme:
            return hints

        # 匹配类似 lab1, lab2, hw1, homework1 的模式
        lab_pattern = r'(?:lab|hw|homework|实验|作业)\s*\d+'
        matches = re.findall(lab_pattern, readme.lower())
        hints.extend(matches[:5])

        return hints

    def _calculate_quality_score(self, stars: int, forks: int, readme: str) -> float:
        """计算质量分数"""
        score = 0.0

        # Star 贡献
        if stars >= 10:
            score += 0.4
        elif stars >= 5:
            score += 0.3
        elif stars >= 1:
            score += 0.2

        # Fork 贡献
        if forks >= 5:
            score += 0.2
        elif forks >= 1:
            score += 0.1

        # README 长度贡献
        if len(readme) > 1000:
            score += 0.3
        elif len(readme) > 200:
            score += 0.2
        elif len(readme) > 0:
            score += 0.1

        return min(score, 1.0)

    def _calculate_freshness_score(self, updated_at: str) -> float:
        """计算新鲜度分数"""
        if not updated_at:
            return 0.0

        try:
            from datetime import datetime
            update_time = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(update_time.tzinfo)
            days_old = (now - update_time).days

            if days_old < 180:  # 6个月内
                return 1.0
            elif days_old < 365:  # 1年内
                return 0.8
            elif days_old < 730:  # 2年内
                return 0.6
            elif days_old < 1095:  # 3年内
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.5

    def analyze_owner_context(self, owner_repos: list) -> dict:
        """
        分析仓库所有者的其他仓库，推断学校背景

        Args:
            owner_repos: 该用户的所有公开仓库列表

        Returns:
            用户画像分析结果
        """
        school_mentions = 0
        course_mentions = 0
        total_repos = len(owner_repos)

        school_repo_examples = []
        course_repo_examples = []

        for repo in owner_repos:
            name = repo.get("name", "").lower()
            desc = (repo.get("description") or "").lower()
            topics = [t.lower() for t in repo.get("topics", [])]
            text = f"{name} {desc} {' '.join(topics)}"

            # 检查学校信号
            for term in self.school_terms:
                if contains_signal(text, term):
                    school_mentions += 1
                    if len(school_repo_examples) < 3:
                        school_repo_examples.append(repo.get("full_name", ""))
                    break

            # 检查课程信号
            for term in self.assignment_terms:
                if contains_signal(text, term):
                    course_mentions += 1
                    if len(course_repo_examples) < 3:
                        course_repo_examples.append(repo.get("full_name", ""))
                    break

        # 计算用户学校置信度
        if total_repos > 0:
            school_confidence = min(school_mentions / max(total_repos * 0.3, 1), 1.0)
        else:
            school_confidence = 0.0

        return {
            "total_repos": total_repos,
            "school_mentions": school_mentions,
            "course_mentions": course_mentions,
            "school_confidence": school_confidence,
            "school_repo_examples": school_repo_examples,
            "course_repo_examples": course_repo_examples
        }
