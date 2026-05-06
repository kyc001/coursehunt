"""
用户画像模块
分析 GitHub 用户的学校背景
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .school_kb import get_school_kb
from .matching import contains_signal


@dataclass
class OwnerProfile:
    """用户画像"""
    login: str
    school_confidence: float = 0.0
    school_signals: List[str] = field(default_factory=list)
    school_repo_examples: List[str] = field(default_factory=list)
    course_repo_examples: List[str] = field(default_factory=list)
    total_repos: int = 0
    profile_signals: Dict = field(default_factory=dict)


class OwnerProfiler:
    """用户画像分析器"""

    def __init__(self, github_client=None):
        self.github = github_client
        self.school_kb = get_school_kb()

    def profile(self, username: str, school_id: str) -> dict:
        """
        分析单个用户的学校背景

        Args:
            username: GitHub 用户名
            school_id: 目标学校 ID

        Returns:
            用户画像字典
        """
        if not self.github:
            return {"school_confidence": 0, "total_repos": 0}

        # 获取用户资料
        profile = self.github.get_user_profile(username, use_cache=True)

        # 获取用户仓库
        repos = self.github.get_user_repos(username, per_page=30, use_cache=True)

        # 分析
        return self._analyze_user(profile, repos, school_id)

    def batch_profile(self, usernames: List[str], school_id: str) -> Dict[str, dict]:
        """
        批量分析用户（并行）。
        """
        results = {}
        if not usernames:
            return results
        with ThreadPoolExecutor(max_workers=min(len(usernames), 8)) as executor:
            futures = {executor.submit(self.profile, u, school_id): u for u in usernames}
            for future in as_completed(futures):
                username = futures[future]
                try:
                    results[username] = future.result()
                except Exception:
                    results[username] = {"school_confidence": 0, "total_repos": 0}
        return results

    def _analyze_user(self, profile: dict, repos: list, school_id: str) -> dict:
        """分析用户资料和仓库"""
        result = {
            "login": profile.get("login", ""),
            "total_repos": len(repos),
            "school_confidence": 0.0,
            "school_signals": [],
            "school_repo_examples": [],
            "course_repo_examples": [],
            "profile_signals": {}
        }

        if not profile:
            return result

        # 1. Profile 信号
        profile_score = self._check_profile_signals(profile, school_id, result)

        # 2. 仓库信号
        repo_score = self._check_repo_signals(repos, school_id, result)

        # 3. 计算综合置信度
        result["school_confidence"] = min(
            profile_score * 0.4 + repo_score * 0.6,
            1.0
        )

        return result

    def _check_profile_signals(self, profile: dict, school_id: str,
                               result: dict) -> float:
        """检查用户 Profile 中的学校信号"""
        score = 0.0
        school_info = self.school_kb.get_school(school_id)
        if not school_info:
            return 0.0

        # Bio
        bio = (profile.get("bio") or "").lower()
        company = (profile.get("company") or "").lower()
        location = (profile.get("location") or "").lower()
        blog = (profile.get("blog") or "").lower()

        all_text = f"{bio} {company} {location} {blog}"

        # 检查学校别名
        for alias in self.school_kb.get_all_aliases(school_id):
            if contains_signal(all_text, alias):
                result["school_signals"].append(f"profile: {alias}")
                score += 0.3

        # 检查学校域名
        for domain in self.school_kb.get_domains(school_id):
            if contains_signal(blog, domain) or contains_signal(bio, domain):
                result["school_signals"].append(f"domain: {domain}")
                score += 0.4

        # 检查学校所在城市
        city = school_info.get("city", [])
        for c in city:
            if contains_signal(location, c):
                result["school_signals"].append(f"location: {c}")
                score += 0.1

        return min(score, 1.0)

    def _check_repo_signals(self, repos: list, school_id: str,
                            result: dict) -> float:
        """检查用户仓库中的学校信号"""
        if not repos:
            return 0.0

        school_count = 0
        course_count = 0

        for repo in repos:
            name = repo.get("name", "").lower()
            desc = (repo.get("description") or "").lower()
            topics = [t.lower() for t in repo.get("topics", [])]
            text = f"{name} {desc} {' '.join(topics)}"

            # 检查学校信号
            is_school = False
            for alias in self.school_kb.get_all_aliases(school_id):
                if contains_signal(text, alias):
                    school_count += 1
                    is_school = True
                    if len(result["school_repo_examples"]) < 5:
                        result["school_repo_examples"].append(repo.get("full_name", ""))
                    break

            # 检查课程信号
            if not is_school:
                assignment_keywords = [
                    "homework", "lab", "experiment", "project",
                    "作业", "实验", "课程设计"
                ]
                for kw in assignment_keywords:
                    if contains_signal(text, kw):
                        course_count += 1
                        if len(result["course_repo_examples"]) < 5:
                            result["course_repo_examples"].append(repo.get("full_name", ""))
                        break

        # 计算分数
        total = len(repos)
        if total == 0:
            return 0.0

        school_ratio = school_count / total
        score = min(school_ratio * 2, 1.0)  # 30% 以上的仓库有学校信号就满分

        return score

    def expand_by_graph(self, seed_username: str, school_id: str,
                        max_users: int = 10) -> List[dict]:
        """
        通过关注图谱扩展用户

        Args:
            seed_username: 种子用户名
            school_id: 目标学校 ID
            max_users: 最大扩展用户数

        Returns:
            扩展的用户画像列表
        """
        if not self.github:
            return []

        # 获取种子用户的关注者和正在关注的人
        followers = self.github.get_user_followers(seed_username, per_page=30)
        following = self.github.get_user_following(seed_username, per_page=30)

        # 合并去重
        candidate_users = set()
        for user in followers + following:
            login = user.get("login", "")
            if login:
                candidate_users.add(login)

        # 分析候选用户
        results = []
        for username in list(candidate_users)[:max_users * 2]:
            if len(results) >= max_users:
                break

            profile = self.profile(username, school_id)

            # 只保留中高置信度的用户
            if profile.get("school_confidence", 0) >= 0.3:
                results.append(profile)

        return results
