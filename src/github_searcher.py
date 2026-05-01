"""
GitHub API 搜索模块
封装 GitHub REST API 进行仓库搜索
"""

import requests
import time
from typing import Optional


class GitHubSearcher:
    """GitHub 仓库搜索器"""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        """
        初始化搜索器

        Args:
            token: GitHub Personal Access Token (可选，提高速率限制)
        """
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CourseRepoFinder/1.0"
        })
        if token:
            self.session.headers.update({"Authorization": f"token {token}"})

        self.rate_limit_remaining = 30
        self.rate_limit_reset = 0

    def _check_rate_limit(self):
        """检查并等待速率限制"""
        if self.rate_limit_remaining <= 1:
            wait_time = max(0, self.rate_limit_reset - time.time()) + 1
            if wait_time > 0:
                time.sleep(wait_time)

    def search_repositories(self, query: str, per_page: int = 30, page: int = 1) -> dict:
        """
        搜索 GitHub 仓库

        Args:
            query: 搜索查询字符串
            per_page: 每页结果数 (最大 100)
            page: 页码

        Returns:
            API 响应字典
        """
        self._check_rate_limit()

        url = f"{self.BASE_URL}/search/repositories"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page,
            "sort": "best-match",  # 可选: stars, forks, updated
            "order": "desc"
        }

        try:
            response = self.session.get(url, params=params, timeout=10)

            # 更新速率限制信息
            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 30))
            self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                return {"error": "API 速率限制，请稍后再试或配置 GitHub Token", "items": []}
            else:
                return {"error": f"API 错误: {response.status_code}", "items": []}

        except requests.exceptions.RequestException as e:
            return {"error": f"网络错误: {str(e)}", "items": []}

    def search_code(self, query: str, per_page: int = 30, page: int = 1) -> dict:
        """
        搜索 GitHub 代码

        Args:
            query: 搜索查询字符串
            per_page: 每页结果数
            page: 页码

        Returns:
            API 响应字典
        """
        self._check_rate_limit()

        url = f"{self.BASE_URL}/search/code"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page
        }

        try:
            response = self.session.get(url, params=params, timeout=10)

            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 30))
            self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API 错误: {response.status_code}", "items": []}

        except requests.exceptions.RequestException as e:
            return {"error": f"网络错误: {str(e)}", "items": []}

    def get_repo_details(self, owner: str, repo: str) -> dict:
        """
        获取仓库详细信息

        Args:
            owner: 仓库所有者
            repo: 仓库名称

        Returns:
            仓库详情字典
        """
        self._check_rate_limit()

        url = f"{self.BASE_URL}/repos/{owner}/{repo}"

        try:
            response = self.session.get(url, timeout=10)

            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 30))

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"无法获取仓库详情: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"网络错误: {str(e)}"}

    def get_repo_readme(self, owner: str, repo: str) -> str:
        """
        获取仓库 README 内容

        Args:
            owner: 仓库所有者
            repo: 仓库名称

        Returns:
            README 内容字符串
        """
        self._check_rate_limit()

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/readme"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                import base64
                content = response.json().get("content", "")
                encoding = response.json().get("encoding", "base64")
                if encoding == "base64":
                    return base64.b64decode(content).decode("utf-8", errors="ignore")
                return content
            else:
                return ""

        except Exception:
            return ""

    def get_user_repos(self, username: str, per_page: int = 30) -> list:
        """
        获取用户的公开仓库列表

        Args:
            username: GitHub 用户名
            per_page: 返回数量

        Returns:
            仓库列表
        """
        self._check_rate_limit()

        url = f"{self.BASE_URL}/users/{username}/repos"
        params = {"per_page": per_page, "type": "public"}

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return []

        except Exception:
            return []

    def multi_query_search(self, queries: list, max_results_per_query: int = 10) -> list:
        """
        执行多路召回搜索

        Args:
            queries: 查询列表，每个元素为 (query_string, search_type, description)
            max_results_per_query: 每个查询最多返回结果数

        Returns:
            去重后的仓库列表
        """
        all_repos = {}
        search_log = []

        for query, search_type, desc in queries:
            result = self.search_repositories(query, per_page=max_results_per_query)

            if "error" in result:
                search_log.append(f"[失败] {desc}: {result['error']}")
                continue

            items = result.get("items", [])
            search_log.append(f"[成功] {desc}: 找到 {len(items)} 个仓库")

            for repo in items:
                repo_id = repo.get("full_name")
                if repo_id and repo_id not in all_repos:
                    all_repos[repo_id] = {
                        "repo": repo,
                        "matched_queries": [desc],
                        "query_count": 1
                    }
                elif repo_id:
                    all_repos[repo_id]["matched_queries"].append(desc)
                    all_repos[repo_id]["query_count"] += 1

        return list(all_repos.values()), search_log
