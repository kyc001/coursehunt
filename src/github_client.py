"""
GitHub API 客户端
支持缓存、限流、重试、ETag
"""

import requests
import time
from typing import Optional, Dict, List

from .cache_store import (
    get_cache_store, make_query_cache_key, make_repo_cache_key,
    make_readme_cache_key, make_owner_cache_key
)


class GitHubClient:
    """GitHub API 客户端"""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CourseRepoFinder/1.0"
        })
        if token:
            self.session.headers.update({"Authorization": f"token {token}"})

        self.cache = get_cache_store()

        # 速率限制
        self.rate_limit_remaining = 30
        self.rate_limit_reset = 0
        self.search_rate_remaining = 10

    def _check_rate_limit(self):
        """检查并等待速率限制"""
        if self.rate_limit_remaining <= 2:
            wait_time = max(0, self.rate_limit_reset - time.time()) + 1
            if wait_time > 0 and wait_time < 60:
                time.sleep(wait_time)

    def search_repositories(self, query: str, per_page: int = 30,
                            page: int = 1, sort: str = "best-match",
                            use_cache: bool = True) -> dict:
        """
        搜索仓库

        Args:
            query: 搜索查询
            per_page: 每页结果数
            page: 页码
            sort: 排序方式
            use_cache: 是否使用缓存

        Returns:
            搜索结果
        """
        # 检查缓存
        cache_key = make_query_cache_key(query, "repo", page, sort)
        if use_cache:
            cached = self.cache.get(cache_key, "query")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/search/repositories"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page,
            "sort": sort,
            "order": "desc"
        }

        try:
            response = self.session.get(url, params=params, timeout=15)

            # 更新速率限制
            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 30))
            self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
            self.search_rate_remaining = int(response.headers.get("X-RateLimit-Remaining", 10))

            if response.status_code == 200:
                result = response.json()
                # 缓存结果
                if use_cache:
                    self.cache.set(cache_key, result, "query")
                return result
            elif response.status_code == 304:
                # 未修改，返回缓存
                return self.cache.get(cache_key, "query") or {"items": [], "total_count": 0}
            elif response.status_code == 403:
                return {"error": "API 速率限制", "items": [], "total_count": 0}
            else:
                return {"error": f"API 错误: {response.status_code}", "items": [], "total_count": 0}

        except requests.exceptions.RequestException as e:
            return {"error": f"网络错误: {str(e)}", "items": [], "total_count": 0}

    def get_repo(self, owner: str, repo: str, use_cache: bool = True) -> dict:
        """
        获取仓库详情

        Args:
            owner: 所有者
            repo: 仓库名
            use_cache: 是否使用缓存

        Returns:
            仓库详情
        """
        cache_key = make_repo_cache_key(owner, repo)

        # 检查缓存
        if use_cache:
            cached = self.cache.get(cache_key, "repo")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/repos/{owner}/{repo}"

        # 条件请求
        headers = {}
        etag = self.cache.get_etag(cache_key)
        if etag:
            headers["If-None-Match"] = etag

        try:
            response = self.session.get(url, headers=headers, timeout=10)

            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 30))

            if response.status_code == 200:
                result = response.json()
                new_etag = response.headers.get("ETag", "")
                # 缓存结果
                if use_cache:
                    self.cache.set(cache_key, result, "repo", etag=new_etag)
                return result
            elif response.status_code == 304:
                # 未修改
                return self.cache.get(cache_key, "repo") or {}
            else:
                return {"error": f"API 错误: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"error": f"网络错误: {str(e)}"}

    def get_readme(self, owner: str, repo: str, use_cache: bool = True) -> str:
        """
        获取 README 内容

        Args:
            owner: 所有者
            repo: 仓库名
            use_cache: 是否使用缓存

        Returns:
            README 内容
        """
        cache_key = make_readme_cache_key(owner, repo)

        # 检查缓存
        if use_cache:
            cached = self.cache.get(cache_key, "readme")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/readme"

        # 条件请求
        headers = {}
        etag = self.cache.get_etag(cache_key)
        if etag:
            headers["If-None-Match"] = etag

        try:
            response = self.session.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                import base64
                data = response.json()
                content = data.get("content", "")
                encoding = data.get("encoding", "base64")

                if encoding == "base64":
                    text = base64.b64decode(content).decode("utf-8", errors="ignore")
                else:
                    text = content

                new_etag = response.headers.get("ETag", "")
                if use_cache:
                    self.cache.set(cache_key, text, "readme", etag=new_etag)
                return text

            elif response.status_code == 304:
                return self.cache.get(cache_key, "readme") or ""
            else:
                return ""

        except Exception:
            return ""

    def search_code(self, query: str, per_page: int = 30,
                    page: int = 1, use_cache: bool = True) -> dict:
        """
        搜索 GitHub 代码/路径。

        GitHub 的 code search API 对未认证请求较严格；失败时返回空 items，
        调用方仍可依赖普通仓库搜索和种子仓库召回。
        """
        cache_key = make_query_cache_key(query, "code", page, "indexed")
        if use_cache:
            cached = self.cache.get(cache_key, "query")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/search/code"
        params = {
            "q": query,
            "per_page": min(per_page, 100),
            "page": page,
        }

        try:
            response = self.session.get(url, params=params, timeout=15)
            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 30))
            self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))

            if response.status_code == 200:
                result = response.json()
                if use_cache:
                    self.cache.set(cache_key, result, "query")
                return result
            elif response.status_code in (401, 403):
                return {"error": "Code Search 需要 GitHub Token 或已触发限流", "items": [], "total_count": 0}
            else:
                return {"error": f"Code Search API 错误: {response.status_code}", "items": [], "total_count": 0}
        except requests.exceptions.RequestException as e:
            return {"error": f"网络错误: {str(e)}", "items": [], "total_count": 0}

    def get_repo_tree_paths(self, owner: str, repo: str, recursive: bool = True,
                            use_cache: bool = True) -> List[str]:
        """获取仓库目录树路径，用于识别多课程合集仓库中的课程子文件夹。"""
        cache_key = make_repo_cache_key(owner, f"{repo}:tree:{'recursive' if recursive else 'root'}")
        if use_cache:
            cached = self.cache.get(cache_key, "repo_tree")
            if cached:
                return cached

        repo_data = self.get_repo(owner, repo, use_cache=use_cache)
        default_branch = repo_data.get("default_branch") or "main"
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/{default_branch}"
        params = {"recursive": "1"} if recursive else {}

        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                paths = [
                    item.get("path", "")
                    for item in data.get("tree", [])
                    if item.get("path")
                ]
                if use_cache:
                    self.cache.set(cache_key, paths, "repo_tree")
                return paths
            return []
        except requests.exceptions.RequestException:
            return []

    def get_user_repos(self, username: str, per_page: int = 30,
                       use_cache: bool = True) -> list:
        """
        获取用户公开仓库

        Args:
            username: 用户名
            per_page: 返回数量
            use_cache: 是否使用缓存

        Returns:
            仓库列表
        """
        cache_key = make_owner_cache_key(f"{username}_repos")

        # 检查缓存
        if use_cache:
            cached = self.cache.get(cache_key, "owner_repos")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/users/{username}/repos"
        params = {"per_page": per_page, "type": "public", "sort": "updated"}

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if use_cache:
                    self.cache.set(cache_key, result, "owner_repos")
                return result
            else:
                return []

        except Exception:
            return []

    def get_user_profile(self, username: str, use_cache: bool = True) -> dict:
        """
        获取用户资料

        Args:
            username: 用户名
            use_cache: 是否使用缓存

        Returns:
            用户资料
        """
        cache_key = make_owner_cache_key(username)

        # 检查缓存
        if use_cache:
            cached = self.cache.get(cache_key, "owner")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/users/{username}"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if use_cache:
                    self.cache.set(cache_key, result, "owner")
                return result
            else:
                return {}

        except Exception:
            return {}

    def get_user_followers(self, username: str, per_page: int = 30,
                          use_cache: bool = True) -> list:
        """
        获取用户关注者

        Args:
            username: 用户名
            per_page: 返回数量
            use_cache: 是否使用缓存

        Returns:
            关注者列表
        """
        cache_key = make_owner_cache_key(f"{username}_followers")

        if use_cache:
            cached = self.cache.get(cache_key, "followers")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/users/{username}/followers"
        params = {"per_page": min(per_page, 100)}

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if use_cache:
                    self.cache.set(cache_key, result, "followers")
                return result
            else:
                return []

        except Exception:
            return []

    def get_user_following(self, username: str, per_page: int = 30,
                          use_cache: bool = True) -> list:
        """
        获取用户正在关注的人

        Args:
            username: 用户名
            per_page: 返回数量
            use_cache: 是否使用缓存

        Returns:
            正在关注的用户列表
        """
        cache_key = make_owner_cache_key(f"{username}_following")

        if use_cache:
            cached = self.cache.get(cache_key, "followers")
            if cached:
                return cached

        self._check_rate_limit()

        url = f"{self.BASE_URL}/users/{username}/following"
        params = {"per_page": min(per_page, 100)}

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if use_cache:
                    self.cache.set(cache_key, result, "followers")
                return result
            else:
                return []

        except Exception:
            return []

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return self.cache.get_stats()

    def clear_cache(self, cache_type: str = None):
        """清除缓存"""
        self.cache.clear(cache_type)


# 全局实例
_client = None


def get_github_client(token: str = None) -> GitHubClient:
    """获取 GitHub 客户端单例"""
    global _client
    if _client is None:
        _client = GitHubClient(token)
    return _client
