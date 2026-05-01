"""
搜索引擎主流程
整合查询理解、检索计划、多路召回、RRF 融合、证据抽取、混合排序
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

from .query_parser import QueryIntent, parse_query
from .search_planner import SearchPlan, build_search_plan, SearchTask
from .github_client import get_github_client
from .school_kb import get_school_kb
from .course_kb import get_course_kb
from .matching import contains_signal
from .rrf import get_rrf_fuser
from .repo_analyzer import RepoAnalyzer
from .owner_profiler import OwnerProfiler
from .evidence_builder import get_evidence_builder, RepoEvidence
from .hybrid_ranker import get_hybrid_ranker


@dataclass
class SearchResult:
    """搜索结果"""
    repo: str
    score: float
    confidence: str
    reasons: List[str]
    risks: List[str]
    evidence: RepoEvidence
    repo_data: dict
    owner_context: dict
    rrf_score: float


class SearchEngine:
    """搜索引擎"""

    def __init__(self):
        load_dotenv()
        token = os.getenv("GITHUB_TOKEN")
        self.github = get_github_client(token)
        self.analyzer = RepoAnalyzer()
        self.owner_profiler = OwnerProfiler(self.github)
        self.evidence_builder = get_evidence_builder()
        self.ranker = get_hybrid_ranker()
        self.fuser = get_rrf_fuser()
        self.school_kb = get_school_kb()
        self.course_kb = get_course_kb()

    def search(self, query: str, mode: str = "fast") -> List[SearchResult]:
        """
        执行搜索

        Args:
            query: 用户查询
            mode: 搜索模式 (fast/deep)

        Returns:
            搜索结果列表
        """
        # 1. 查询理解
        intent = parse_query(query)

        # 2. 构建检索计划
        budget = 6 if mode == "fast" else 10
        plan = build_search_plan(intent, budget)

        # 3. 多路召回
        route_results = self._execute_plan(plan)
        self._add_seed_repositories(route_results, intent)

        # 4. RRF 融合
        candidates = self.fuser.fuse(route_results)

        # 5. Top N 丰富化
        enrich_limit = 20 if mode == "fast" else 50
        enriched = self._enrich_top(candidates[:enrich_limit], intent)

        # 6. 用户画像
        self._profile_owners(enriched, intent)

        # 7. 证据抽取
        self._build_evidence(enriched, intent)

        # 8. 混合排序
        ranked = self.ranker.rank(enriched, intent)

        # 9. 构建最终结果
        results = self._build_results(ranked)

        return results

    def _execute_plan(self, plan: SearchPlan) -> Dict[str, List[dict]]:
        """执行检索计划"""
        route_results = {}

        for task in plan.tasks:
            if task.route == "code_path":
                result = self.github.search_code(
                    query=task.query,
                    per_page=10,
                    use_cache=True
                )
                items = self._repositories_from_code_results(result.get("items", []))
            else:
                result = self.github.search_repositories(
                    query=task.query,
                    per_page=10,
                    use_cache=True
                )
                items = result.get("items", [])

            if items:
                route_results[f"{task.route}_{task.name}"] = items

        return route_results

    def _repositories_from_code_results(self, items: List[dict]) -> List[dict]:
        """把 code search 的文件命中归并成仓库候选。"""
        repos = {}
        for item in items:
            repo = item.get("repository") or {}
            full_name = repo.get("full_name")
            if not full_name:
                continue

            current = repos.setdefault(full_name, repo.copy())
            current.setdefault("code_path_matches", [])
            current["code_path_matches"].append({
                "path": item.get("path", ""),
                "html_url": item.get("html_url", ""),
                "name": item.get("name", ""),
            })

        return list(repos.values())

    def _add_seed_repositories(self, route_results: Dict[str, List[dict]], intent: QueryIntent):
        """加入人工维护的课程资料合集仓库和种子用户仓库。"""
        if not intent.school:
            return

        seed_candidates = []
        for full_name in self.school_kb.get_seed_repositories(intent.school):
            repo = self._get_repo_by_full_name(full_name)
            if not repo:
                repo = self._minimal_repo_from_full_name(full_name)
            repo["known_collection"] = True
            repo.setdefault("collection_reason", "南开课程资料合集种子仓库")
            curated_paths = self.school_kb.get_seed_repository_paths(intent.school, full_name)
            if curated_paths:
                repo["curated_tree_paths"] = curated_paths
                repo["tree_paths"] = curated_paths
            seed_candidates.append(repo)

        for username in self.school_kb.get_seed_users(intent.school):
            for repo in self.github.get_user_repos(username, per_page=100, use_cache=True):
                if self._is_relevant_seed_user_repo(repo, intent):
                    repo = repo.copy()
                    repo["known_collection"] = True
                    repo.setdefault("collection_reason", f"种子用户 {username} 的课程相关仓库")
                    seed_candidates.append(repo)

        deduped = {}
        for repo in seed_candidates:
            full_name = repo.get("full_name")
            if full_name:
                deduped[full_name] = repo

        if deduped:
            route_results["seed_collections"] = list(deduped.values())

    def _get_repo_by_full_name(self, full_name: str) -> dict:
        if "/" not in full_name:
            return {}
        owner, repo = full_name.split("/", 1)
        data = self.github.get_repo(owner, repo, use_cache=True)
        if data.get("error"):
            return {}
        return data

    def _minimal_repo_from_full_name(self, full_name: str) -> dict:
        owner, name = full_name.split("/", 1)
        return {
            "full_name": full_name,
            "name": name,
            "owner": {"login": owner},
            "html_url": f"https://github.com/{full_name}",
            "description": "",
            "stargazers_count": 0,
            "forks_count": 0,
            "topics": [],
        }

    def _is_relevant_seed_user_repo(self, repo: dict, intent: QueryIntent) -> bool:
        text = f"{repo.get('name') or ''} {repo.get('description') or ''} {' '.join(repo.get('topics') or [])}"

        if intent.course and self.course_kb.check_course_signal(text, intent.course)["found"]:
            return True

        negative_terms = ["finance", "金融", "辅修"]
        if any(contains_signal(text, term) for term in negative_terms):
            return False

        school_hit = self.school_kb.check_school_signal(text, intent.school)["found"]
        cs_terms = ["计算机", "网安", "网络空间安全", "cs", "computer", "cyber"]
        collection_terms = [
            "course", "courses", "lab", "labs", "report", "exam", "note",
            "课程", "实验", "报告", "复习", "笔记", "资料", "作业"
        ]
        return (
            school_hit
            and any(contains_signal(text, term) for term in collection_terms)
            and any(contains_signal(text, term) for term in cs_terms)
        )

    def _enrich_top(self, candidates: List[dict], intent: QueryIntent) -> List[dict]:
        """丰富 Top N 候选的详细信息"""
        enriched = []

        for candidate in candidates:
            repo_data = candidate.get("repo", candidate)
            owner = repo_data.get("owner", {}).get("login", "")
            name = repo_data.get("name", "")

            if not owner or not name:
                continue

            # 获取 README
            readme = self.github.get_readme(owner, name, use_cache=True)
            repo_data["readme_text"] = readme

            if repo_data.get("known_collection") or repo_data.get("code_path_matches"):
                fetched_paths = self.github.get_repo_tree_paths(
                    owner, name, recursive=True, use_cache=True
                )
                curated_paths = repo_data.get("curated_tree_paths") or []
                repo_data["tree_paths"] = list(dict.fromkeys(curated_paths + fetched_paths))

            # 分析仓库
            analysis = self.analyzer.analyze_repo(repo_data, readme)
            repo_data["analysis"] = analysis

            candidate["repo_data"] = repo_data
            candidate["rrf_score"] = repo_data.get("rrf_score", candidate.get("rrf_score", 0))
            enriched.append(candidate)

        return enriched

    def _profile_owners(self, enriched: List[dict], intent: QueryIntent):
        """分析用户画像"""
        if not intent.school:
            return

        # 收集所有 owner
        owners = set()
        for item in enriched:
            repo_data = item.get("repo_data", {})
            owner = repo_data.get("owner", {}).get("login", "")
            if owner:
                owners.add(owner)

        # 批量分析
        owner_profiles = self.owner_profiler.batch_profile(
            list(owners)[:20],
            intent.school
        )

        # 附加到结果
        for item in enriched:
            repo_data = item.get("repo_data", {})
            owner = repo_data.get("owner", {}).get("login", "")
            if owner in owner_profiles:
                item["owner_context"] = owner_profiles[owner]
            else:
                item["owner_context"] = {}

    def _build_evidence(self, enriched: List[dict], intent: QueryIntent):
        """构建证据"""
        for item in enriched:
            repo_data = item.get("repo_data", {})
            owner_context = item.get("owner_context", {})

            evidence = self.evidence_builder.build_evidence(
                repo_data, owner_context,
                school_id=intent.school,
                course_id=intent.course
            )
            item["evidence"] = evidence

    def _build_results(self, ranked: List[dict]) -> List[SearchResult]:
        """构建最终结果"""
        results = []

        for item in ranked:
            repo_data = item.get("repo_data", {})
            evidence = item.get("evidence")

            if not evidence:
                continue

            result = SearchResult(
                repo=repo_data.get("full_name", ""),
                score=item.get("final_score", 0),
                confidence=evidence.confidence,
                reasons=evidence.reasons,
                risks=evidence.risks,
                evidence=evidence,
                repo_data=repo_data,
                owner_context=item.get("owner_context", {}),
                rrf_score=item.get("rrf_score", 0)
            )
            results.append(result)

        return results


# 全局实例
_engine = None


def get_search_engine() -> SearchEngine:
    """获取搜索引擎单例"""
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


def search_course_repos(query: str, mode: str = "fast") -> List[SearchResult]:
    """搜索课程仓库的便捷函数"""
    return get_search_engine().search(query, mode)
