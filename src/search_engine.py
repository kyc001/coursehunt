"""
搜索引擎主流程
整合查询理解、检索计划、多路召回、RRF 融合、证据抽取、混合排序

三路召回架构：
- 路 A：GitHub Search API（外部黑盒 BM25 + 多 query 扩展）
- 路 B：自建 BM25 倒排索引（在持久化的 LocalCorpus 上）
- 路 C：BGE-M3 稠密向量召回（在持久化的 LocalCorpus 上）

三路结果用 RRF 融合，再走证据抽取和混合排序。
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .local_corpus import (
    CorpusDocument, doc_from_repo_data, get_local_corpus,
)
from .bm25_indexer import get_bm25_index
from .dense_retriever import get_dense_retriever


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


@dataclass
class SearchTrace:
    """单次检索的可观测信息（评测/调试用）。"""
    intent_type: str = ""
    api_routes: int = 0
    api_candidates: int = 0
    bm25_hits: int = 0
    dense_hits: int = 0
    corpus_size: int = 0
    embedding_added: int = 0
    embedding_available: bool = False
    # 各步骤耗时（秒），perf_counter 差值
    t_query_parse: float = 0.0
    t_build_plan: float = 0.0
    t_execute_api: float = 0.0
    t_seed_repos: float = 0.0
    t_rrf_first: float = 0.0
    t_enrich: float = 0.0
    t_persist: float = 0.0
    t_bm25: float = 0.0
    t_dense_embed: float = 0.0
    t_dense_search: float = 0.0
    t_rrf_final: float = 0.0
    t_merge: float = 0.0
    t_owner_profile: float = 0.0
    t_evidence: float = 0.0
    t_ranking: float = 0.0
    t_build_results: float = 0.0
    t_total: float = 0.0


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
        # 三路召回新增组件
        self.corpus = get_local_corpus()
        self.dense_retriever = get_dense_retriever()
        # 评测/调试用：每次 search 写入
        self.last_trace: SearchTrace = SearchTrace()
        # 控制开关：评测脚本可临时关闭某些路
        self.enable_bm25 = True
        self.enable_dense = True
        # 控制 dense embedding 在线补算的预算（避免拖慢首次查询）
        self.dense_max_new_per_query = 20

    def search(self, query: str, mode: str = "fast") -> List[SearchResult]:
        """
        执行搜索

        Args:
            query: 用户查询
            mode: 搜索模式 (fast/deep)

        Returns:
            搜索结果列表
        """
        t0 = time.perf_counter()
        trace = SearchTrace()
        trace.embedding_available = self.dense_retriever.available

        # 1. 查询理解
        t1 = time.perf_counter()
        intent = parse_query(query)
        trace.intent_type = intent.intent_type
        trace.t_query_parse = time.perf_counter() - t1

        # 2. 构建检索计划
        t2 = time.perf_counter()
        budget = 6 if mode == "fast" else 10
        plan = build_search_plan(intent, budget)
        trace.t_build_plan = time.perf_counter() - t2

        # 3. 路 A：GitHub API 多路召回
        t3 = time.perf_counter()
        route_results = self._execute_plan(plan)
        trace.t_execute_api = time.perf_counter() - t3
        self._add_seed_repositories(route_results, intent)
        trace.t_seed_repos = time.perf_counter() - trace.t_execute_api - t3
        trace.api_routes = len(route_results)

        # 4. 第一轮 RRF（仅 API 路）→ 得到候选用于丰富
        t4 = time.perf_counter()
        api_fused = self.fuser.fuse(route_results)
        trace.api_candidates = len(api_fused)
        trace.t_rrf_first = time.perf_counter() - t4
        enrich_limit = 20 if mode == "fast" else 50
        enriched = self._enrich_top(api_fused[:enrich_limit], intent)
        trace.t_enrich = time.perf_counter() - t4 - trace.t_rrf_first

        # 5. 把丰富后的候选写入本地语料库（持久化，让 BM25/Dense 受益）
        t5 = time.perf_counter()
        self._persist_to_corpus(enriched)
        trace.corpus_size = self.corpus.count()
        trace.t_persist = time.perf_counter() - t5

        # 6. 路 B：自建 BM25 倒排索引
        t6 = time.perf_counter()
        if self.enable_bm25:
            bm25_repos = self._bm25_route(query, enriched)
            trace.bm25_hits = len(bm25_repos)
            if bm25_repos:
                route_results["local_bm25"] = bm25_repos
        trace.t_bm25 = time.perf_counter() - t6

        # 7. 路 C：BGE-M3 稠密向量召回
        t7 = time.perf_counter()
        if self.enable_dense and self.dense_retriever.available:
            new_emb = self.dense_retriever.ensure_embeddings(
                self.corpus, max_new=self.dense_max_new_per_query
            )
            trace.embedding_added = new_emb
            self.dense_retriever.load(self.corpus, force=bool(new_emb))
            trace.t_dense_embed = time.perf_counter() - t7
            dense_repos = self._dense_route(query, enriched)
            trace.dense_hits = len(dense_repos)
            if dense_repos:
                route_results["local_dense"] = dense_repos
            trace.t_dense_search = time.perf_counter() - t7 - trace.t_dense_embed
        else:
            trace.t_dense_embed = 0
            trace.t_dense_search = 0

        # 8. 三路 RRF 融合
        t8 = time.perf_counter()
        fused = self.fuser.fuse(route_results)
        trace.t_rrf_final = time.perf_counter() - t8

        # 9. 合并已 enriched 的详细数据，对新增候选构造 minimal repo_data
        t9 = time.perf_counter()
        merged = self._merge_with_enriched(fused, enriched, enrich_limit)
        trace.t_merge = time.perf_counter() - t9

        # 10. 用户画像、证据、排序、最终结果
        ta = time.perf_counter()
        self._profile_owners(merged, intent)
        trace.t_owner_profile = time.perf_counter() - ta

        tb = time.perf_counter()
        self._build_evidence(merged, intent)
        trace.t_evidence = time.perf_counter() - tb

        tc = time.perf_counter()
        ranked = self.ranker.rank(merged, intent)
        trace.t_ranking = time.perf_counter() - tc

        td = time.perf_counter()
        results = self._build_results(ranked)
        trace.t_build_results = time.perf_counter() - td

        trace.t_total = time.perf_counter() - t0
        self.last_trace = trace
        return results

    def _execute_plan(self, plan: SearchPlan) -> Dict[str, List[dict]]:
        """执行检索计划（并行调用各路由）。"""
        route_results = {}

        def _run_task(task: SearchTask):
            if task.route == "code_path":
                result = self.github.search_code(
                    query=task.query, per_page=10, use_cache=True
                )
                items = self._repositories_from_code_results(result.get("items", []))
            else:
                result = self.github.search_repositories(
                    query=task.query, per_page=10, use_cache=True
                )
                items = result.get("items", [])
            return f"{task.route}_{task.name}", items

        with ThreadPoolExecutor(max_workers=min(len(plan.tasks), 8)) as executor:
            futures = {executor.submit(_run_task, t): t for t in plan.tasks}
            for future in as_completed(futures):
                try:
                    key, items = future.result()
                    if items:
                        route_results[key] = items
                except Exception:
                    continue

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
        """加入人工维护的课程资料合集仓库和种子用户仓库（并行获取）。"""
        if not intent.school:
            return

        seed_candidates = []

        # 并行获取种子仓库详情
        seed_full_names = self.school_kb.get_seed_repositories(intent.school)
        seed_users = self.school_kb.get_seed_users(intent.school)

        def _fetch_seed_repo(full_name):
            repo = self._get_repo_by_full_name(full_name)
            if not repo:
                repo = self._minimal_repo_from_full_name(full_name)
            repo["known_collection"] = True
            repo.setdefault("collection_reason", "南开课程资料合集种子仓库")
            curated_paths = self.school_kb.get_seed_repository_paths(intent.school, full_name)
            if curated_paths:
                repo["curated_tree_paths"] = curated_paths
                repo["tree_paths"] = curated_paths
            return repo

        def _fetch_seed_user_repos(username):
            repos = []
            for repo in self.github.get_user_repos(username, per_page=100, use_cache=True):
                if self._is_relevant_seed_user_repo(repo, intent):
                    repo = repo.copy()
                    repo["known_collection"] = True
                    repo.setdefault("collection_reason", f"种子用户 {username} 的课程相关仓库")
                    repos.append(repo)
            return repos

        with ThreadPoolExecutor(max_workers=8) as executor:
            seed_futures = [executor.submit(_fetch_seed_repo, fn) for fn in seed_full_names]
            user_futures = [executor.submit(_fetch_seed_user_repos, u) for u in seed_users]
            for future in as_completed(seed_futures + user_futures):
                try:
                    result = future.result()
                    if isinstance(result, list):
                        seed_candidates.extend(result)
                    elif result:
                        seed_candidates.append(result)
                except Exception:
                    continue

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
        """丰富 Top N 候选的详细信息（并行获取 README / 目录树）。"""
        enriched_order: List[dict] = []
        enriched_map: Dict[str, dict] = {}

        # 收集需要丰富的信息
        enrich_tasks = []
        for candidate in candidates:
            repo_data = candidate.get("repo", candidate)
            owner = repo_data.get("owner", {}).get("login", "")
            name = repo_data.get("name", "")
            full_name = repo_data.get("full_name", "")
            if not owner or not name:
                continue
            enrich_tasks.append((candidate, repo_data, owner, name, full_name))
            if full_name:
                enriched_order.append({"full_name": full_name, "candidate": candidate, "repo_data": repo_data})

        # 并行获取 README 和目录树
        def _fetch(candidate, repo_data, owner, name):
            readme = self.github.get_readme(owner, name, use_cache=True)
            repo_data["readme_text"] = readme
            needs_tree = repo_data.get("known_collection") or repo_data.get("code_path_matches")
            tree_paths = None
            if needs_tree:
                tree_paths = self.github.get_repo_tree_paths(
                    owner, name, recursive=True, use_cache=True
                )
            return candidate, repo_data, readme, tree_paths

        if enrich_tasks:
            with ThreadPoolExecutor(max_workers=min(len(enrich_tasks), 10)) as executor:
                futures = [executor.submit(_fetch, *args) for args in enrich_tasks]
                for future in as_completed(futures):
                    try:
                        candidate, repo_data, readme, tree_paths = future.result()
                        if tree_paths is not None:
                            curated = repo_data.get("curated_tree_paths") or []
                            repo_data["tree_paths"] = list(dict.fromkeys(curated + tree_paths))
                        analysis = self.analyzer.analyze_repo(repo_data, readme)
                        repo_data["analysis"] = analysis
                        candidate["repo_data"] = repo_data
                        candidate["rrf_score"] = repo_data.get("rrf_score", candidate.get("rrf_score", 0))
                        fn = repo_data.get("full_name", "")
                        if fn:
                            enriched_map[fn] = candidate
                    except Exception:
                        continue

        # 保持原有顺序
        enriched = []
        seen = set()
        for entry in enriched_order:
            fn = entry["full_name"]
            if fn in enriched_map and fn not in seen:
                seen.add(fn)
                enriched.append(enriched_map[fn])
            elif fn not in seen:
                seen.add(fn)
                # 并行未覆盖的候选用同步回退
                c = entry["candidate"]
                rd = entry["repo_data"]
                rd2 = c.get("repo", c)
                owner = rd.get("owner", {}).get("login", "") or rd2.get("owner", {}).get("login", "")
                name = rd.get("name", "") or rd2.get("name", "")
                if owner and name and not rd.get("readme_text"):
                    rd["readme_text"] = self.github.get_readme(owner, name, use_cache=True)
                if not rd.get("analysis"):
                    rd["analysis"] = self.analyzer.analyze_repo(rd, rd.get("readme_text", ""))
                c["repo_data"] = rd
                c["rrf_score"] = rd.get("rrf_score", c.get("rrf_score", 0))
                enriched.append(c)

        return enriched

    def _persist_to_corpus(self, enriched: List[dict]):
        """把 enriched 候选写入本地语料库，供 BM25/Dense 使用。"""
        for item in enriched:
            repo_data = item.get("repo_data") or {}
            if not repo_data.get("full_name"):
                continue
            doc = doc_from_repo_data(repo_data)
            if not doc.full_name:
                continue
            self.corpus.upsert(doc)

    def _bm25_route(self, query: str, enriched: List[dict],
                     top_k: int = 30) -> List[dict]:
        """BM25 路（作为重排序信号）：只对已在 API 候选集内的仓库重新排序，
        不引入新的候选，避免在小语料场景下稀释精确匹配的 top。"""
        index = get_bm25_index()
        hits = index.search(query, top_k=top_k * 3)
        if not hits:
            return []
        allowed = {
            (item.get("repo_data") or {}).get("full_name", "")
            for item in enriched
        }
        allowed.discard("")
        hits = [(fn, s) for fn, s in hits if fn in allowed][:top_k]
        if not hits:
            return []
        return self._materialize_hits(hits, enriched)

    def _dense_route(self, query: str, enriched: List[dict],
                     top_k: int = 30) -> List[dict]:
        """Dense 路（作为重排序信号）：BGE-M3 余弦近邻在 API 候选集上重排序。"""
        hits = self.dense_retriever.search(query, top_k=top_k * 3)
        if not hits:
            return []
        allowed = {
            (item.get("repo_data") or {}).get("full_name", "")
            for item in enriched
        }
        allowed.discard("")
        hits = [(fn, s) for fn, s in hits if fn in allowed][:top_k]
        if not hits:
            return []
        return self._materialize_hits(hits, enriched)

    def _materialize_hits(self, hits: List, enriched: List[dict]) -> List[dict]:
        """把 (full_name, score) 命中转成完整 repo dict 列表。

        - 若 full_name 在 enriched 中，复用其 repo_data（保留 README、画像等已有字段）
        - 否则从 LocalCorpus 反查并构造一份 minimal 但带 README 的 dict
        - 仍找不到的（不应该出现）跳过

        返回的列表保持 hits 的相对顺序，供 RRF 使用。
        """
        enriched_map = {}
        for item in enriched:
            repo_data = item.get("repo_data") or {}
            full_name = repo_data.get("full_name", "")
            if full_name:
                enriched_map[full_name] = repo_data

        materialized: List[dict] = []
        for full_name, _score in hits:
            if full_name in enriched_map:
                materialized.append(enriched_map[full_name])
                continue

            doc = self.corpus.get(full_name)
            if not doc:
                continue
            owner_login = full_name.split("/", 1)[0] if "/" in full_name else ""
            materialized.append({
                "full_name": full_name,
                "name": doc.name,
                "owner": {"login": owner_login},
                "html_url": f"https://github.com/{full_name}",
                "description": doc.description,
                "readme_text": doc.readme,
                "topics": doc.topics,
                "tree_paths": doc.paths,
                "language": doc.language,
                "stargazers_count": doc.stars,
                "forks_count": doc.forks,
                "pushed_at": doc.pushed_at,
            })

        return materialized

    def _merge_with_enriched(self, fused: List[dict], enriched: List[dict],
                              limit: int) -> List[dict]:
        """三路 RRF 融合后，把每个候选包成 ranker 需要的形态。

        返回元素结构：{repo_data, rrf_score, owner_context(待填), evidence(待填)}
        """
        enriched_items = {}
        for item in enriched:
            repo_data = item.get("repo_data") or {}
            full_name = repo_data.get("full_name", "")
            if full_name:
                enriched_items[full_name] = item

        merged: List[dict] = []
        for repo in fused[:limit]:
            full_name = repo.get("full_name", "")
            if not full_name:
                continue

            rrf_score = repo.get("rrf_score", 0.0)

            if full_name in enriched_items:
                base = enriched_items[full_name]
                base["rrf_score"] = rrf_score
                base.setdefault("repo_data", repo)
                merged.append(base)
                continue

            # 新候选：fused 中已经把 repo_data 平铺了，直接当 repo_data
            merged.append({
                "repo_data": repo,
                "rrf_score": rrf_score,
            })

        return merged

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
