"""
Reciprocal Rank Fusion (RRF) 多路融合模块
将多路召回结果进行融合排序
"""

from typing import Dict, List, Tuple
from collections import defaultdict


def rrf_score(rankings: Dict[str, List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion 算法

    Args:
        rankings: 多路召回结果 {route_name: [repo_ids]}
        k: 平滑参数 (默认 60)

    Returns:
        融合后的 (repo_id, score) 列表，按分数降序
    """
    scores = defaultdict(float)

    for route_name, repos in rankings.items():
        for rank, repo_id in enumerate(repos, start=1):
            scores[repo_id] += 1.0 / (k + rank)

    # 按分数降序排序
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def rrf_score_with_source(rankings: Dict[str, List[str]], k: int = 60) -> List[Tuple[str, float, List[str]]]:
    """
    带来源信息的 RRF

    Args:
        rankings: 多路召回结果 {route_name: [repo_ids]}
        k: 平滑参数

    Returns:
        融合后的 (repo_id, score, sources) 列表
    """
    scores = defaultdict(float)
    sources = defaultdict(list)

    for route_name, repos in rankings.items():
        for rank, repo_id in enumerate(repos, start=1):
            score = 1.0 / (k + rank)
            scores[repo_id] += score
            sources[repo_id].append({
                "route": route_name,
                "rank": rank,
                "contribution": score
            })

    # 按分数降序排序
    results = []
    for repo_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        results.append((repo_id, score, sources[repo_id]))

    return results


def weighted_rrf_score(rankings: Dict[str, List[str]],
                       route_weights: Dict[str, float],
                       k: int = 60) -> List[Tuple[str, float]]:
    """
    加权 RRF

    Args:
        rankings: 多路召回结果
        route_weights: 路由权重 {route_name: weight}
        k: 平滑参数

    Returns:
        融合后的 (repo_id, score) 列表
    """
    scores = defaultdict(float)

    for route_name, repos in rankings.items():
        weight = route_weights.get(route_name, 1.0)
        for rank, repo_id in enumerate(repos, start=1):
            scores[repo_id] += weight / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def merge_with_source_bonus(rrf_results: List[Tuple[str, float, List[str]]],
                            source_bonus: Dict[str, float] = None) -> List[Tuple[str, float]]:
    """
    融合结果加上来源加成

    Args:
        rrf_results: RRF 结果 (repo_id, score, sources)
        source_bonus: 来源加成 {route_name: bonus}

    Returns:
        (repo_id, final_score) 列表
    """
    if source_bonus is None:
        source_bonus = {
            "exact": 0.15,
            "code": 0.10,
            "path": 0.12,
            "code_path": 0.12,
            "tech": 0.08,
            "broad": 0.05,
            "owner": 0.10,
            "seed_collections": 0.18,
            "seed": 0.18,
            "original": 0.08,
            "reverse": 0.12,
            "english": 0.12,
            "resource": 0.08,
            "github": 0.06,
            "local_bm25": 0.08,
            "local_dense": 0.08,
        }

    results = []
    for repo_id, base_score, sources in rrf_results:
        # 计算来源加成
        bonus = 0.0
        seen_routes = set()
        for src in sources:
            route = src["route"]
            route_key = _source_bonus_key(route, source_bonus)
            if route_key not in seen_routes:
                bonus += source_bonus.get(route_key, 0.0)
                seen_routes.add(route_key)

        final_score = base_score + bonus
        results.append((repo_id, final_score))

    # 按最终分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _source_bonus_key(route_name: str, source_bonus: Dict[str, float]) -> str:
    """Map task names like exact_xxx or code_path_xxx back to route families."""
    if route_name in source_bonus:
        return route_name
    if route_name.startswith("code_path"):
        return "code_path"
    if route_name.startswith("seed"):
        return "seed"
    if route_name.startswith("resource"):
        return "resource"
    if route_name.startswith("english"):
        return "english"
    if route_name.startswith("github"):
        return "github"
    if route_name.startswith("local_bm25"):
        return "local_bm25"
    if route_name.startswith("local_dense"):
        return "local_dense"
    return route_name.split("_", 1)[0]


class RRFFuser:
    """RRF 融合器"""

    def __init__(self, k: int = 60, source_bonus: Dict[str, float] = None):
        self.k = k
        self.source_bonus = source_bonus or {
            "exact": 0.15,
            "code": 0.10,
            "path": 0.12,
            "code_path": 0.12,
            "tech": 0.08,
            "broad": 0.05,
            "owner": 0.10,
            "seed_collections": 0.18,
            "seed": 0.18,
            "original": 0.08,
            "reverse": 0.12,
            "english": 0.12,
            "resource": 0.08,
            "github": 0.06,
            "local_bm25": 0.08,
            "local_dense": 0.08,
        }

    def fuse(self, route_results: Dict[str, List[dict]]) -> List[dict]:
        """
        融合多路召回结果

        Args:
            route_results: {route_name: [repo_data_list]}
                每个 repo_data 至少包含 {"full_name": "xxx/yyy", ...}

        Returns:
            融合后的 repo_data 列表，按分数降序，添加了 rrf_score 和 sources 字段
        """
        # 提取 rankings
        rankings = {}
        repo_data_map = {}

        for route_name, repos in route_results.items():
            repo_ids = []
            for repo in repos:
                repo_id = repo.get("full_name", "")
                if repo_id:
                    repo_ids.append(repo_id)
                    repo_data_map[repo_id] = repo
            rankings[route_name] = repo_ids

        # RRF 融合
        rrf_results = rrf_score_with_source(rankings, self.k)

        # 加来源加成
        final_results = merge_with_source_bonus(rrf_results, self.source_bonus)

        # 构建最终结果
        output = []
        for repo_id, final_score in final_results:
            if repo_id in repo_data_map:
                repo_data = repo_data_map[repo_id].copy()
                repo_data["rrf_score"] = final_score
                # 找出来源
                for r_id, _, sources in rrf_results:
                    if r_id == repo_id:
                        repo_data["rrf_sources"] = sources
                        break
                output.append(repo_data)

        return output


# 全局实例
_fuser = None


def get_rrf_fuser() -> RRFFuser:
    """获取 RRF 融合器单例"""
    global _fuser
    if _fuser is None:
        _fuser = RRFFuser()
    return _fuser
