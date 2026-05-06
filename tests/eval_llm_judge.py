"""
LLM-as-judge 自动评测脚本

不需要人工标注，用 OpenRouter 上的免费 LLM 当裁判，对每个 (query, repo)
对打 0-3 分作为伪 ground truth。然后用同一份判分来比较三个版本：

    A. api_only : 仅 GitHub Search API
    B. +bm25    : GitHub API + 自建 BM25 倒排索引
    C. full     : GitHub API + BM25 + BGE-M3 稠密向量

指标：
    - P@5      （以 score >= 2 为相关阈值）
    - MRR      （首个 score >= 2 的倒数）
    - NDCG@10  （连续 0-3 分作为 gain）

判分缓存到 tests/eval_results/judgments.json，重复运行不会重复消耗 LLM 配额。
最终结果写入 tests/eval_results/eval_summary.json。

使用：
    python tests/eval_llm_judge.py            # 全量
    python tests/eval_llm_judge.py --quick    # 仅 8 个查询（验证流程）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

# 让脚本能从 tests/ 目录直接运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.search_engine import get_search_engine  # noqa: E402


# ----- 测试查询集 -----

# 25 个查询，覆盖：specific(学校+课程+作业)、school+course、course+assignment、course only、
# school only。每条 (query, intent_label) 用于后续按意图分组报告。
QUERIES: List[Tuple[str, str]] = [
    # 学校 + 课程 + 作业
    ("南开 操作系统 lab2", "specific"),
    ("南开 操作系统 实验3", "specific"),
    ("南开 并行程序设计 lab1", "specific"),
    ("南开大学 编译原理 lab", "specific"),
    ("南开 数据结构 hw1", "specific"),
    ("nankai operating system lab2", "specific"),
    ("nku parallel programming homework", "specific"),
    # 学校 + 课程
    ("南开 编译原理", "school_course"),
    ("南开 操作系统", "school_course"),
    ("南开 数据库", "school_course"),
    ("南开 计算机网络", "school_course"),
    ("南开 数据结构", "school_course"),
    ("南开 信息检索", "school_course"),
    ("南开 机器学习", "school_course"),
    ("南开 深度学习", "school_course"),
    ("nku compiler", "school_course"),
    ("nankai database system", "school_course"),
    # 课程 + 作业
    ("操作系统 xv6 lab", "course_assignment"),
    ("编译原理 lab", "course_assignment"),
    ("并行程序设计 mpi homework", "course_assignment"),
    # 课程 only
    ("信息检索系统原理", "course_only"),
    ("机器学习及应用", "course_only"),
    # 学校 only
    ("南开 计算机", "school_only"),
    ("nku cs course", "school_only"),
    # 通用
    ("data structures and algorithms course material", "generic"),
]


VERSIONS = ["api_only", "+bm25", "full"]


# ----- LLM Judge -----


JUDGE_PROMPT = """你是一个严格的 GitHub 仓库相关性评分员。请对下面的 (查询, 仓库) 对打分。

## 评分标准（0-3）
- 3 = 完美匹配。仓库明显是该查询要找的内容，覆盖查询中的所有重要维度（学校 + 课程 + 作业号 等）。
- 2 = 高度相关。命中查询的主要维度（例如学校 + 课程，但作业号没明确说明），用户大概率会用。
- 1 = 弱相关。只命中一个维度（仅同课程不同学校；或仅同学校不同课程；或仅同主题）。
- 0 = 无关。完全不是用户要找的内容（例如通用教程库、awesome-* 列表、读书笔记、其他领域）。

## 待评分对
- 查询: {query}
- 仓库全名: {full_name}
- 仓库描述: {description}
- README 节选 (前 800 字):
{readme_excerpt}
- Topics: {topics}
- Star 数: {stars}

## 输出
只输出一个数字（0、1、2、3 之一），不要任何解释、不要任何标点、不要思考过程。
"""


class LLMJudge:
    """OpenRouter 兼容的 LLM 评分客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8899/api/v1"
        ).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-120b:free"
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def judge(self, query: str, repo_data: dict) -> Optional[int]:
        if not self.available:
            return None

        full_name = repo_data.get("full_name", "")
        description = (repo_data.get("description") or "").strip()
        readme = (repo_data.get("readme_text") or "")[:800].strip()
        topics = ", ".join(repo_data.get("topics") or []) or "(none)"
        stars = repo_data.get("stargazers_count", 0)

        prompt = JUDGE_PROMPT.format(
            query=query,
            full_name=full_name,
            description=description or "(无描述)",
            readme_excerpt=readme or "(README 不可用)",
            topics=topics,
            stars=stars,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "CourseHunt-Eval",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You output only one digit between 0 and 3."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }

        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    time.sleep(5 + attempt * 5)
                    continue
                if resp.status_code != 200:
                    return None
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # gpt-oss 偶尔会带 reasoning_content 或多输出
                for ch in content:
                    if ch in "0123":
                        return int(ch)
                return None
            except requests.exceptions.RequestException:
                time.sleep(2 + attempt)
                continue
        return None


# ----- 指标 -----


def precision_at_k(scores: List[int], k: int = 5, threshold: int = 2) -> float:
    if not scores:
        return 0.0
    relevant = sum(1 for s in scores[:k] if s >= threshold)
    return relevant / k


def mrr(scores: List[int], threshold: int = 2) -> float:
    for i, s in enumerate(scores, start=1):
        if s >= threshold:
            return 1.0 / i
    return 0.0


def dcg(scores: List[int]) -> float:
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(scores: List[int], k: int = 10) -> float:
    actual = scores[:k]
    if not actual or all(s == 0 for s in actual):
        return 0.0
    ideal = sorted(actual, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(actual) / ideal_dcg


# ----- 主流程 -----


def collect_results(
    queries: List[Tuple[str, str]], top_k: int = 10
) -> Dict[str, List[Dict]]:
    """三个版本各跑一次，返回 {version: [{query, intent, top: [{repo, repo_data}]}]}.

    第一遍 'full' 跑完会把所有候选写入语料库；后续 +bm25/api_only 只是关闭对应路。
    """
    engine = get_search_engine()

    # 先跑 full 一遍把语料填满 + 算 embedding
    print("[1/3] 预跑 full 模式以充实语料库与 embedding...")
    engine.enable_bm25 = True
    engine.enable_dense = True
    engine.dense_max_new_per_query = 30
    for q, _ in queries:
        engine.search(q, mode="deep")
        time.sleep(0.4)

    by_version: Dict[str, List[Dict]] = {v: [] for v in VERSIONS}
    repo_data_cache: Dict[str, dict] = {}  # full_name -> repo_data 用于 LLM 判分

    print("[2/3] 跑三个版本（api_only / +bm25 / full）...")
    for version in VERSIONS:
        print(f"  -> 版本: {version}")
        engine.enable_bm25 = version != "api_only"
        engine.enable_dense = version == "full"
        for q, intent_label in queries:
            results = engine.search(q, mode="deep")
            top: List[Dict] = []
            for r in results[:top_k]:
                top.append({"repo": r.repo})
                repo_data_cache[r.repo] = r.repo_data
            by_version[version].append({"query": q, "intent": intent_label, "top": top})

    return by_version, repo_data_cache


def load_judgments(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_judgments(path: Path, judgments: Dict[str, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(judgments, f, ensure_ascii=False, indent=2)


def judge_pairs(
    by_version: Dict[str, List[Dict]],
    repo_data_cache: Dict[str, dict],
    judgments_path: Path,
) -> Dict[str, int]:
    """对所有 (query, repo) 唯一对调用 LLM 判分；带磁盘缓存。"""
    judgments = load_judgments(judgments_path)
    judge = LLMJudge()

    pairs = set()
    for version_results in by_version.values():
        for entry in version_results:
            q = entry["query"]
            for r in entry["top"]:
                pairs.add((q, r["repo"]))

    pending = [
        (q, repo) for q, repo in pairs if f"{q}\t{repo}" not in judgments
    ]
    print(f"[3/3] LLM 判分: 总对数 {len(pairs)}, 待判 {len(pending)}, 已缓存 {len(pairs) - len(pending)}")

    if not judge.available:
        print("  ! OPENAI_API_KEY 未设置，跳过 LLM 判分（所有未判定的对会算作 0 分）")
        return judgments

    for i, (q, repo) in enumerate(pending, start=1):
        repo_data = repo_data_cache.get(repo, {"full_name": repo})
        score = judge.judge(q, repo_data)
        if score is None:
            score = 0
        judgments[f"{q}\t{repo}"] = score
        if i % 5 == 0:
            save_judgments(judgments_path, judgments)
            print(f"    progress: {i}/{len(pending)}")
        time.sleep(0.6)  # 免费档限速

    save_judgments(judgments_path, judgments)
    return judgments


def compute_metrics(
    by_version: Dict[str, List[Dict]], judgments: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for version, entries in by_version.items():
        p5, mrr_l, ndcg10 = [], [], []
        for entry in entries:
            q = entry["query"]
            scores = [judgments.get(f"{q}\t{r['repo']}", 0) for r in entry["top"]]
            p5.append(precision_at_k(scores, 5, threshold=2))
            mrr_l.append(mrr(scores, threshold=2))
            ndcg10.append(ndcg_at_k(scores, 10))
        metrics[version] = {
            "P@5": round(mean(p5), 4) if p5 else 0.0,
            "MRR": round(mean(mrr_l), 4) if mrr_l else 0.0,
            "NDCG@10": round(mean(ndcg10), 4) if ndcg10 else 0.0,
            "n_queries": len(entries),
        }
    return metrics


def per_intent_breakdown(
    by_version: Dict[str, List[Dict]], judgments: Dict[str, int]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """按 intent 分组的指标。返回 {intent: {version: metrics}}."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    intents = sorted({e["intent"] for e in by_version[VERSIONS[0]]})
    for intent in intents:
        out[intent] = {}
        for version in VERSIONS:
            entries = [e for e in by_version[version] if e["intent"] == intent]
            p5, mrr_l, ndcg10 = [], [], []
            for entry in entries:
                q = entry["query"]
                scores = [judgments.get(f"{q}\t{r['repo']}", 0) for r in entry["top"]]
                p5.append(precision_at_k(scores, 5, threshold=2))
                mrr_l.append(mrr(scores, threshold=2))
                ndcg10.append(ndcg_at_k(scores, 10))
            out[intent][version] = {
                "P@5": round(mean(p5), 4) if p5 else 0.0,
                "MRR": round(mean(mrr_l), 4) if mrr_l else 0.0,
                "NDCG@10": round(mean(ndcg10), 4) if ndcg10 else 0.0,
                "n_queries": len(entries),
            }
    return out


def print_summary(metrics: Dict[str, Dict[str, float]]):
    print("\n=== 总体指标 ===")
    print(f"{'version':<10} {'P@5':>8} {'MRR':>8} {'NDCG@10':>10} {'n':>4}")
    for v in VERSIONS:
        m = metrics[v]
        print(
            f"{v:<10} {m['P@5']:>8.4f} {m['MRR']:>8.4f} {m['NDCG@10']:>10.4f} {m['n_queries']:>4}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑前 8 个查询验证流程")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    queries = QUERIES[:8] if args.quick else QUERIES
    print(f"评测查询数: {len(queries)}")

    out_dir = ROOT / "tests" / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / "judgments.json"
    summary_path = out_dir / "eval_summary.json"
    detail_path = out_dir / "eval_detail.json"

    by_version, repo_data_cache = collect_results(queries, top_k=args.top_k)

    judgments = judge_pairs(by_version, repo_data_cache, judgments_path)

    metrics = compute_metrics(by_version, judgments)
    breakdown = per_intent_breakdown(by_version, judgments)

    print_summary(metrics)

    summary = {
        "n_queries": len(queries),
        "top_k": args.top_k,
        "overall": metrics,
        "per_intent": breakdown,
        "judge_model": LLMJudge().model,
        "n_judgments": len(judgments),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(by_version, f, ensure_ascii=False, indent=2)

    print(f"\n保存: {summary_path}")
    print(f"保存: {detail_path}")
    print(f"保存: {judgments_path}")


if __name__ == "__main__":
    main()
