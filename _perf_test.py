"""本地性能测试脚本"""
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.search_engine import get_search_engine

engine = get_search_engine()

# 跑两轮相同查询，验证增量索引效果
for round_num in (1, 2):
    print(f"\n{'#'*60}")
    print(f"# 第 {round_num} 轮查询")
    print(f"{'#'*60}")
    for mode in ("fast",):
        for query in ("南开 并行程序设计 lab2",):
            print(f"\n{'='*60}")
            print(f"mode={mode}, query={query}")
            t0 = time.perf_counter()
            results = engine.search(query, mode=mode)
            total = time.perf_counter() - t0
            trace = engine.last_trace
            print(f"总耗时: {total:.2f}s  结果数: {len(results)}")

            steps = [
                ("查询解析", trace.t_query_parse),
                ("构建计划", trace.t_build_plan),
                ("API多路搜索", trace.t_execute_api),
                ("种子仓库", trace.t_seed_repos),
                ("RRF第一轮", trace.t_rrf_first),
                ("丰富README", trace.t_enrich),
                ("写入语料库", trace.t_persist),
                ("BM25检索", trace.t_bm25),
                ("Embedding补算", trace.t_dense_embed),
                ("Dense搜索", trace.t_dense_search),
                ("RRF三路", trace.t_rrf_final),
                ("候选合并", trace.t_merge),
                ("用户画像", trace.t_owner_profile),
                ("证据构建", trace.t_evidence),
                ("混合排序", trace.t_ranking),
                ("构建结果", trace.t_build_results),
            ]
            max_t = max(s[1] for s in steps if s[1] > 0.001) or 1
            for name, t in steps:
                if t > 0.005:
                    bar = "#" * int(t / max_t * 40)
                    print(f"  {name:14s} |{bar:40s}| {t:.2f}s")

            io_t = (trace.t_execute_api + trace.t_seed_repos + trace.t_enrich
                    + trace.t_owner_profile + trace.t_dense_embed + trace.t_dense_search)
            print(f"  I/O总计: {io_t:.2f}s  CPU总计: {total - io_t:.2f}s")
            print(f"  API路由数: {trace.api_routes}  候选数: {trace.api_candidates}")
