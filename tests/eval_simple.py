"""极简评测：逐 query 跑，逐 query 打进度，不缓冲。"""
import sys, os, json, time, math, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from dotenv import load_dotenv; load_dotenv()

from src.search_engine import get_search_engine

QUERIES = [
    ("南开 操作系统 lab2", "specific"),
    ("南开 并行程序设计 lab1", "specific"),
    ("南开 编译原理 lab", "specific"),
    ("南开 数据结构 hw1", "specific"),
    ("南开 编译原理", "school_course"),
    ("南开 数据库", "school_course"),
    ("南开 信息检索", "school_course"),
    ("nku compiler", "school_course"),
]

VERSIONS = ["api_only", "+bm25", "full"]
TOP_K = 10
RESULT_DIR = os.path.join(ROOT, "tests", "eval_results")
os.makedirs(RESULT_DIR, exist_ok=True)
JUDGE_PATH = os.path.join(RESULT_DIR, "judgments.json")
DETAIL_PATH = os.path.join(RESULT_DIR, "eval_detail.json")
SUMMARY_PATH = os.path.join(RESULT_DIR, "eval_summary.json")

engine = get_search_engine()
all_repo_data = {}

# ---------- Phase 1: fill corpus ----------
print("Phase 1: fill corpus", flush=True)
engine.enable_bm25 = True
engine.enable_dense = True
for i, (q, lbl) in enumerate(QUERIES):
    print(f"  [{i+1}/{len(QUERIES)}] {q}", flush=True)
    try:
        results = engine.search(q, mode="fast")
        for r in results:
            all_repo_data[r.repo] = r.repo_data
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
    time.sleep(0.2)
print(f"  corpus docs: {engine.corpus.count()}", flush=True)

# ---------- Phase 2: collect ----------
detail = {v: [] for v in VERSIONS}
print("Phase 2: collect", flush=True)
for v in VERSIONS:
    engine.enable_bm25 = v != "api_only"
    engine.enable_dense = v == "full"
    for i, (q, lbl) in enumerate(QUERIES):
        print(f"  [{v}] [{i+1}/{len(QUERIES)}] {q}", flush=True)
        try:
            results = engine.search(q, mode="fast")
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            results = []
        top = [{"repo": r.repo} for r in results[:TOP_K]]
        for r in results[:TOP_K]:
            all_repo_data[r.repo] = r.repo_data
        detail[v].append({"query": q, "intent": lbl, "top": top})
        time.sleep(0.15)

with open(DETAIL_PATH, "w", encoding="utf-8") as f:
    json.dump(detail, f, ensure_ascii=False, indent=2)
print(f"  saved {sum(len(detail[v]) for v in VERSIONS)} entries", flush=True)

# ---------- Phase 3: judge ----------
judgments = {}
if os.path.exists(JUDGE_PATH):
    with open(JUDGE_PATH, "r", encoding="utf-8") as f:
        judgments = json.load(f)
print(f"  cached judgments: {len(judgments)}", flush=True)

new_pairs = []
for v in VERSIONS:
    for entry in detail[v]:
        q = entry["query"]
        for r in entry["top"]:
            key = f"{q}\t{r['repo']}"
            if key not in judgments:
                new_pairs.append((q, r["repo"]))
print(f"  new pairs to judge: {len(new_pairs)}", flush=True)

if new_pairs:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8899/api/v1").rstrip("/")
    for pi, (q, repo) in enumerate(new_pairs):
        rd = all_repo_data.get(repo, {"full_name": repo})
        desc = (rd.get("description") or "")[:200]
        readme = (rd.get("readme_text") or "")[:500]
        prompt = (
            f"Score the relevance of this GitHub repository to the query on a 0-3 scale.\n"
            f"3=perfect match (school+course+assignment all correct), "
            f"2=highly relevant (main dimensions match), "
            f"1=weakly relevant (only one dimension matches), "
            f"0=irrelevant.\n\n"
            f"Query: {q}\n"
            f"Repo: {repo}\n"
            f"Description: {desc}\n"
            f"README excerpt: {readme}\n\n"
            f"Output ONLY one digit (0/1/2/3)."
        )
        score = 0
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "openai/gpt-oss-120b:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,
                        "temperature": 0,
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    for ch in content:
                        if ch in "0123":
                            score = int(ch)
                            break
                    break
            except Exception:
                time.sleep(2)
        judgments[f"{q}\t{repo}"] = score
        if (pi + 1) % 5 == 0:
            print(f"    judged {pi+1}/{len(new_pairs)}", flush=True)
            with open(JUDGE_PATH, "w", encoding="utf-8") as f:
                json.dump(judgments, f, ensure_ascii=False, indent=2)

    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(judgments, f, ensure_ascii=False, indent=2)
    print(f"  total judgments: {len(judgments)}", flush=True)

# ---------- Phase 4: metrics ----------
def p_at_k(scores, k=5, t=2):
    return sum(1 for s in scores[:k] if s >= t) / k

def mrr_fn(scores, t=2):
    for i, s in enumerate(scores, 1):
        if s >= t:
            return 1.0 / i
    return 0.0

def ndcg_at_k(scores, k=10):
    actual = scores[:k]
    if not actual:
        return 0.0
    def dcg(s):
        return sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(s))
    idcg = dcg(sorted(actual, reverse=True))
    return dcg(actual) / idcg if idcg > 0 else 0.0

print("\n=== RESULTS ===", flush=True)
summary = {"n_queries": len(QUERIES), "overall": {}, "per_intent": {}}

for v in VERSIONS:
    p5l, mrl, ndl = [], [], []
    for entry in detail[v]:
        scores = [judgments.get(f"{entry['query']}\t{r['repo']}", 0) for r in entry["top"]]
        p5l.append(p_at_k(scores))
        mrl.append(mrr_fn(scores))
        ndl.append(ndcg_at_k(scores))
    avg = lambda x: sum(x) / len(x) if x else 0
    summary["overall"][v] = {
        "P@5": round(avg(p5l), 4), "MRR": round(avg(mrl), 4),
        "NDCG@10": round(avg(ndl), 4), "n_queries": len(p5l),
    }
    print(f"{v:<12} P@5={avg(p5l):.4f}  MRR={avg(mrl):.4f}  NDCG@10={avg(ndl):.4f}", flush=True)

intents = sorted(set(e["intent"] for e in detail["api_only"]))
for intent in intents:
    print(f"--- {intent} ---", flush=True)
    summary["per_intent"][intent] = {}
    for v in VERSIONS:
        entries = [e for e in detail[v] if e["intent"] == intent]
        p5l, mrl, ndl = [], [], []
        for entry in entries:
            scores = [judgments.get(f"{entry['query']}\t{r['repo']}", 0) for r in entry["top"]]
            p5l.append(p_at_k(scores))
            mrl.append(mrr_fn(scores))
            ndl.append(ndcg_at_k(scores))
        avg = lambda x: sum(x) / len(x) if x else 0
        summary["per_intent"][intent][v] = {
            "P@5": round(avg(p5l), 4), "MRR": round(avg(mrl), 4),
            "NDCG@10": round(avg(ndl), 4), "n_queries": len(p5l),
        }
        print(f"  {v:<12} P@5={avg(p5l):.4f}  MRR={avg(mrl):.4f}  NDCG@10={avg(ndl):.4f}", flush=True)

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {SUMMARY_PATH}", flush=True)
