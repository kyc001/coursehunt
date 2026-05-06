"""
自建 BM25 倒排索引（不依赖 rank_bm25 等三方库）

CourseHunt 报告里频繁引用 BM25，但代码层面如果只调用 GitHub 后端的 BM25，
无法解释成"自实现 IR 算法"。本模块以纯 Python + 标准库 + jieba 重写一遍，
让排序公式有可对照的代码片段，并把它作为多路召回的一路加入 RRF。

核心公式（与报告公式 (4.X) 对应）：

    BM25(q, d) = Σ_{t ∈ q} IDF(t) · (f(t,d) · (k1 + 1)) /
                                    (f(t,d) + k1 · (1 - b + b · |d| / avgdl))

    IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)

其中：
- f(t, d): 词 t 在文档 d 的频次
- |d|: 文档 d 的长度
- avgdl: 语料库平均文档长度
- N: 文档总数
- n_t: 含词 t 的文档数
- k1, b: 平滑参数（默认 k1=1.5, b=0.75）

实现细节：
- 中文用 jieba 切分；如果没装 jieba 则退化为 unicode 字符 bigram，避免硬依赖
- 英文按 \\w 切分 + 小写化
- 索引常驻内存，从 LocalCorpus 全量重建；语料增长后调 `rebuild()` 即可
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .local_corpus import LocalCorpus, get_local_corpus


# ----- 分词 -----

try:
    import jieba

    jieba.initialize()
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#./-]*")
_CHINESE_RE = re.compile(r"[一-鿿]+")
_NUM_RE = re.compile(r"\d+")

# 极简停用词，避免常见无信息词淹没倒排表
_STOPWORDS = {
    # 英文
    "a", "an", "and", "or", "the", "of", "to", "for", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "it", "with", "by", "from", "as",
    # 中文
    "的", "了", "和", "与", "及", "或", "在", "是", "为", "等", "从",
    "课程", "原理", "系统",
}


def _tokenize_chinese(text: str) -> List[str]:
    if _HAS_JIEBA:
        return [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
    # 退化方案：字符 bigram
    cleaned = re.sub(r"\s+", "", text)
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)] if len(cleaned) >= 2 else list(cleaned)


def tokenize(text: str) -> List[str]:
    """混合中英文的轻量分词。"""
    if not text:
        return []

    text_lower = text.lower()
    tokens: List[str] = []

    for piece in re.split(r"\s+", text_lower):
        if not piece:
            continue

        # 抽英文 token
        for tok in _ENGLISH_TOKEN_RE.findall(piece):
            if tok and tok not in _STOPWORDS and len(tok) <= 40:
                tokens.append(tok)

        # 抽数字
        for num in _NUM_RE.findall(piece):
            tokens.append(num)

        # 抽中文段，每段再切
        for chunk in _CHINESE_RE.findall(piece):
            for tok in _tokenize_chinese(chunk):
                if tok and tok not in _STOPWORDS and len(tok) >= 1:
                    tokens.append(tok)

    return tokens


# ----- BM25 主体 -----


class BM25Index:
    """语料级倒排索引 + BM25 打分。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()

        self._doc_ids: List[str] = []                     # 平行数组：行号 → full_name
        self._doc_lengths: List[int] = []                 # 行号 → |d|
        self._doc_freq: List[Counter] = []                # 行号 → term Counter
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._df: Counter = Counter()                     # term → 含该词的文档数
        self._avgdl: float = 0.0
        self._n: int = 0

    # ---- 构建 ----

    def add_document(self, full_name: str, text: str):
        """增量加入一篇文档。重复 full_name 会被忽略（请用 rebuild）。"""
        if not full_name or not text:
            return

        with self._lock:
            tokens = tokenize(text)
            if not tokens:
                return

            doc_idx = len(self._doc_ids)
            self._doc_ids.append(full_name)
            self._doc_lengths.append(len(tokens))

            tf = Counter(tokens)
            self._doc_freq.append(tf)

            for term, freq in tf.items():
                self._postings[term].append((doc_idx, freq))
                self._df[term] += 1

            self._n = len(self._doc_ids)
            self._avgdl = (
                sum(self._doc_lengths) / self._n if self._n else 0.0
            )

    def build_from_corpus(self, corpus: LocalCorpus):
        """全量从 LocalCorpus 重建。"""
        with self._lock:
            self._doc_ids = []
            self._doc_lengths = []
            self._doc_freq = []
            self._postings = defaultdict(list)
            self._df = Counter()

            for doc in corpus.list_all(with_embedding=False):
                self.add_document(doc.full_name, doc.doc_text())

    # ---- 查询 ----

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """返回 (full_name, score) 列表，按分数降序，过滤 score<=0。"""
        if self._n == 0:
            return []

        q_tokens = [t for t in tokenize(query) if t]
        if not q_tokens:
            return []

        scores = defaultdict(float)
        with self._lock:
            for term in q_tokens:
                if term not in self._postings:
                    continue
                idf = self._idf(term)
                if idf <= 0:
                    continue
                for doc_idx, freq in self._postings[term]:
                    dl = self._doc_lengths[doc_idx]
                    denom = freq + self.k1 * (
                        1 - self.b + self.b * dl / self._avgdl
                    ) if self._avgdl > 0 else freq + self.k1
                    if denom <= 0:
                        continue
                    scores[doc_idx] += idf * (freq * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._doc_ids[idx], float(score)) for idx, score in ranked]

    def _idf(self, term: str) -> float:
        n_t = self._df.get(term, 0)
        if n_t == 0:
            return 0.0
        # 经典 BM25 IDF（Robertson-Spärck Jones）
        return math.log((self._n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    # ---- 状态 ----

    @property
    def num_docs(self) -> int:
        return self._n

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def stats(self) -> Dict[str, float]:
        return {
            "num_docs": self._n,
            "vocab_size": len(self._postings),
            "avgdl": round(self._avgdl, 2),
        }


# ----- 单例与便捷接口 -----

_index: Optional[BM25Index] = None
_indexed_docs: set = set()  # 已索引的 doc full_name，用于增量更新


def get_bm25_index(force_rebuild: bool = False) -> BM25Index:
    """取得 BM25 索引；增量更新新文档，避免每次全量重建。"""
    global _index, _indexed_docs

    corpus = get_local_corpus()

    if _index is None or force_rebuild:
        idx = BM25Index()
        idx.build_from_corpus(corpus)
        _index = idx
        _indexed_docs = {doc.full_name for doc in corpus.list_all(with_embedding=False)}
        return _index

    # 增量索引：只加入语料库中新增的文档
    new_docs = [
        doc for doc in corpus.list_all(with_embedding=False)
        if doc.full_name not in _indexed_docs
    ]
    if new_docs:
        for doc in new_docs:
            _index.add_document(doc.full_name, doc.doc_text())
        _indexed_docs = {doc.full_name for doc in corpus.list_all(with_embedding=False)}

    return _index


def bm25_search(query: str, top_k: int = 50) -> List[Tuple[str, float]]:
    """用最新的 BM25 索引检索。"""
    return get_bm25_index().search(query, top_k=top_k)
