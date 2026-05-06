"""
本地语料库（持久化索引层）

CourseHunt 把 GitHub Search API 召回的候选仓库的文本字段（仓库名、描述、
README、目录路径、topics）累积持久化到 SQLite，作为自建 BM25 索引和 BGE-M3
向量召回的共享底座。这一层让"重复查询"和"近邻查询"能从历史数据中获益，
而不必每次都依赖 GitHub API 实时召回。

设计要点：
1. SQLite 单表 `documents`，主键为 GitHub 全名 `owner/repo`
2. 文本字段统一拼接为 doc_text，供 BM25 与 Dense 检索共用
3. embedding 单独列 BLOB，懒计算（首次需要 Dense 检索时再调 API 写回）
4. 写入时记录 updated_at，方便后续做 TTL/refresh
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_DB_LOCK = threading.RLock()


@dataclass
class CorpusDocument:
    """语料库文档（一个仓库对应一条）。"""

    full_name: str
    name: str = ""
    description: str = ""
    readme: str = ""
    topics: List[str] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    language: str = ""
    stars: int = 0
    forks: int = 0
    pushed_at: str = ""
    updated_at: float = 0.0
    embedding: Optional[List[float]] = None

    def doc_text(self) -> str:
        """BM25 与 Dense 共用的拼接文本。"""
        parts = [
            self.full_name,
            self.name,
            self.description,
            " ".join(self.topics),
            " ".join(self.paths[:200]),  # 路径列表可能很长，截断
            self.readme,
        ]
        return " ".join(p for p in parts if p)


class LocalCorpus:
    """SQLite 语料库存储。"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS documents (
        full_name TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        readme TEXT,
        topics TEXT,
        paths TEXT,
        language TEXT,
        stars INTEGER,
        forks INTEGER,
        pushed_at TEXT,
        updated_at REAL,
        embedding BLOB
    );
    """

    def __init__(self, db_path: str = ".cache/local_corpus.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _DB_LOCK:
            with self._connect() as conn:
                conn.executescript(self.SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, doc: CorpusDocument):
        """写入或更新一篇文档（不覆盖已有 embedding，除非显式提供）。"""
        with _DB_LOCK:
            with self._connect() as conn:
                existing_emb = None
                if doc.embedding is None:
                    row = conn.execute(
                        "SELECT embedding FROM documents WHERE full_name = ?",
                        (doc.full_name,),
                    ).fetchone()
                    if row and row["embedding"] is not None:
                        existing_emb = row["embedding"]

                emb_blob = (
                    self._encode_embedding(doc.embedding)
                    if doc.embedding is not None
                    else existing_emb
                )

                conn.execute(
                    """
                    INSERT INTO documents (
                        full_name, name, description, readme, topics, paths,
                        language, stars, forks, pushed_at, updated_at, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(full_name) DO UPDATE SET
                        name = excluded.name,
                        description = excluded.description,
                        readme = CASE
                            WHEN excluded.readme != '' THEN excluded.readme
                            ELSE documents.readme
                        END,
                        topics = excluded.topics,
                        paths = CASE
                            WHEN excluded.paths != '[]' THEN excluded.paths
                            ELSE documents.paths
                        END,
                        language = excluded.language,
                        stars = excluded.stars,
                        forks = excluded.forks,
                        pushed_at = excluded.pushed_at,
                        updated_at = excluded.updated_at,
                        embedding = COALESCE(excluded.embedding, documents.embedding)
                    """,
                    (
                        doc.full_name,
                        doc.name,
                        doc.description,
                        doc.readme,
                        json.dumps(doc.topics, ensure_ascii=False),
                        json.dumps(doc.paths, ensure_ascii=False),
                        doc.language,
                        doc.stars,
                        doc.forks,
                        doc.pushed_at,
                        doc.updated_at,
                        emb_blob,
                    ),
                )
                conn.commit()

    def upsert_many(self, docs: Iterable[CorpusDocument]):
        for doc in docs:
            self.upsert(doc)

    def update_embedding(self, full_name: str, embedding: List[float]):
        """单独写入向量（避免和文本字段竞争 readme 等覆盖逻辑）。"""
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE documents SET embedding = ? WHERE full_name = ?",
                    (self._encode_embedding(embedding), full_name),
                )
                conn.commit()

    def get(self, full_name: str) -> Optional[CorpusDocument]:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE full_name = ?",
                    (full_name,),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_doc(row)

    def list_all(self, with_embedding: bool = False) -> List[CorpusDocument]:
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM documents").fetchall()
                return [self._row_to_doc(r, with_embedding=with_embedding) for r in rows]

    def list_missing_embeddings(self) -> List[str]:
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT full_name FROM documents WHERE embedding IS NULL"
                ).fetchall()
                return [r["full_name"] for r in rows]

    def count(self) -> int:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
                return int(row["c"]) if row else 0

    def stats(self) -> Dict[str, int]:
        with _DB_LOCK:
            with self._connect() as conn:
                total = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
                with_readme = conn.execute(
                    "SELECT COUNT(*) AS c FROM documents WHERE readme != ''"
                ).fetchone()["c"]
                with_emb = conn.execute(
                    "SELECT COUNT(*) AS c FROM documents WHERE embedding IS NOT NULL"
                ).fetchone()["c"]
                return {
                    "total_docs": int(total),
                    "with_readme": int(with_readme),
                    "with_embedding": int(with_emb),
                }

    @staticmethod
    def _encode_embedding(embedding: List[float]) -> bytes:
        # 用紧凑的 JSON-bytes 存储，1024 维的 float32 约 16KB；可读、跨平台
        return json.dumps(embedding).encode("utf-8")

    @staticmethod
    def _decode_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
        if blob is None:
            return None
        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return None

    @classmethod
    def _row_to_doc(cls, row: sqlite3.Row, with_embedding: bool = False) -> CorpusDocument:
        topics = json.loads(row["topics"]) if row["topics"] else []
        paths = json.loads(row["paths"]) if row["paths"] else []
        embedding = cls._decode_embedding(row["embedding"]) if with_embedding else None
        return CorpusDocument(
            full_name=row["full_name"],
            name=row["name"] or "",
            description=row["description"] or "",
            readme=row["readme"] or "",
            topics=topics,
            paths=paths,
            language=row["language"] or "",
            stars=int(row["stars"] or 0),
            forks=int(row["forks"] or 0),
            pushed_at=row["pushed_at"] or "",
            updated_at=float(row["updated_at"] or 0.0),
            embedding=embedding,
        )


# ----- 单例 -----

_corpus: Optional[LocalCorpus] = None


def get_local_corpus() -> LocalCorpus:
    global _corpus
    if _corpus is None:
        _corpus = LocalCorpus()
    return _corpus


def doc_from_repo_data(repo_data: dict) -> CorpusDocument:
    """从 search_engine 中流转的 repo_data 字典构建语料库文档。"""
    import time

    full_name = repo_data.get("full_name") or ""
    if not full_name:
        return CorpusDocument(full_name="")

    paths = []
    for path in repo_data.get("tree_paths") or []:
        if path:
            paths.append(str(path))
    for match in repo_data.get("code_path_matches") or []:
        p = match.get("path") if isinstance(match, dict) else None
        if p:
            paths.append(str(p))

    return CorpusDocument(
        full_name=full_name,
        name=repo_data.get("name") or "",
        description=repo_data.get("description") or "",
        readme=repo_data.get("readme_text") or "",
        topics=[str(t) for t in (repo_data.get("topics") or []) if t],
        paths=paths,
        language=repo_data.get("language") or "",
        stars=int(repo_data.get("stargazers_count") or 0),
        forks=int(repo_data.get("forks_count") or 0),
        pushed_at=str(repo_data.get("pushed_at") or ""),
        updated_at=time.time(),
    )
