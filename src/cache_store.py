"""
缓存层模块
支持查询缓存、仓库缓存、用户缓存
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    ttl: int  # 秒
    etag: str = ""
    metadata: Dict = None

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "etag": self.etag,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CacheEntry':
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            ttl=data["ttl"],
            etag=data.get("etag", ""),
            metadata=data.get("metadata", {})
        )


class CacheStore:
    """缓存存储"""

    # TTL 配置 (秒)
    TTL_CONFIG = {
        "query": 6 * 3600,      # 查询缓存 6 小时
        "repo": 7 * 86400,      # 仓库缓存 7 天
        "repo_hot": 86400,      # 热门仓库 1 天
        "readme": 7 * 86400,    # README 缓存 7 天
        "repo_tree": 7 * 86400,  # 仓库目录树缓存 7 天
        "owner": 14 * 86400,    # 用户缓存 14 天
        "owner_repos": 7 * 86400,  # 用户仓库 7 天
        "followers": 30 * 86400,   # 关注列表 30 天
        "school_score": 30 * 86400,  # 学校分数 30 天
    }

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # 内存缓存
        self._memory_cache: Dict[str, CacheEntry] = {}

        # 加载磁盘缓存索引
        self._index_file = self.cache_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """加载缓存索引"""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """保存缓存索引"""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f)
        except Exception:
            pass

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str, cache_type: str = "query") -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键
            cache_type: 缓存类型

        Returns:
            缓存值，不存在或过期返回 None
        """
        # 先查内存
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                return entry.value
            else:
                del self._memory_cache[key]

        # 再查磁盘
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entry = CacheEntry.from_dict(data)
                if not entry.is_expired():
                    # 加载到内存
                    self._memory_cache[key] = entry
                    return entry.value
                else:
                    # 删除过期缓存
                    cache_path.unlink(missing_ok=True)
            except Exception:
                pass

        return None

    def set(self, key: str, value: Any, cache_type: str = "query",
            ttl: int = None, etag: str = "", metadata: Dict = None):
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            cache_type: 缓存类型
            ttl: 过期时间 (秒)，None 使用默认值
            etag: ETag
            metadata: 元数据
        """
        if ttl is None:
            ttl = self.TTL_CONFIG.get(cache_type, 3600)

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl,
            etag=etag,
            metadata=metadata
        )

        # 保存到内存
        self._memory_cache[key] = entry

        # 保存到磁盘
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False)
        except Exception:
            pass

    def get_etag(self, key: str) -> Optional[str]:
        """获取缓存的 ETag"""
        if key in self._memory_cache:
            return self._memory_cache[key].etag

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("etag", "")
            except Exception:
                pass
        return None

    def delete(self, key: str):
        """删除缓存"""
        self._memory_cache.pop(key, None)
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)

    def clear(self, cache_type: str = None):
        """清除缓存"""
        if cache_type:
            # 只清除特定类型
            keys_to_delete = []
            for key, entry in self._memory_cache.items():
                if entry.metadata and entry.metadata.get("type") == cache_type:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                self.delete(key)
        else:
            # 清除所有
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "index.json":
                    cache_file.unlink(missing_ok=True)

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        memory_count = len(self._memory_cache)
        disk_count = len(list(self.cache_dir.glob("*.json"))) - 1  # 减去 index.json

        memory_size = sum(
            len(json.dumps(e.value, ensure_ascii=False))
            for e in self._memory_cache.values()
        )

        return {
            "memory_count": memory_count,
            "disk_count": disk_count,
            "memory_size_bytes": memory_size
        }


# 查询缓存键生成
def make_query_cache_key(query: str, route: str, page: int = 1, sort: str = "best-match") -> str:
    """生成查询缓存键"""
    raw = f"{query}|{route}|{page}|{sort}"
    return f"query:{hashlib.md5(raw.encode()).hexdigest()}"


# 仓库缓存键生成
def make_repo_cache_key(owner: str, repo: str) -> str:
    """生成仓库缓存键"""
    return f"repo:{owner}/{repo}"


# 用户缓存键生成
def make_owner_cache_key(username: str) -> str:
    """生成用户缓存键"""
    return f"owner:{username}"


# README 缓存键生成
def make_readme_cache_key(owner: str, repo: str) -> str:
    """生成 README 缓存键"""
    return f"readme:{owner}/{repo}"


# 全局实例
_cache_store = None


def get_cache_store() -> CacheStore:
    """获取缓存存储单例"""
    global _cache_store
    if _cache_store is None:
        _cache_store = CacheStore(".cache")
    return _cache_store
