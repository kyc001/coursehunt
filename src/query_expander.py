"""
Query expansion and search planning helpers.

The user input is an intent, not a GitHub Search query. This module expands a
parsed QueryIntent into a small, ranked set of GitHub-ready queries using:

1. Local school/course/resource knowledge bases.
2. Deterministic templates for word order, aliases, resource terms, and paths.
3. Optional OpenAI-compatible LLM expansion for extra aliases/queries.
4. Pruning rules that control GitHub API budget and filter low-value queries.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

import requests

from .course_kb import get_course_kb
from .matching import contains_signal
from .query_parser import QueryIntent, RESOURCE_TYPE_MAP
from .school_kb import get_school_kb


DEFAULT_RESOURCE_ALIASES = [
    "资料", "笔记", "课件", "试卷", "复习", "作业",
    "notes", "slides", "exam", "review", "homework", "assignment",
]


@dataclass
class ExpandedSearchQuery:
    """A GitHub-ready query candidate with pruning/ranking metadata."""

    text: str
    route: str
    priority: int
    expected_precision: float
    expected_recall: float
    cost: int = 1
    reason: str = ""
    source: str = "rules"


@dataclass
class QueryExpansion:
    """Expanded aliases and selected GitHub queries."""

    school_aliases: List[str]
    course_aliases: List[str]
    resource_aliases: List[str]
    candidates: List[ExpandedSearchQuery]
    selected: List[ExpandedSearchQuery]
    llm_enabled: bool = False
    llm_used: bool = False
    llm_error: str = ""


class QueryExpander:
    """Knowledge-first, optionally LLM-assisted query expander."""

    def __init__(self):
        self.school_kb = get_school_kb()
        self.course_kb = get_course_kb()

    def expand(self, intent: QueryIntent, max_queries: int = 8) -> QueryExpansion:
        school_aliases = self._school_aliases(intent)
        course_aliases = self._course_aliases(intent)
        resource_aliases = self._resource_aliases(intent)

        candidates = self._rule_candidates(intent, school_aliases, course_aliases, resource_aliases)

        llm_enabled = _env_truthy("ENABLE_LLM_QUERY_EXPANSION")
        llm_used = False
        llm_error = ""
        if llm_enabled:
            try:
                llm_aliases, llm_queries = self._llm_candidates(
                    intent, school_aliases, course_aliases, resource_aliases
                )
                school_aliases = _dedupe(school_aliases + llm_aliases.get("school", []))
                course_aliases = _dedupe(course_aliases + llm_aliases.get("course", []))
                resource_aliases = _dedupe(resource_aliases + llm_aliases.get("resource", []))
                candidates.extend(llm_queries)
                llm_used = bool(llm_aliases or llm_queries)
            except Exception as exc:
                llm_error = str(exc)

        selected = self.prune(candidates, intent, max_queries=max_queries)
        return QueryExpansion(
            school_aliases=school_aliases,
            course_aliases=course_aliases,
            resource_aliases=resource_aliases,
            candidates=candidates,
            selected=selected,
            llm_enabled=llm_enabled,
            llm_used=llm_used,
            llm_error=llm_error,
        )

    def prune(self, queries: List[ExpandedSearchQuery], intent: QueryIntent,
              max_queries: int = 8) -> List[ExpandedSearchQuery]:
        """Deduplicate, filter, and budget query candidates."""
        selected: List[ExpandedSearchQuery] = []
        seen = set()
        route_count = defaultdict(int)

        for query in sorted(queries, key=lambda q: (q.priority, -q.expected_precision, q.text)):
            normalized = _normalize_query(query.text)
            if normalized in seen:
                continue
            if route_count[query.route] >= self._route_limit(query.route):
                continue
            if self._too_broad(query, intent):
                continue
            if len(query.text) > 180:
                continue

            selected.append(query)
            seen.add(normalized)
            route_count[query.route] += 1

            if sum(q.cost for q in selected) >= max_queries:
                break

        return selected

    def _school_aliases(self, intent: QueryIntent) -> List[str]:
        if intent.school:
            return self.school_kb.get_all_aliases(intent.school)
        return intent.school_aliases

    def _course_aliases(self, intent: QueryIntent) -> List[str]:
        if intent.course:
            return self.course_kb.get_all_aliases(intent.course)
        return intent.course_aliases or [intent.raw_query]

    def _resource_aliases(self, intent: QueryIntent) -> List[str]:
        aliases = list(intent.resource_aliases)
        if not aliases and intent.resource_type:
            aliases = RESOURCE_TYPE_MAP.get(intent.resource_type, [])
        if not aliases and any(contains_signal(intent.raw_query, word) for word in ["资料", "资源", "material"]):
            aliases = RESOURCE_TYPE_MAP.get("materials", [])
        return _dedupe(aliases + DEFAULT_RESOURCE_ALIASES[:4])

    def _rule_candidates(self, intent: QueryIntent, school_aliases: List[str],
                         course_aliases: List[str], resource_aliases: List[str]) -> List[ExpandedSearchQuery]:
        candidates: List[ExpandedSearchQuery] = []
        schools_zh = _prefer_aliases(school_aliases, ascii_only=False, limit=2)
        schools_en = _prefer_aliases(school_aliases, ascii_only=True, limit=2)
        courses_zh = _prefer_aliases(course_aliases, ascii_only=False, limit=3)
        courses_en = _prefer_aliases(course_aliases, ascii_only=True, limit=3)
        resources_zh = _prefer_aliases(resource_aliases, ascii_only=False, limit=4)
        resources_en = _prefer_aliases(resource_aliases, ascii_only=True, limit=4)

        # Preserve the original expression as a high-priority query, but avoid relying on it alone.
        assignment_suffix = ""
        if intent.assignment_number:
            assignment_suffix = f" {intent.assignment_type or 'lab'}{intent.assignment_number}"
        resource_priority = 2 if intent.resource_type else 3
        resource_precision = 0.86 if intent.resource_type else 0.78

        candidates.append(ExpandedSearchQuery(
            text=f'{intent.raw_query} in:name,description,readme',
            route="original",
            priority=1,
            expected_precision=0.82,
            expected_recall=0.35,
            reason="用户原始输入",
        ))

        # School + course, both word orders, high precision.
        for school in schools_zh[:2]:
            for course in courses_zh[:2]:
                candidates.extend([
                    self._query(f'"{school}" "{course}"{assignment_suffix} in:name,description,readme',
                                "exact_zh", 1, 0.95, 0.35, "中文学校/课程精确匹配"),
                    self._query(f'"{course}" "{school}"{assignment_suffix} in:name,description,readme',
                                "reverse_zh", 1, 0.92, 0.35, "中文课程/学校反序匹配"),
                ])

        for school in schools_en[:2]:
            for course in courses_en[:2]:
                candidates.append(self._query(
                    f'"{school}" "{course}"{assignment_suffix} in:name,description,readme',
                    "english_alias", 2, 0.82, 0.45, "英文/缩写别名匹配"
                ))

        # Resource terms: notes/exams/slides/homework etc.
        resource_pairs = list(zip(resources_zh[:3], resources_en[:3]))
        for school in schools_zh[:1]:
            for course in courses_zh[:2]:
                for resource in resources_zh[:3]:
                    candidates.append(self._query(
                        f'"{school}" "{course}" "{resource}" in:name,description,readme',
                        f"resource_{intent.resource_type or 'generic'}",
                        resource_priority, resource_precision, 0.55, "资料类型扩展"
                    ))

        for school in schools_en[:1]:
            for course in courses_en[:2]:
                for resource in resources_en[:3]:
                    candidates.append(self._query(
                        f'"{school}" "{course}" {resource} in:name,description,readme',
                        "resource_en",
                        resource_priority, max(0.70, resource_precision - 0.08), 0.55, "英文资料类型扩展"
                    ))

        # GitHub naming habits and path search for collection repos.
        for course in _dedupe(courses_zh[:2] + courses_en[:2]):
            candidates.append(self._query(
                f'"{course}" in:path',
                "code_path",
                3, 0.66, 0.62, "目录/文件路径命中课程"
            ))

        for school in _dedupe(schools_en[:2] + schools_zh[:1]):
            for course in _dedupe(courses_en[:2] + courses_zh[:1]):
                candidates.append(self._query(
                    f'{school} {course} course notes in:name,description,readme',
                    "github_style",
                    4, 0.56, 0.70, "GitHub 仓库命名习惯扩展"
                ))

        # Broad course/material queries, useful when school-specific recall is too sparse.
        for course in _dedupe(courses_zh[:1] + courses_en[:1]):
            for resource in _dedupe([r for pair in resource_pairs for r in pair if r])[:3]:
                candidates.append(self._query(
                    f'"{course}" {resource} in:name,description,readme',
                    "broad_material",
                    5, 0.42, 0.78, "宽召回：课程 + 资料类型"
                ))

        if intent.assignment_number:
            assign_type = intent.assignment_type or "lab"
            for school in _dedupe(schools_zh[:1] + schools_en[:1]):
                for course in _dedupe(courses_zh[:1] + courses_en[:1]):
                    candidates.append(self._query(
                        f'"{school}" "{course}" {assign_type}{intent.assignment_number} in:name,description,readme',
                        "assignment_number",
                        2, 0.90, 0.45, "作业/实验编号扩展"
                    ))

        return candidates

    def _llm_candidates(self, intent: QueryIntent, school_aliases: List[str],
                        course_aliases: List[str], resource_aliases: List[str]):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {}, []

        base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8899/api/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b:free")
        prompt = self._llm_prompt(intent, school_aliases, course_aliases, resource_aliases)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in base_url:
            headers.setdefault("HTTP-Referer", os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"))
            headers.setdefault("X-Title", os.getenv("OPENROUTER_APP_NAME", "CourseRepoFinder"))

        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You generate concise JSON for GitHub course-resource search expansion."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 900,
            },
            timeout=20,
        )
        if response.status_code != 200:
            raise RuntimeError(f"LLM API error: {response.status_code}")

        content = response.json()["choices"][0]["message"]["content"]
        data = _load_json_object(content)
        aliases = data.get("aliases", {}) if isinstance(data, dict) else {}
        queries = []
        for item in data.get("queries", [])[:16]:
            text = item.get("query") or item.get("text")
            if not text:
                continue
            queries.append(ExpandedSearchQuery(
                text=text,
                route=item.get("route", "llm"),
                priority=int(item.get("priority", 4)),
                expected_precision=float(item.get("expected_precision", 0.55)),
                expected_recall=float(item.get("expected_recall", 0.55)),
                reason=item.get("reason", "LLM 补充搜索表达"),
                source="llm",
            ))

        return aliases, queries

    def _llm_prompt(self, intent: QueryIntent, school_aliases: List[str],
                    course_aliases: List[str], resource_aliases: List[str]) -> str:
        return json.dumps({
            "task": "Expand a university course-resource search intent into GitHub Search queries.",
            "rules": [
                "Return JSON only.",
                "Do not return more than 12 queries.",
                "Prefer existing known aliases; only add plausible aliases.",
                "Every query must include a course signal.",
                "If a school is specified, most queries should include a school signal.",
                "Avoid broad queries such as only materials/homework/notes.",
            ],
            "input": intent.to_dict(),
            "known_aliases": {
                "school": school_aliases[:8],
                "course": course_aliases[:10],
                "resource": resource_aliases[:10],
            },
            "output_schema": {
                "aliases": {"school": [], "course": [], "resource": []},
                "queries": [
                    {
                        "query": "\"南开大学\" \"高数\"",
                        "route": "llm_exact",
                        "priority": 2,
                        "expected_precision": 0.8,
                        "expected_recall": 0.5,
                        "reason": "why this query helps",
                    }
                ],
            },
        }, ensure_ascii=False)

    @staticmethod
    def _query(text: str, route: str, priority: int, precision: float,
               recall: float, reason: str) -> ExpandedSearchQuery:
        return ExpandedSearchQuery(
            text=text,
            route=route,
            priority=priority,
            expected_precision=precision,
            expected_recall=recall,
            reason=reason,
        )

    def _too_broad(self, query: ExpandedSearchQuery, intent: QueryIntent) -> bool:
        text = query.text
        course_aliases = self._course_aliases(intent)
        has_course = any(contains_signal(text, alias) for alias in course_aliases if alias)
        if not has_course:
            return True

        if intent.school and query.priority <= 4:
            school_aliases = self._school_aliases(intent)
            has_school = any(contains_signal(text, alias) for alias in school_aliases if alias)
            if not has_school and query.route not in {"code_path", "broad_material"}:
                return True

        resource_only = all(
            contains_signal(" ".join(DEFAULT_RESOURCE_ALIASES), token)
            for token in re.findall(r"[\w\u4e00-\u9fff]+", text)
        )
        return resource_only

    @staticmethod
    def _route_limit(route: str) -> int:
        limits = {
            "exact_zh": 2,
            "reverse_zh": 1,
            "english_alias": 2,
            "resource_materials": 2,
            "resource_notes": 2,
            "resource_exam": 2,
            "resource_generic": 2,
            "resource_en": 2,
            "code_path": 2,
            "broad_material": 2,
            "github_style": 2,
            "original": 1,
        }
        return limits.get(route, 2)


_expander = None


def get_query_expander() -> QueryExpander:
    global _expander
    if _expander is None:
        _expander = QueryExpander()
    return _expander


def expand_query(intent: QueryIntent, max_queries: int = 8) -> QueryExpansion:
    return get_query_expander().expand(intent, max_queries=max_queries)


def _prefer_aliases(aliases: Iterable[str], ascii_only: bool, limit: int) -> List[str]:
    selected = []
    for alias in aliases:
        if not alias:
            continue
        is_ascii = str(alias).isascii()
        if is_ascii == ascii_only:
            selected.append(str(alias))
    if not selected and not ascii_only:
        selected = [str(a) for a in aliases if a]
    return _dedupe(selected)[:limit]


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item is None:
            continue
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace('"', "")).strip()


def _load_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
