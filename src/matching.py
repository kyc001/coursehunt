"""
Text matching helpers for course, school, and query signals.

Short ASCII aliases such as AI, OS, DB, C, and IR are useful search signals, but
plain substring matching makes them very noisy. For example, "nankai" contains
"ai" and "computer" contains "c". This module keeps fuzzy substring matching
for Chinese and longer phrases while requiring token boundaries for compact
ASCII aliases and course codes.
"""

import re


_ASCII_TOKEN_RE = re.compile(r"^[a-z0-9+#./-]+$")
_COURSE_CODE_RE = re.compile(r"^[a-z]{2,5}\d{3,5}$")


def contains_signal(text: object, signal: object) -> bool:
    """Return whether signal appears in text with sensible boundary rules."""
    if not text or not signal:
        return False

    text_lower = str(text).lower()
    signal_lower = str(signal).strip().lower()
    if not signal_lower:
        return False

    if _needs_token_boundary(signal_lower):
        pattern = rf"(?<![a-z0-9]){re.escape(signal_lower)}(?![a-z0-9])"
        return re.search(pattern, text_lower) is not None

    return signal_lower in text_lower


def _needs_token_boundary(signal: str) -> bool:
    """Use strict matching for short ASCII aliases and course code-like terms."""
    if _COURSE_CODE_RE.match(signal):
        return True

    if _ASCII_TOKEN_RE.match(signal) and len(signal) <= 4:
        return True

    return False
