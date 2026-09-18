"""cache.py — Cache helpers (core agents only)."""

from __future__ import annotations

from typing import Callable

from agent.storage.queries.members import (
    find_member_by_identity,
    get_member_contact,
)


def _clear(fn: Callable) -> None:
    try:
        fn.cache_clear()
    except AttributeError:
        pass


def clear_caches() -> None:
    for fn in (find_member_by_identity, get_member_contact):
        _clear(fn)
