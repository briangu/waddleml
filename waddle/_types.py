"""Shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RepoInfo:
    id: str
    name: str
    path: str
    origin_url: Optional[str]
    default_branch: str
