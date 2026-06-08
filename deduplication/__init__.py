"""
Deduplication module for exact, near-duplicate, and semantic deduplication.
"""

from .deduplicator import Deduplicator
from .exact_dedup import ExactDeduplicator
from .minhash_dedup import MinHashDeduplicator
from .semantic_dedup import SemanticDeduplicator

__all__ = [
    "Deduplicator",
    "ExactDeduplicator",
    "MinHashDeduplicator",
    "SemanticDeduplicator"
]
