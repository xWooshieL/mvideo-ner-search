"""
Переиспользуемая предобработка поисковых запросов M.Video NER.

Импорт из других ноутбуков/скриптов::

    from src.preprocessing import QueryPreprocessor, PreprocessResult
"""

from __future__ import annotations

from .pipeline import (
    PreprocessResult,
    QueryPreprocessor,
    build_model_lexicon_from_titles,
    load_phrase_list,
    save_phrase_list,
    split_glued_alnum,
)
from .spellfix import (
    SpellFixer,
    has_mixed_script,
    keyboard_adjacent,
    load_alias_map,
    normalize_homoglyphs,
    similarity_score,
    weighted_edit_distance,
)

__all__ = [
    "PreprocessResult",
    "QueryPreprocessor",
    "build_model_lexicon_from_titles",
    "load_phrase_list",
    "save_phrase_list",
    "split_glued_alnum",
    "SpellFixer",
    "weighted_edit_distance",
    "similarity_score",
    "keyboard_adjacent",
    "normalize_homoglyphs",
    "has_mixed_script",
    "load_alias_map",
]
