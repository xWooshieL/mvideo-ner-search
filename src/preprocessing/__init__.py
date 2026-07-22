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

__all__ = [
    "PreprocessResult",
    "QueryPreprocessor",
    "build_model_lexicon_from_titles",
    "load_phrase_list",
    "save_phrase_list",
    "split_glued_alnum",
]
