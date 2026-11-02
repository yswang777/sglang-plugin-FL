"""Runtime compatibility hooks for XingChen4 and Xing4.0."""

from __future__ import annotations

def register_xingchen4_hooks() -> None:
    from .xingchen4_detokenizer import register_xingchen4_detokenizer_hook

    register_xingchen4_detokenizer_hook()
