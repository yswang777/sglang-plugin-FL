# Copyright 2026 FlagOS Contributors

from types import SimpleNamespace

from sglang_fl.dispatch.backends.vendor.mthreads.patches.attention_backend_handler import (
    _use_paged_mla,
)


class _ForwardMode:
    def __init__(self, decode):
        self._decode = decode

    def is_decode(self):
        return self._decode


def test_paged_mla_only_matches_zero_rope_decode(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSA_PAGED_MLA", "1")
    decode = SimpleNamespace(forward_mode=_ForwardMode(True))
    extend = SimpleNamespace(forward_mode=_ForwardMode(False))

    assert _use_paged_mla(SimpleNamespace(qk_rope_head_dim=0), decode)
    assert not _use_paged_mla(SimpleNamespace(qk_rope_head_dim=64), decode)
    assert not _use_paged_mla(SimpleNamespace(qk_rope_head_dim=0), extend)


def test_paged_mla_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSA_PAGED_MLA", "0")
    decode = SimpleNamespace(forward_mode=_ForwardMode(True))

    assert not _use_paged_mla(SimpleNamespace(qk_rope_head_dim=0), decode)
