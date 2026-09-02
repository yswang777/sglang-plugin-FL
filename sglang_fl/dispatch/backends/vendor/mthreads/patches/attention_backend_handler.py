# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Attention-method routing for GLM-5 zero-RoPE MLA on MUSA."""

from __future__ import annotations

import os


def _use_paged_mla(attn, forward_batch) -> bool:
    return (
        os.getenv("SGLANG_MUSA_PAGED_MLA", "1") == "1"
        and getattr(attn, "qk_rope_head_dim", None) == 0
        and forward_batch.forward_mode.is_decode()
    )


def patch_glm5_attention_backend_handler() -> None:
    """Route zero-RoPE decode through the plugin's paged latent-cache kernel."""
    from sglang.srt.models.deepseek_common.attention_backend_handler import (
        AttentionBackendRegistry,
        handle_attention_fa3 as upstream_handle_attention_fa3,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
        AttnForwardMethod,
    )

    def handle_attention_fa3(attn, forward_batch):
        if _use_paged_mla(attn, forward_batch):
            return AttnForwardMethod.MLA
        return upstream_handle_attention_fa3(attn, forward_batch)

    AttentionBackendRegistry.register("fa3", handle_attention_fa3)
