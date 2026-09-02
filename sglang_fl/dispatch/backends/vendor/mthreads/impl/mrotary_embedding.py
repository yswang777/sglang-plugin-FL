# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MUSA MRotaryEmbedding operator implementation.
#
# Reuse SGLang's device-agnostic fused Triton path on MUSA.  The dispatch shim
# is required because stock SGLang exposes this implementation through
# ``forward_triton`` rather than a dedicated ``forward_musa`` entry point.

from __future__ import annotations

import os
from typing import Tuple

import torch


def mrotary_embedding_musa(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-modal rotary position embedding on MUSA.

    Args:
        obj: The MRotaryEmbedding instance
        positions: Position tensor (1D or 2D for multi-modal)
        query: Query tensor
        key: Key tensor

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    if int(os.environ.get("SGLANG_FL_MROPE_FORCE_TORCH", "0")) or not getattr(
        obj, "mrope_section", None
    ):
        from sglang_fl.dispatch.backends.reference.impl.mrotary_embedding import (
            mrotary_embedding_torch,
        )

        return mrotary_embedding_torch(obj, positions, query, key)

    # Text-only decode supplies [T].  The fused kernel consumes [3, T]; a
    # zero-stride expanded view avoids an allocation and gives all axes the
    # same positions.
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError(
            "MRotaryEmbedding expects positions [T] or [3, T], "
            f"got {tuple(positions.shape)}"
        )

    # SGLang's fused wrapper accepts flattened Q/K.  Preserve the caller's
    # head-shaped view and the in-place semantics around that wrapper.
    query_shape = query.shape
    key_shape = key.shape
    query_2d = query.reshape(query.shape[0], -1)
    key_2d = key.reshape(key.shape[0], -1)
    query_2d, key_2d = obj.forward_triton(positions, query_2d, key_2d)
    return query_2d.reshape(query_shape), key_2d.reshape(key_shape)
