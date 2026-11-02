# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FlagOS mHC implementations backed by FlagGems."""

from __future__ import annotations

from typing import Optional

import torch


def mhc_pre_flagos(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the FlagGems mHC pre kernel.

    ``n_splits_pre`` is accepted for compatibility with SGLang's TileLang
    wrapper. FlagGems selects its own pre-projection strategy.
    """
    del n_splits_pre
    from flag_gems.fused.mhc import mhc_pre

    return mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )


def mhc_post_flagos(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Run the FlagGems mHC post kernel."""
    from flag_gems.fused.mhc import mhc_post

    return mhc_post(x, residual, post_layer_mix, comb_res_mix)
