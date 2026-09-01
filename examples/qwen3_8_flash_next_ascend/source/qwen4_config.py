"""Qwen4 experiment (internal Qwen3.8-Flash-Next) configuration for sglang.

Ported from vllm-plugin-fl-qwen4-day0.patch ``vllm_fl/models/qwen3_8_flash_next/config.py``.
The checkpoint carries a composite ``config.json`` (model_type ``qwen4_exp``) with
``text_config`` (model_type ``qwen4_exp_text``) and ``vision_config`` sub-configs.
"""

from __future__ import annotations

from typing import Any, cast

from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig

_QSA_CONFIG_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)


class Qwen4ExpVisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen4_exp"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen4ExpTextConfig(Qwen3NextConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int | None = None,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_ngram_vocab_size_divisible_by: int = 128,
        output_gate_type: str = "sigmoid",
        rope_parameters: dict[str, Any] | None = None,
        layer_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if hc_count <= 1:
            raise ValueError(
                f"Qwen4-Exp requires hc_count > 1, got {hc_count}."
            )

        # The checkpoint supplies RoPE settings under rope_parameters
        # (mrope_interleaved/mrope_section/rope_theta/partial_rotary_factor).
        # Normalize before parent init so downstream code sees expected values.
        if rope_parameters is not None:
            if kwargs.get("rope_scaling") is None:
                kwargs["rope_scaling"] = rope_parameters
            if kwargs.get("rope_theta") is None and "rope_theta" in rope_parameters:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
            if (
                kwargs.get("partial_rotary_factor") is None
                and "partial_rotary_factor" in rope_parameters
            ):
                kwargs["partial_rotary_factor"] = rope_parameters[
                    "partial_rotary_factor"
                ]

        rope_scaling = kwargs.get("rope_scaling")
        rope_theta = kwargs.get("rope_theta", 10_000.0)
        super().__init__(layer_types=layer_types, **kwargs)

        # sglang Qwen3NextConfig drops layer_types (it recomputes the pattern
        # from full_attention_interval). The checkpoint ships an explicit
        # 48-entry list; keep it for downstream layer construction.
        if layer_types is not None:
            self.layer_types = list(layer_types)

        # Qwen4-Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q. sglang model code reads ``attn_output_gate``.
        self.attn_output_gate = (
            output_gate_type == "sigmoid" if output_gate_type else True
        )
        self.output_gate_type = output_gate_type

        self.rope_scaling = (
            rope_scaling or rope_parameters or getattr(self, "rope_scaling", None)
        )
        self.rope_parameters = rope_parameters or self.rope_scaling
        self.rope_theta = rope_theta

        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.ple_layer_ids = ple_layer_ids or []
        self.ple_embed_dim = (
            self.hidden_size if ple_embed_dim is None else ple_embed_dim
        )
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by

        self._validate_ple_config()
        self._validate_ple_layer_ids()
        self._validate_qsa_config()

    def _validate_ple_config(self) -> None:
        if self.hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {self.hc_lowrank}")
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(
                f"heads_per_ngram must be positive, got {self.heads_per_ngram}"
            )
        if self.ple_embed_dim <= 0:
            raise ValueError(
                f"ple_embed_dim must be positive, got {self.ple_embed_dim}"
            )
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ple_embed_dim} % {ngram_heads} != 0"
            )
        if self.ple_conv_kernel_size <= 0:
            raise ValueError(
                "ple_conv_kernel_size must be positive, got "
                f"{self.ple_conv_kernel_size}"
            )
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be positive")
        if self.make_ngram_vocab_size_divisible_by <= 0:
            raise ValueError("make_ngram_vocab_size_divisible_by must be positive")

    def _validate_ple_layer_ids(self) -> None:
        invalid = [
            layer_id
            for layer_id in self.ple_layer_ids
            if not 1 <= int(layer_id) <= self.num_hidden_layers
        ]
        if invalid:
            raise ValueError(
                "ple_layer_ids are 1-based and must refer to an existing layer; "
                f"got {invalid} for {self.num_hidden_layers} layers"
            )

    def _validate_qsa_config(self) -> None:
        configured = {name: getattr(self, name, None) for name in _QSA_CONFIG_FIELDS}
        if all(value is None for value in configured.values()):
            return

        missing = [name for name, value in configured.items() if value is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")

        values = {name: int(cast(int, value)) for name, value in configured.items()}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "QSA requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > values["indexer_head_dim"]:
            raise ValueError(
                "QSA indexer_head_dim must cover the attention rotary "
                f"dimension, got {values['indexer_head_dim']} < {rotary_dim}"
            )

    @property
    def layers_block_type(self) -> list[str]:
        layer_types = getattr(self, "layer_types", None)
        if not layer_types:
            return [
                "attention"
                if (l + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for l in range(self.num_hidden_layers)
            ]
        return [
            "attention" if layer_type == "full_attention" else layer_type
            for layer_type in layer_types
        ]

    @property
    def short_conv_layer_ids(self) -> list[int]:
        if not self.ple_layer_ids:
            return []
        return sorted({int(layer_id) - 1 for layer_id in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self) -> tuple[int, int] | None:
        if not self.short_conv_layer_ids:
            return None
        ple_state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        ple_channels = self.hidden_size * self.hc_count
        return ple_channels, ple_state_len

    @property
    def ngram_context_len(self) -> int:
        if not self.ple_layer_ids:
            return 0
        return max(int(self.ngram_size) - 1, 0)

    @property
    def spec_hidden_size(self) -> int:
        return int(self.hc_count * self.hidden_size)


class Qwen4ExpConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = [
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
