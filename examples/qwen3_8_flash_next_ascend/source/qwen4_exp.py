# coding=utf-8
"""Qwen4Exp (Qwen-Air / Qwen4 experimental) configuration.

Composite config: `Qwen4ExpConfig` holds a `text_config` (Qwen4ExpTextConfig)
and a `vision_config` (Qwen4ExpVisionConfig). The text config reuses the
Qwen3Next/Gated-Delta-Net machinery in SGLang (including the hybrid linear
attention backend) and adds the Qwen4Exp-specific fields:

- 4-stream Hyper-Connection (hc_count=4, hc_lowrank=320)
- PLE (Prefix Language Expert) layer with ngram embeddings
- QSA full-attention layers (indexer_* fields)
- explicit per-layer attention types (layer_types list)
"""

from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig


class Qwen4ExpVisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen4_exp"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen4ExpTextConfig(Qwen3NextConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"

    def __init__(
        self,
        hc_count=4,
        hc_lowrank=320,
        hc_per_branch_norm=True,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        ngram_size=3,
        ngram_vocab_size_base=20000000,
        heads_per_ngram=8,
        split_ngram_parts=128,
        make_ngram_vocab_size_divisible_by=128,
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ple_layer_ids=None,
        output_gate_type="sigmoid",
        mamba_ssm_dtype="float32",
        mtp=None,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        layer_types=None,
        **kwargs,
    ):
        # HF Qwen4Exp checkpoints provide RoPE settings under rope_parameters.
        # Normalize it before parent init so downstream code sees the expected values.
        rope_parameters = kwargs.pop("rope_parameters", None)
        if kwargs.get("rope_scaling") is None and rope_parameters is not None:
            kwargs["rope_scaling"] = rope_parameters

        super().__init__(layer_types=layer_types, **kwargs)
        if self.rope_scaling is None:
            self.rope_scaling = rope_parameters or {}

        # Keep both names for compatibility with model code paths that read either.
        self.rope_parameters = rope_parameters or self.rope_scaling

        # Hyper-Connection
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        # Checkpoint weights have hc_norm.weight of shape (hc_count * hidden_size,),
        # i.e. one RMS-norm affine per branch concatenated -> per-branch norm.
        self.hc_per_branch_norm = hc_per_branch_norm

        # QSA indexer (full-attention layers)
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio

        # PLE / ngram
        self.ngram_size = ngram_size
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.heads_per_ngram = heads_per_ngram
        self.split_ngram_parts = split_ngram_parts
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.ple_embed_dim = ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ple_layer_ids = ple_layer_ids or []

        # GDN output gate / ssm state dtype
        self.output_gate_type = output_gate_type
        self.mamba_ssm_dtype = mamba_ssm_dtype

        # MTP
        self.mtp = mtp
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        # Explicit per-layer attention types. Qwen3NextConfig declares
        # layer_types as a param but never stores it, so we store it here.
        # full_attention_interval is also not a named param: it only exists as
        # an attribute when passed via kwargs (PretrainedConfig stores unknown
        # kwargs). No-arg construction (e.g. PretrainedConfig.to_diff_dict ->
        # cls() for repr/diff) must not crash, so read it from kwargs instead
        # of the instance and persist it explicitly.
        full_attention_interval = kwargs.get("full_attention_interval", 4)
        if layer_types is None:
            layer_types = []
            for l in range(self.num_hidden_layers):
                if (l + 1) % full_attention_interval == 0:
                    layer_types.append("attention")
                else:
                    layer_types.append("linear_attention")
        self.layer_types = layer_types
        self.full_attention_interval = getattr(
            self, "full_attention_interval", full_attention_interval
        )

    @property
    def layers_block_type(self):
        # Qwen4Exp checkpoints carry an explicit layer_types list; honor it.
        # Canonical strings are "attention" / "linear_attention" (the values of
        # HybridLayerType), which downstream code (model construction, linear
        # attention backend full_attention_layer_ids) expects.
        return [
            "attention" if t == "full_attention" else t
            for t in self.layer_types
        ][: self.num_hidden_layers]


class Qwen4ExpConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`Qwen4ExpForConditionalGeneration`]. It is used to instantiate a Qwen4Exp
    model according to the specified arguments, defining the model
    architecture. Configuration objects inherit from [`PretrainedConfig`] and
    can be used to control the model outputs.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`, *optional*):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*): The image token index.
        video_token_id (`int`, *optional*): The video token index.
        vision_start_token_id (`int`, *optional*): Start-of-vision marker.
        vision_end_token_id (`int`, *optional*): End-of-vision marker.
        language_model_only (`bool`, *optional*): Whether to only load the LM.
        tie_word_embeddings (`bool`, *optional*): Whether to tie embeddings.
    """

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
        language_model_only=False,
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
        self.language_model_only = language_model_only
        self.use_ngram_embedding = kwargs.pop("use_ngram_embedding", True)
        self.ngram_embedding_n = kwargs.pop("ngram_embedding_n", 3)
        self.ngram_embedding_k = kwargs.pop("ngram_embedding_k", 8)

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
