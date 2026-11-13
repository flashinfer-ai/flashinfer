"""
FlashInfer-optimized backbone for HunyuanImage-3.0-Instruct (Tencent).

Original model: tencent/HunyuanImage-3.0-Instruct on HuggingFace (loads via
``trust_remote_code=True``). The upstream architecture is a 32-layer MoE
decoder with:

- Grouped-Query Attention (32 Q heads, 8 KV heads, head_dim=128) with QK-norm
  and a custom 2D RoPE that already mixes spatial positions for image tokens.
- Pre-norm RMSNorm (hidden_size=4096, eps=1e-5) on attention input, post-norm
  RMSNorm before the MLP/MoE.
- Mixture of Experts: 64 routed experts top-8, plus 1 shared expert per layer.
  SwiGLU (silu) FFN with intermediate_size=3072. We replace the whole
  ``HunyuanMoE`` with ``FlashInferHunyuanMoE`` (shared ``FlashInferFusedMoE``
  over stacked expert weights; upstream gate kept bit-identical).
- AR text generation reuses the same backbone for diffusion-step image
  denoising through a UNet patch in/out (``patch_embed`` / ``final_layer``);
  the VAE and SigLIP-2 ViT stay on the AR path unchanged.

Optimisations applied here (all built on the shared, model-independent
modules in ``examples/pytorch/flashinfer_modules.py`` — the same set the wan
example uses):
- ``HunyuanRMSNorm`` (hidden and per-head QK-norm) -> shared
  ``FlashInferRMSNorm`` (``flashinfer.rmsnorm``).
- ``nn.Linear`` projections in attention and MLP -> shared
  ``FlashInferLinear`` (any of the GEMM backends from ``flashinfer_modules``).
- SwiGLU (silu+mul) -> ``flashinfer.silu_and_mul``.
- ``HunyuanMoE`` -> ``FlashInferHunyuanMoE``: upstream gate (softmax -> top-8
  -> renormalize) + the shared ``FlashInferFusedMoE``, which dispatches to
  ``flashinfer.fused_moe.cutlass_fused_moe`` (BF16 / per-tensor FP8 /
  SM90 FP8 block-scale / SM90 MXFP4-W4A16 / SM100+ NVFP4) or the
  trtllm-gen routed MoE entry points (BF16 / DeepSeek-FP8 / NVFP4,
  SM100/SM103) over stacked expert weights.
- ``HunyuanImage3SDPAAttention`` -> shared ``FlashInferAttentionDispatcher``
  (single / cudnn / trtllm / torch paths) for mask-less prefill,
  ``flashinfer.decode.single_decode_with_kv_cache`` for decode steps; falls
  back to ``torch.nn.functional.scaled_dot_product_attention`` when a 4D bool
  mask is provided, because the multimodal causal+full mask used during
  ``gen_text``/``gen_image`` prefill has no equivalent on FlashInfer's
  prefill APIs.

The 2D RoPE, KV cache (``HunyuanStaticCache``), VAE/UNet, SigLIP-2 ViT, and
the ``HunyuanImage3ForCausalMM`` generation orchestration are reused as-is
from the upstream ``trust_remote_code`` modules.

Usage::

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "tencent/HunyuanImage-3.0-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    replace_backbone_with_flashinfer(
        model, gemm_backend="bf16", online_act_quant=True,
        attention_backend="auto", moe_backend="cutlass",
    )
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_EXAMPLES_PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_PYTORCH_DIR))

from flashinfer_modules import (  # noqa: E402
    GEMMBackend,
    MoEBackend,
    FlashInferAttentionDispatcher,
    FlashInferFusedMoE,
    FlashInferLinear,
    FlashInferRMSNorm,
    create_linear_layer,
)

try:
    import flashinfer  # noqa: F401
    from flashinfer import silu_and_mul as _flashinfer_silu_and_mul
    from flashinfer.decode import single_decode_with_kv_cache
except Exception as e:  # pragma: no cover
    raise ImportError(
        "FlashInfer is required to use the HunyuanImage-3 FlashInfer example."
    ) from e


# ----------------------------------------------------------------------------
# Config-style options carried through ``replace_backbone_with_flashinfer``.
# Mirrors the fields the wan example exposes via ``WanTransformer3DConfig``.
# ----------------------------------------------------------------------------

_GEMM_BACKENDS = tuple(b.value for b in GEMMBackend)
_MOE_BACKENDS = tuple(b.value for b in MoEBackend) + ("eager",)


@dataclass
class FlashInferBackboneOptions:
    """Knobs for the FlashInfer swap, mirroring the wan example layout."""

    gemm_backend: Literal[
        "torch",
        "bf16",
        "fp8",
        "fp8_sm90",
        "bmm_fp8",
        "fp8_groupwise",
        "fp8_blockscaled",
        "batch_deepgemm_fp8",
        "fp4",
        "bmm_bf16",
        "mxfp8",
        "bmm_mxfp8",
    ] = "torch"
    online_act_quant: bool = True
    # Attention backend used when the caller supplies no attention mask
    # (typical for ``gen_text`` decode steps). When a 4D mask is provided we
    # always fall back to SDPA, because FlashInfer's prefill APIs do not take
    # arbitrary boolean masks. Valid bases match the shared wan dispatcher
    # (``torch`` and its legacy alias ``sdpa`` route through SDPA).
    attention_backend: Literal["auto", "single", "cudnn", "trtllm", "torch", "sdpa"] = (
        "auto"
    )
    # Fused-MoE backend for the routed experts (see
    # ``flashinfer_modules.MoEBackend``): ``cutlass`` (BF16
    # cutlass_fused_moe, SM89/SM90/SM100+), ``cutlass_fp8`` (per-tensor FP8
    # W8A8), ``cutlass_fp8_blockscale`` (DeepSeek-style 128x128 block-scale
    # FP8 W8A8, SM90 only), ``cutlass_w4a16`` (MXFP4 weight-only, SM90
    # only), ``cutlass_nvfp4`` (NVFP4 W4A4, SM100/SM110/SM120/SM121),
    # ``trtllm`` (trtllm-gen BF16, SM100/SM103), ``trtllm_fp8_blockscale``
    # (trtllm-gen DeepSeek FP8, SM100/SM103), ``trtllm_fp4`` (trtllm-gen
    # NVFP4, SM100/SM103), ``torch`` (eager loop inside FlashInferFusedMoE),
    # or ``eager`` to keep the upstream per-expert HunyuanMoE loop entirely.
    moe_backend: Literal[
        "cutlass",
        "cutlass_fp8",
        "cutlass_fp8_blockscale",
        "cutlass_w4a16",
        "cutlass_nvfp4",
        "trtllm",
        "trtllm_fp8_blockscale",
        "trtllm_fp4",
        "torch",
        "eager",
    ] = "cutlass"


_FLASHINFER_ENV_OVERRIDES = {
    "gemm_backend": "FLASHINFER_GEMM_BACKEND",
    "online_act_quant": "FLASHINFER_ONLINE_ACT_QUANT",
    "attention_backend": "FLASHINFER_ATTENTION_BACKEND",
    "moe_backend": "FLASHINFER_MOE_BACKEND",
}

# Legacy ``moe_impl`` values (pre-FlashInferFusedMoE) mapped onto the new
# ``moe_backend`` knob for backward compatibility.
_LEGACY_MOE_IMPL_MAP = {"flashinfer": "cutlass", "eager": "eager"}


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value (1/0, true/false, yes/no, on/off), got {value!r}."
    )


def _resolve_options(**overrides: Any) -> FlashInferBackboneOptions:
    """Build options from defaults + env vars + explicit kwargs.

    Same precedence as the wan example: env vars override defaults; explicit
    keyword arguments override env vars. The legacy ``moe_impl`` knob
    (kwarg or ``FLASHINFER_MOE_IMPL``) is translated onto ``moe_backend``.
    """
    opts = FlashInferBackboneOptions()
    legacy_moe_impl = os.getenv("FLASHINFER_MOE_IMPL")
    if legacy_moe_impl is not None:
        opts.moe_backend = _LEGACY_MOE_IMPL_MAP.get(legacy_moe_impl, legacy_moe_impl)
    for field, env_name in _FLASHINFER_ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        if field == "online_act_quant":
            setattr(opts, field, _env_bool(env_name))
        else:
            setattr(opts, field, raw)
    if "moe_impl" in overrides:
        legacy = overrides.pop("moe_impl")
        # The explicit ``moe_backend`` kwarg wins over the deprecated alias.
        if legacy is not None and overrides.get("moe_backend") is None:
            overrides["moe_backend"] = _LEGACY_MOE_IMPL_MAP.get(legacy, legacy)
    for k, v in overrides.items():
        if v is None:
            continue
        if k not in opts.__dataclass_fields__:
            raise TypeError(f"Unknown FlashInfer backbone option {k!r}.")
        setattr(opts, k, v)
    if opts.gemm_backend not in _GEMM_BACKENDS:
        raise ValueError(
            f"Unsupported gemm_backend {opts.gemm_backend!r}; "
            f"expected one of {_GEMM_BACKENDS}."
        )
    if opts.moe_backend not in _MOE_BACKENDS:
        raise ValueError(
            f"Unsupported moe_backend {opts.moe_backend!r}; "
            f"expected one of {_MOE_BACKENDS}."
        )
    return opts


# ----------------------------------------------------------------------------
# Drop-in replacement modules
# ----------------------------------------------------------------------------


def _device_ctx(t: torch.Tensor):
    """CUDA-device context for the tensor's device (no-op on CPU).

    FlashInfer kernels are launched via raw pointers on the *current* CUDA
    device. Under HuggingFace ``device_map`` sharding the model's layers live
    on cuda:1/2/3, but accelerate's hooks move the activations without changing
    the current device, so a kernel would launch in cuda:0's context against
    cuda:k pointers -> ``misaligned address`` / ``invalid argument``. Wrapping
    each swapped forward in this context pins the launch to the right GPU.
    """
    if t.is_cuda:
        return torch.cuda.device(t.device)
    return contextlib.nullcontext()


class _FlashInferHunyuanMLP(nn.Module):
    """SwiGLU MLP using ``FlashInferLinear`` + ``flashinfer.silu_and_mul``.

    Matches the parameter names of the upstream ``HunyuanMLP``
    (``gate_and_up_proj`` and ``down_proj``) so ``load_state_dict`` from the
    HuggingFace checkpoint works without remapping.

    Notes:
    - Upstream stores the gate and up projections in one fused linear of
      width ``2 * intermediate_size`` and splits them via ``chunk(2, dim=-1)``
      before applying ``silu(x2) * x1``. We replicate that exactly via
      ``flashinfer.silu_and_mul`` which does ``silu(gate) * up`` on a
      ``(..., 2*hidden)`` tensor.
    - We only support the ``silu`` activation (the only one actually used by
      the released checkpoint).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_and_up_proj = create_linear_layer(
            hidden_size,
            2 * intermediate_size,
            bias=bias,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
            device=device,
            dtype=dtype,
        )
        self.down_proj = create_linear_layer(
            intermediate_size,
            hidden_size,
            bias=bias,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_and_up_proj(x)
        # Upstream layout is ``[gate, up]`` along the last dim, and the forward
        # computes ``silu(up) * gate`` (``x1 * silu(x2)`` after chunk(2)).
        # ``flashinfer.silu_and_mul`` consumes ``[gate, up]`` and returns
        # ``silu(gate) * up``. To match upstream we swap halves before calling.
        gate, up = gate_up.chunk(2, dim=-1)
        fused_input = torch.cat([up, gate], dim=-1).contiguous()
        if os.environ.get("FI_DEBUG_MLP"):
            print(
                f"[FI_DEBUG_MLP] silu in shape={tuple(fused_input.shape)} "
                f"dtype={fused_input.dtype} dev={fused_input.device} "
                f"contig={fused_input.is_contiguous()} "
                f"cur_dev={torch.cuda.current_device()}",
                flush=True,
            )
        if fused_input.dtype == torch.float32:
            # ``silu_and_mul`` is BF16/FP16-only; a round-trip through BF16
            # would silently drop float32 precision, so compute the SwiGLU
            # eagerly in float32 instead (same ``x1 * silu(x2)`` convention).
            hidden = gate * F.silu(up)
        else:
            with _device_ctx(fused_input):
                hidden = _flashinfer_silu_and_mul(fused_input)
        return self.down_proj(hidden)


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Same convention as the upstream ``rotate_half``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Matches upstream ``apply_rotary_pos_emb`` (unsqueeze_dim=1).

    q,k shape: ``(B, num_heads, seq, head_dim)``.
    cos,sin shape: ``(B, seq, head_dim)``.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class FlashInferHunyuanImage3Attention(nn.Module):
    """GQA self-attention using FlashInfer for QK-norm, projections, and core attn.

    Matches the upstream ``HunyuanImage3SDPAAttention`` signature so the
    upstream ``HunyuanImage3DecoderLayer`` can call us unchanged.

    Layout: ``num_attention_heads=32``, ``num_key_value_heads=8``,
    ``head_dim=128``. The fused ``qkv_proj`` emits
    ``num_kv_heads * (num_kv_groups + 2) * head_dim`` channels in the
    ``(num_kv_groups + 2)`` block order ``[q_groups, k, v]``.
    """

    def __init__(self, original_attn: nn.Module, opts: FlashInferBackboneOptions):
        super().__init__()
        cfg = original_attn.config
        self.config = cfg
        self.layer_idx = original_attn.layer_idx
        self.attention_type = "self"

        self.attention_dropout = cfg.attention_dropout
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim: int = cfg.attention_head_dim
        self.num_key_value_heads = (
            cfg.num_key_value_heads if cfg.num_key_value_heads else self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = cfg.max_position_embeddings
        self.rope_theta = cfg.rope_theta
        self.is_causal = True
        self.use_qk_norm = cfg.use_qk_norm
        self.use_rotary_pos_emb = cfg.use_rotary_pos_emb
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # FlashInfer-backed Linear projections. The upstream ``qkv_proj`` is a
        # single fused (Q + K + V) Linear, and the output projection ``o_proj``
        # maps the heads back to hidden_size.
        device = next(original_attn.parameters()).device
        dtype = next(original_attn.parameters()).dtype

        self.qkv_proj = create_linear_layer(
            self.hidden_size,
            self.hidden_size_q + 2 * self.hidden_size_kv,
            bias=cfg.attention_bias,
            gemm_backend=opts.gemm_backend,
            online_act_quant=opts.online_act_quant,
            device=device,
            dtype=dtype,
        )
        self.o_proj = create_linear_layer(
            self.hidden_size_q,
            self.hidden_size,
            bias=cfg.attention_bias,
            gemm_backend=opts.gemm_backend,
            online_act_quant=opts.online_act_quant,
            device=device,
            dtype=dtype,
        )

        # Copy weights from the upstream attention. Both projections share
        # the same shape and name as upstream, so a plain state_dict load is
        # safe: FlashInferLinear keeps the underlying ``weight``/``bias``
        # parameters and re-quantizes them in ``prepare_weights``.
        with torch.no_grad():
            _copy_linear_weights(self.qkv_proj, original_attn.qkv_proj)
            _copy_linear_weights(self.o_proj, original_attn.o_proj)

        if self.use_qk_norm:
            # Create on the same device/dtype as this layer's weights. ``copy_``
            # only copies values, so without the explicit ``.to(...)`` these
            # per-head norm weights would stay on CPU and the FlashInfer rmsnorm
            # kernel (raw pointers, no auto host->device move) would fault with
            # "expected device_type=cuda" — especially under device_map sharding
            # where each layer lives on a different GPU.
            self.query_layernorm = FlashInferRMSNorm(
                self.head_dim, eps=cfg.rms_norm_eps
            ).to(device=device, dtype=dtype)
            self.key_layernorm = FlashInferRMSNorm(
                self.head_dim, eps=cfg.rms_norm_eps
            ).to(device=device, dtype=dtype)
            with torch.no_grad():
                self.query_layernorm.weight.copy_(original_attn.query_layernorm.weight)
                self.key_layernorm.weight.copy_(original_attn.key_layernorm.weight)

        # Mask-less prefill goes through the shared wan-style dispatcher
        # ("sdpa" is kept as a legacy alias of its "torch" path).
        # ``use_skip_softmax_sparse`` is not used here: HunyuanImage-3 uses
        # causal attention with custom masks, so the wan-style sparse path
        # doesn't apply.
        backend = opts.attention_backend
        if backend == "sdpa":
            backend = "torch"
        self.attention_dispatcher = FlashInferAttentionDispatcher(
            heads=self.num_heads,
            dim_head=self.head_dim,
            attention_backend=backend,
        )

    # ----- core attention dispatchers --------------------------------------

    def _attention_decode(
        self,
        q: torch.Tensor,  # (B, num_heads, 1, head_dim)
        k: torch.Tensor,  # (B, num_heads, kv_len, head_dim)
        v: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        assert q_len == 1
        if bsz != 1:
            # The HF generation loop is typically bsz=cfg_factor (1 or 2).
            # For bsz>1 fall back to SDPA which handles batched decode
            # without paged KV setup.
            return self._attention_sdpa(q, k, v, attn_mask=None)
        q_2d = q[0, :, 0, :].contiguous()  # (num_heads, head_dim)
        k_nhd = k[0].transpose(0, 1).contiguous()  # (kv_len, num_heads, head_dim)
        v_nhd = v[0].transpose(0, 1).contiguous()
        out = single_decode_with_kv_cache(q_2d, k_nhd, v_nhd)
        # (num_heads, head_dim) -> (1, num_heads, 1, head_dim)
        return out.view(1, self.num_heads, 1, self.head_dim)

    @staticmethod
    def _attention_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # SDPA: q,k,v in (B, num_heads, seq, head_dim) — same layout as inputs.
        if q.device.type == "cuda" and attn_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )

    # ----- forward ----------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = False,
        custom_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        if output_attentions:
            raise NotImplementedError(
                "FlashInferHunyuanImage3Attention does not support "
                "output_attentions=True."
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        # Same split convention as upstream: [num_kv_groups, 1, 1] along dim=3.
        q, k, v = torch.split(qkv, [self.num_key_value_groups, 1, 1], dim=3)
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_qk_norm:
            # Per-head RMSNorm on head_dim=128. Flashinfer rmsnorm needs (M, N).
            q_shape = q.shape
            q = self.query_layernorm(q.reshape(-1, self.head_dim)).view(q_shape)
            k_shape = k.shape
            k = self.key_layernorm(k.reshape(-1, self.head_dim)).view(k_shape)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            q = q.to(k.dtype)

        # Repeat KV heads to match Q heads (GQA).
        if self.num_key_value_groups > 1:
            k = _repeat_kv(k, self.num_key_value_groups)
            v = _repeat_kv(v, self.num_key_value_groups)

        kv_len = k.shape[2]
        # For pure FlashInfer paths we use ``causal=True`` only when we know
        # there's no mask AND we're in a prefill step (q_len == kv_len, no
        # cached prefix). The model uses bidirectional attention inside
        # image blocks via a custom mask, so we only hit the FlashInfer
        # branches in mask-less contexts (decode steps and similar).
        causal = q_len == kv_len

        with _device_ctx(q):
            if attention_mask is not None:
                # Custom 4D bool mask (mixed causal text + full-attention
                # image blocks) has no FlashInfer prefill equivalent.
                attn = self._attention_sdpa(q, k, v, attn_mask=attention_mask)
                attn = attn.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            elif q_len == 1:
                attn = self._attention_decode(q, k, v)
                attn = attn.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            else:
                # Mask-less prefill: the shared dispatcher handles the
                # single / cudnn / trtllm / torch selection (incl. per-SM
                # fallbacks). It consumes (B, S, H, D) and returns
                # (B, S, H*D).
                backend, _ = self.attention_dispatcher._resolve_attention_backend(
                    bsz, q.device
                )
                attn = self.attention_dispatcher._dispatch_attention(
                    backend,
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    bsz,
                    q_len,
                    kv_len,
                    use_sparse=False,
                    causal=causal,
                )

        out = self.o_proj(attn)
        return out, None, past_key_value


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``torch.repeat_interleave(x, dim=1, repeats=n_rep)`` — GQA expansion."""
    bsz, n_kv, seqlen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, n_kv, n_rep, seqlen, head_dim)
    return x.reshape(bsz, n_kv * n_rep, seqlen, head_dim)


def _copy_linear_weights(target: nn.Module, source: nn.Module) -> None:
    """Copy ``nn.Linear``-shaped weights into a ``FlashInferLinear`` (or nn.Linear).

    Both modules store ``weight`` of shape ``(out_features, in_features)`` and
    optionally ``bias`` of shape ``(out_features,)``. ``FlashInferLinear``
    further requires ``prepare_weights()`` after the copy to quantize and
    cache the format expected by the selected GEMM backend; we leave that to
    the caller (``replace_backbone_with_flashinfer``) so it runs once at the
    end on every swapped layer.
    """
    if not hasattr(source, "weight"):
        raise TypeError(f"Source {type(source).__name__} has no 'weight' attribute.")
    target.weight.data.copy_(source.weight.data)
    if getattr(source, "bias", None) is not None and target.bias is not None:
        target.bias.data.copy_(source.bias.data)


# ----------------------------------------------------------------------------
# Mixture of Experts
# ----------------------------------------------------------------------------


class FlashInferHunyuanMoE(nn.Module):
    """HunyuanMoE on the shared ``FlashInferFusedMoE``.

    Keeps the upstream routing exactly: the original ``HunyuanTopKGate``
    module (FP32 ``wg`` linear) is reused as-is with its ``easy_topk``
    implementation (softmax over all experts -> top-k -> renormalize by the
    top-k sum), so routing decisions are bit-identical to the upstream
    ``moe_impl='eager'`` / ``'flashinfer'`` paths. Only the expert
    computation moves onto the fused FlashInfer kernel.

    The shared expert (``use_mixed_mlp_moe``) stays a dense
    ``_FlashInferHunyuanMLP`` — it runs on every token, so it benefits from
    the regular GEMM backends rather than the grouped MoE kernel.

    Upstream ``HunyuanMLP.gate_and_up_proj`` stacks ``[x1, x2]`` with
    ``out = x1 * silu(x2)``, which is exactly ``FlashInferFusedMoE``'s
    weight convention — expert weights are stacked into ``w13_weight``
    without reordering (same as the upstream flashinfer path that fed
    ``cutlass_fused_moe`` directly).
    """

    def __init__(self, original_moe: nn.Module, opts: FlashInferBackboneOptions):
        super().__init__()
        cfg = original_moe.config
        self.config = cfg
        self.layer_idx = original_moe.layer_idx
        self.moe_topk = original_moe.moe_topk
        self.num_experts = original_moe.num_experts
        self.use_mixed_mlp_moe = bool(getattr(cfg, "use_mixed_mlp_moe", False))

        # Reuse the upstream gate (FP32 router linear + easy_topk).
        self.gate = original_moe.gate

        first_expert = original_moe.experts[0]
        if first_expert.hidden_act != "silu":
            raise ValueError(
                f"FlashInferHunyuanMoE only supports SwiGLU experts, got "
                f"hidden_act={first_expert.hidden_act!r}."
            )
        if first_expert.gate_and_up_proj.bias is not None:
            raise ValueError(
                "FlashInferHunyuanMoE does not support expert biases "
                "(the released HunyuanImage-3 checkpoints use mlp_bias=False)."
            )
        hidden_size = first_expert.hidden_size
        # Upstream doubles ``intermediate_size`` for SwiGLU; the per-branch
        # width the MoE kernel wants is half of that.
        intermediate_size = first_expert.intermediate_size // 2

        device = first_expert.gate_and_up_proj.weight.device
        dtype = first_expert.gate_and_up_proj.weight.dtype

        if self.use_mixed_mlp_moe:
            _replace_mlp(self, "shared_mlp", original_moe.shared_mlp, opts)

        self.fused_moe = FlashInferFusedMoE(
            num_experts=self.num_experts,
            top_k=self.moe_topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_backend=opts.moe_backend,
            online_act_quant=opts.online_act_quant,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            for i, expert in enumerate(original_moe.experts):
                self.fused_moe.w13_weight.data[i].copy_(expert.gate_and_up_proj.weight)
                self.fused_moe.w2_weight.data[i].copy_(expert.down_proj.weight)
                # Free the per-expert weights as we go so peak memory stays
                # one expert above the stacked copy (the upstream flashinfer
                # path does the same in _initialize_weights_on_device).
                expert.gate_and_up_proj.weight.data = torch.empty(0, device=device)
                expert.down_proj.weight.data = torch.empty(0, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        with _device_ctx(hidden_states):
            # Upstream gate: FP32 softmax -> top-k -> renormalize.
            topk_weight, topk_index = self.gate(hidden_states, topk_impl="easy")

        reshaped = hidden_states.reshape(-1, hidden_size)
        combined = self.fused_moe(reshaped, topk_index, topk_weight)
        combined = combined.reshape(bsz, seq_len, hidden_size)

        if self.use_mixed_mlp_moe:
            return hidden_states_mlp + combined
        return combined


# ----------------------------------------------------------------------------
# Swap pipeline
# ----------------------------------------------------------------------------


def _replace_rmsnorm(parent: nn.Module, name: str, original: nn.Module) -> None:
    """Replace a single ``HunyuanRMSNorm`` attribute with the shared
    ``FlashInferRMSNorm``, copying the affine weight in-place."""
    eps = getattr(original, "variance_epsilon", 1e-5)
    hidden_size = original.weight.shape[0]
    new = FlashInferRMSNorm(hidden_size, eps=eps)
    with torch.no_grad():
        new.weight.copy_(original.weight)
    # Move to the same device/dtype as the original parameter, then attach.
    new = new.to(device=original.weight.device, dtype=original.weight.dtype)
    setattr(parent, name, new)


def _replace_mlp(
    parent: nn.Module, name: str, original: nn.Module, opts: FlashInferBackboneOptions
) -> None:
    """Replace a single ``HunyuanMLP`` with the FlashInfer SwiGLU MLP.

    The upstream class supports both ``silu`` and ``gelu`` activations, but
    the released HunyuanImage-3 checkpoint only uses ``silu`` (see
    ``config.json``). We only handle that case here.
    """
    if original.hidden_act != "silu":
        warnings.warn(
            f"Skipping MLP swap on layer {name}: hidden_act={original.hidden_act!r} "
            f"is not 'silu'. Only SwiGLU MLPs are FlashInfer-accelerated.",
            stacklevel=2,
        )
        return
    hidden_size = original.hidden_size
    # Upstream sets ``intermediate_size *= 2`` for SwiGLU, so the actual
    # per-branch dim is ``intermediate_size // 2``.
    intermediate = original.intermediate_size // 2

    device = original.gate_and_up_proj.weight.device
    dtype = original.gate_and_up_proj.weight.dtype

    new = _FlashInferHunyuanMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        bias=original.gate_and_up_proj.bias is not None,
        gemm_backend=opts.gemm_backend,
        online_act_quant=opts.online_act_quant,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        _copy_linear_weights(new.gate_and_up_proj, original.gate_and_up_proj)
        _copy_linear_weights(new.down_proj, original.down_proj)
    setattr(parent, name, new)


def _replace_attention(
    parent: nn.Module, name: str, original: nn.Module, opts: FlashInferBackboneOptions
) -> None:
    new = FlashInferHunyuanImage3Attention(original, opts)
    setattr(parent, name, new)


def _prepare_flashinfer_weights(model: nn.Module) -> None:
    """Run ``prepare_weights()`` on every swapped linear and fused MoE.

    Some GEMM backends (FP8 / FP4 / MXFP8 / blockscaled / ...) and the
    quantized / trtllm MoE backends only prepare on the first call. Running
    ``prepare_weights`` up front means the model forward path doesn't pay
    quantisation / weight-shuffle cost on the first inference.
    """
    for mod in model.modules():
        if isinstance(mod, (FlashInferLinear, FlashInferFusedMoE)):
            mod.prepare_weights()


def replace_backbone_with_flashinfer(
    model: nn.Module,
    *,
    gemm_backend: Optional[str] = None,
    online_act_quant: Optional[bool] = None,
    attention_backend: Optional[str] = None,
    moe_backend: Optional[str] = None,
    moe_impl: Optional[str] = None,
    prepare_weights: bool = True,
) -> FlashInferBackboneOptions:
    """Swap the hot kernels in a loaded HunyuanImage-3 model in-place.

    Targets every ``HunyuanImage3DecoderLayer`` in ``model.model.layers`` and
    swaps:

    1. ``input_layernorm`` / ``post_attention_layernorm`` (HunyuanRMSNorm)
       -> shared ``FlashInferRMSNorm``.
    2. ``self_attn`` (HunyuanImage3SDPAAttention)
       -> ``FlashInferHunyuanImage3Attention`` (built on the shared
       ``FlashInferAttentionDispatcher``).
    3. ``mlp`` (HunyuanMoE) -> ``FlashInferHunyuanMoE`` (shared
       ``FlashInferFusedMoE`` over stacked expert weights, upstream gate
       kept as-is); the shared expert becomes a ``_FlashInferHunyuanMLP``.
       With ``moe_backend='eager'`` the upstream HunyuanMoE loop is kept and
       only its per-expert MLPs are swapped.
    4. Dense ``mlp`` (HunyuanMLP) -> ``_FlashInferHunyuanMLP`` (only when
       ``hidden_act == 'silu'``).

    The final ``model.model.ln_f`` (used only for text logits) is also
    swapped. The VAE, SigLIP-2 ViT, time embedders, UNet patch in/out, and
    the ``HunyuanImage3ForCausalMM`` generation orchestration are left
    untouched because they are not on the per-token hot path.

    Args:
        model: A loaded ``HunyuanImage3ForCausalMM`` instance (typically from
            ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``).
        gemm_backend: GEMM backend for swapped linears. See
            ``flashinfer_modules.GEMMBackend`` for options. Default: ``torch``.
        online_act_quant: ``True`` (default) computes activation scales from
            the current tensor; ``False`` uses a fixed default scale. Only
            FP8/FP4-family GEMM backends and the ``cutlass_fp8`` MoE backend
            consult this flag.
        attention_backend: ``auto`` (default), ``single``, ``cudnn``,
            ``trtllm``, ``torch``, or its alias ``sdpa``.
        moe_backend: ``cutlass`` (default; BF16 cutlass_fused_moe,
            SM89/SM90/SM100+), ``cutlass_fp8`` (per-tensor FP8 W8A8),
            ``cutlass_fp8_blockscale`` (DeepSeek-style 128x128 block-scale
            FP8 W8A8, SM90 only), ``cutlass_w4a16`` (MXFP4 weight-only,
            SM90 only), ``cutlass_nvfp4`` (NVFP4 W4A4,
            SM100/SM110/SM120/SM121), ``trtllm`` (trtllm-gen BF16 routed
            MoE, SM100/SM103), ``trtllm_fp8_blockscale`` /
            ``trtllm_fp4`` (trtllm-gen quantized routed MoE, SM100/SM103),
            ``torch`` (eager loop inside FlashInferFusedMoE), or ``eager``
            (keep the upstream per-expert HunyuanMoE loop).
        moe_impl: Deprecated alias — ``flashinfer`` maps to
            ``moe_backend='cutlass'``, ``eager`` to ``moe_backend='eager'``.
        prepare_weights: If True, eagerly quantize every swapped
            ``FlashInferLinear`` / ``FlashInferFusedMoE`` so the first
            forward call doesn't pay the cost. Set False if you intend to
            call ``.to(...)`` afterwards.

    Returns:
        The resolved ``FlashInferBackboneOptions`` (for logging / debugging).
    """
    opts = _resolve_options(
        gemm_backend=gemm_backend,
        online_act_quant=online_act_quant,
        attention_backend=attention_backend,
        moe_backend=moe_backend,
        moe_impl=moe_impl,
    )

    # ``model.model`` is the ``HunyuanImage3Model`` backbone. The
    # ``HunyuanImage3ForCausalMM`` outer module owns the VAE, ViT, etc.
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError(
            f"Expected a HunyuanImage3ForCausalMM-like model with .model.layers, "
            f"got {type(model).__name__}. Did you forget trust_remote_code=True?"
        )

    n_layers = len(backbone.layers)
    logger.info(
        "Replacing %d HunyuanImage3 decoder layers with FlashInfer kernels.", n_layers
    )

    for layer in backbone.layers:
        # 1. Norms
        _replace_rmsnorm(layer, "input_layernorm", layer.input_layernorm)
        _replace_rmsnorm(
            layer, "post_attention_layernorm", layer.post_attention_layernorm
        )

        # 2. Attention
        _replace_attention(layer, "self_attn", layer.self_attn, opts)

        # 3. MLP / MoE
        mlp = layer.mlp
        if hasattr(mlp, "experts"):  # HunyuanMoE
            if opts.moe_backend == "eager":
                # Keep the upstream per-expert loop; just swap each expert's
                # (and the shared expert's) MLP to the FlashInfer version.
                mlp.moe_impl = "eager"
                if getattr(mlp.config, "use_mixed_mlp_moe", False):
                    _replace_mlp(mlp, "shared_mlp", mlp.shared_mlp, opts)
                for i in range(len(mlp.experts)):
                    _replace_mlp(mlp.experts, str(i), mlp.experts[i], opts)
            else:
                layer.mlp = FlashInferHunyuanMoE(mlp, opts)
        else:  # HunyuanMLP (dense layer)
            # Replace the whole MLP in-place.
            _replace_mlp(layer, "mlp", mlp, opts)

    # 4. Final RMSNorm on the LM-head side.
    if hasattr(backbone, "ln_f"):
        _replace_rmsnorm(backbone, "ln_f", backbone.ln_f)

    if prepare_weights:
        _prepare_flashinfer_weights(model)

    return opts


# ----------------------------------------------------------------------------
# Lightweight self-test entrypoint
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Load HunyuanImage-3.0-Instruct, swap the backbone hot paths to "
            "FlashInfer, and report what was replaced. Does not run inference."
        )
    )
    parser.add_argument(
        "--model",
        default="tencent/HunyuanImage-3.0-Instruct",
        help="HuggingFace repo id or local path.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--gemm-backend",
        default=os.getenv("FLASHINFER_GEMM_BACKEND", "torch"),
        choices=_GEMM_BACKENDS,
    )
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        choices=["auto", "single", "cudnn", "trtllm", "torch", "sdpa"],
    )
    parser.add_argument(
        "--moe-backend",
        default=os.getenv("FLASHINFER_MOE_BACKEND", "cutlass"),
        choices=list(_MOE_BACKENDS),
    )
    parser.add_argument("--offline-act-quant", action="store_true")
    parser.add_argument(
        "--skip-prepare-weights",
        action="store_true",
        help="Skip the eager FlashInferLinear.prepare_weights() pass.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    from transformers import AutoModelForCausalLM

    print(f"Loading {args.model} (trust_remote_code=True) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.eval().cuda()

    opts = replace_backbone_with_flashinfer(
        model,
        gemm_backend=args.gemm_backend,
        attention_backend=args.attention_backend,
        moe_backend=args.moe_backend,
        online_act_quant=not args.offline_act_quant,
        prepare_weights=not args.skip_prepare_weights,
    )
    print(f"FlashInfer options applied: {opts}")

    n_flashinfer_linear = sum(
        1 for m in model.modules() if isinstance(m, FlashInferLinear)
    )
    n_flashinfer_rmsnorm = sum(
        1 for m in model.modules() if isinstance(m, FlashInferRMSNorm)
    )
    n_flashinfer_attn = sum(
        1 for m in model.modules() if isinstance(m, FlashInferHunyuanImage3Attention)
    )
    n_flashinfer_moe = sum(
        1 for m in model.modules() if isinstance(m, FlashInferHunyuanMoE)
    )
    print(f"FlashInferLinear modules: {n_flashinfer_linear}")
    print(f"FlashInferRMSNorm modules: {n_flashinfer_rmsnorm}")
    print(f"FlashInferHunyuanImage3Attention modules: {n_flashinfer_attn}")
    print(f"FlashInferHunyuanMoE modules: {n_flashinfer_moe}")
