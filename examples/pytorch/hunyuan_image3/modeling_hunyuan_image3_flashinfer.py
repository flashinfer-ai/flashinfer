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
  SwiGLU (silu) FFN with intermediate_size=3072. The upstream
  ``HunyuanMoE`` already knows how to call ``flashinfer.fused_moe.cutlass_fused_moe``
  when ``moe_impl == 'flashinfer'``; we just stack the weights and flip the flag.
- AR text generation reuses the same backbone for diffusion-step image
  denoising through a UNet patch in/out (``patch_embed`` / ``final_layer``);
  the VAE and SigLIP-2 ViT stay on the AR path unchanged.

Optimisations applied here:
- ``HunyuanRMSNorm`` (hidden) -> ``flashinfer.rmsnorm``
- ``HunyuanRMSNorm`` (per-head QK-norm, head_dim=128) -> ``flashinfer.rmsnorm``
  on the flattened ``(batch*heads*seq, head_dim)`` tensor.
- ``nn.Linear`` projections in attention and MLP -> ``FlashInferLinear`` (any
  of the GEMM backends from ``flashinfer_modules``).
- SwiGLU (silu+mul) -> ``flashinfer.silu_and_mul``.
- ``HunyuanMoE`` -> upstream class with ``moe_impl='flashinfer'`` (calls
  ``flashinfer.fused_moe.cutlass_fused_moe`` on stacked expert weights).
- ``HunyuanImage3SDPAAttention`` -> FlashInfer prefill / decode path when the
  caller doesn't pass a custom attention mask (decode steps and mask-less
  prefill); falls back to ``torch.nn.functional.scaled_dot_product_attention``
  when a 4D bool mask is provided, because the multimodal causal+full mask
  used during ``gen_text``/``gen_image`` prefill has no equivalent on
  FlashInfer's prefill APIs.

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
        attention_backend="auto",
    )
"""

from __future__ import annotations

import contextlib
import logging
import math
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
    FlashInferLinear,
    create_linear_layer,
)

try:
    import flashinfer  # noqa: F401
    from flashinfer import rmsnorm as _flashinfer_rmsnorm
    from flashinfer import silu_and_mul as _flashinfer_silu_and_mul
    from flashinfer.prefill import single_prefill_with_kv_cache
    from flashinfer.decode import single_decode_with_kv_cache
    from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
except Exception as e:  # pragma: no cover
    raise ImportError(
        "FlashInfer is required to use the HunyuanImage-3 FlashInfer example."
    ) from e


# ----------------------------------------------------------------------------
# Config-style options carried through ``replace_backbone_with_flashinfer``.
# Mirrors the fields the wan example exposes via ``WanTransformer3DConfig``.
# ----------------------------------------------------------------------------

_GEMM_BACKENDS = tuple(b.value for b in GEMMBackend)


@dataclass
class FlashInferBackboneOptions:
    """Knobs for the FlashInfer swap, mirroring the wan example layout."""

    gemm_backend: Literal[
        "torch", "bf16", "fp8", "fp8_sm90", "bmm_fp8", "fp8_groupwise",
        "fp8_blockscaled", "batch_deepgemm_fp8", "fp4", "bmm_bf16", "mxfp8",
        "bmm_mxfp8",
    ] = "torch"
    online_act_quant: bool = True
    # Attention backend used when the caller supplies no attention mask
    # (typical for ``gen_text`` decode steps). When a 4D mask is provided we
    # always fall back to SDPA, because FlashInfer's prefill APIs do not take
    # arbitrary boolean masks. The four valid values match the wan dispatcher.
    attention_backend: Literal["auto", "single", "cudnn", "trtllm", "sdpa"] = "auto"
    # Force the MoE path. ``flashinfer`` calls
    # ``flashinfer.fused_moe.cutlass_fused_moe``; ``eager`` keeps the
    # per-expert loop in the upstream class.
    moe_impl: Literal["flashinfer", "eager"] = "flashinfer"


_FLASHINFER_ENV_OVERRIDES = {
    "gemm_backend": "FLASHINFER_GEMM_BACKEND",
    "online_act_quant": "FLASHINFER_ONLINE_ACT_QUANT",
    "attention_backend": "FLASHINFER_ATTENTION_BACKEND",
    "moe_impl": "FLASHINFER_MOE_IMPL",
}


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
    keyword arguments override env vars.
    """
    opts = FlashInferBackboneOptions()
    for field, env_name in _FLASHINFER_ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        if field == "online_act_quant":
            setattr(opts, field, _env_bool(env_name))
        else:
            setattr(opts, field, raw)
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


class FlashInferHunyuanRMSNorm(nn.Module):
    """RMSNorm that delegates to ``flashinfer.rmsnorm``.

    The upstream ``HunyuanRMSNorm`` casts to FP32 for the variance reduction
    and applies a learned ``weight`` of size ``hidden_size``. ``flashinfer.rmsnorm``
    expects ``(M, N)`` input with weight of shape ``(N,)`` in FP16/BF16.

    We therefore: flatten leading dims, cast FP32 inputs/weights to BF16 for
    the kernel, run the kernel, and cast the output back to the original
    dtype. Matches the FP32-accumulated semantics of the upstream
    implementation up to BF16 quantisation of the affine scale.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape
        # ``flashinfer.rmsnorm`` requires 2D input and BF16/FP16 weights.
        x = hidden_states.reshape(-1, orig_shape[-1])
        with _device_ctx(x):
            if x.dtype == torch.float32:
                x_bf16 = x.to(torch.bfloat16)
                w_bf16 = self.weight.to(torch.bfloat16)
                out = _flashinfer_rmsnorm(
                    x_bf16.contiguous(), w_bf16, self.variance_epsilon
                ).to(torch.float32)
            else:
                w = self.weight.to(x.dtype)
                out = _flashinfer_rmsnorm(x.contiguous(), w, self.variance_epsilon)
        return out.view(orig_shape).to(orig_dtype)


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
            print(f"[FI_DEBUG_MLP] silu in shape={tuple(fused_input.shape)} "
                  f"dtype={fused_input.dtype} dev={fused_input.device} "
                  f"contig={fused_input.is_contiguous()} "
                  f"cur_dev={torch.cuda.current_device()}", flush=True)
        with _device_ctx(fused_input):
            if fused_input.dtype == torch.float32:
                # ``silu_and_mul`` is BF16/FP16-only.
                hidden = _flashinfer_silu_and_mul(fused_input.to(torch.bfloat16)).to(
                    torch.float32
                )
            else:
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
            self.query_layernorm = FlashInferHunyuanRMSNorm(
                self.head_dim, eps=cfg.rms_norm_eps
            ).to(device=device, dtype=dtype)
            self.key_layernorm = FlashInferHunyuanRMSNorm(
                self.head_dim, eps=cfg.rms_norm_eps
            ).to(device=device, dtype=dtype)
            with torch.no_grad():
                self.query_layernorm.weight.copy_(original_attn.query_layernorm.weight)
                self.key_layernorm.weight.copy_(original_attn.key_layernorm.weight)

        # ``use_skip_softmax_sparse`` etc. are not used here: HunyuanImage-3
        # uses causal attention with custom masks, so the wan-style sparse
        # path doesn't apply.
        self._attention_backend = opts.attention_backend
        self._workspace: Optional[torch.Tensor] = None

    # ----- attention backend selection --------------------------------------

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        return self._workspace

    def _resolve_backend(self, q_len: int, has_mask: bool) -> str:
        """Pick the kernel for this call.

        Decisions:
        - If the caller passed a custom attention mask (``has_mask``), we
          can't use FlashInfer's prefill APIs (they only know about
          ``causal``); fall back to SDPA.
        - Decode steps (``q_len == 1``) use ``flashinfer.decode``.
        - Otherwise we honor the option: ``auto`` picks ``single`` (batch=1
          contexts dominate in this model since CFG runs both branches
          per-call but each pipeline call is bs=cfg_factor>=1 unbatched).
        - ``sdpa`` always falls back to PyTorch SDPA.
        """
        if has_mask:
            return "sdpa"
        if q_len == 1:
            return "decode"
        backend = self._attention_backend
        if backend == "auto":
            return "single"
        if backend == "sdpa":
            return "sdpa"
        return backend

    # ----- core attention dispatchers --------------------------------------

    def _attention_single(
        self,
        q: torch.Tensor,   # (B, num_heads, q_len, head_dim)
        k: torch.Tensor,   # (B, num_heads, kv_len, head_dim) — already repeat_kv'd
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        kv_len = k.shape[2]
        if bsz != 1:
            raise ValueError("'single' backend requires bsz==1.")
        # ``single_prefill_with_kv_cache`` expects NHD layout:
        # q: (q_len, num_heads, head_dim); kv: (kv_len, num_kv_heads, head_dim).
        q_nhd = q.transpose(1, 2).reshape(q_len, self.num_heads, self.head_dim).contiguous()
        k_nhd = k.transpose(1, 2).reshape(kv_len, self.num_heads, self.head_dim).contiguous()
        v_nhd = v.transpose(1, 2).reshape(kv_len, self.num_heads, self.head_dim).contiguous()
        out = single_prefill_with_kv_cache(q_nhd, k_nhd, v_nhd, causal=causal)
        # back to (B, num_heads, q_len, head_dim)
        return out.view(q_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

    def _attention_decode(
        self,
        q: torch.Tensor,   # (B, num_heads, 1, head_dim)
        k: torch.Tensor,   # (B, num_heads, kv_len, head_dim)
        v: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        assert q_len == 1
        if bsz != 1:
            # The HF generation loop is typically bsz=cfg_factor (1 or 2).
            # For bsz>1 fall back to SDPA which handles batched decode
            # without paged KV setup.
            return self._attention_sdpa(q, k, v, attn_mask=None)
        kv_len = k.shape[2]
        q_2d = q[0, :, 0, :].contiguous()           # (num_heads, head_dim)
        k_nhd = k[0].transpose(0, 1).contiguous()   # (kv_len, num_heads, head_dim)
        v_nhd = v[0].transpose(0, 1).contiguous()
        out = single_decode_with_kv_cache(q_2d, k_nhd, v_nhd)
        # (num_heads, head_dim) -> (1, num_heads, 1, head_dim)
        return out.view(1, self.num_heads, 1, self.head_dim)

    def _attention_cudnn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        bsz, _, q_len, _ = q.shape
        kv_len = k.shape[2]
        device = q.device
        q_flat = q.transpose(1, 2).reshape(
            bsz * q_len, self.num_heads, self.head_dim
        ).contiguous()
        k_flat = k.transpose(1, 2).reshape(
            bsz * kv_len, self.num_heads, self.head_dim
        ).contiguous()
        v_flat = v.transpose(1, 2).reshape(
            bsz * kv_len, self.num_heads, self.head_dim
        ).contiguous()
        actual_q = torch.full((bsz, 1, 1, 1), q_len, dtype=torch.int32, device=device)
        actual_kv = torch.full((bsz, 1, 1, 1), kv_len, dtype=torch.int32, device=device)
        batch_offsets_q = torch.arange(
            0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=device
        )
        batch_offsets_kv = torch.arange(
            0, (bsz + 1) * kv_len, kv_len, dtype=torch.int32, device=device
        )
        workspace = self._get_workspace(device)
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        out, _ = cudnn_batch_prefill_with_kv_cache(
            q=q_flat,
            k_cache=k_flat,
            v_cache=v_flat,
            scale=sm_scale,
            workspace_buffer=workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=kv_len,
            actual_seq_lens_q=actual_q,
            actual_seq_lens_kv=actual_kv,
            causal=causal,
            return_lse=False,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_q,
            batch_offsets_k=batch_offsets_kv,
            batch_offsets_v=batch_offsets_kv,
        )
        return out.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

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
            bsz, q_len, self.num_key_value_heads,
            self.num_key_value_groups + 2, self.head_dim,
        )
        # Same split convention as upstream: [num_kv_groups, 1, 1] along dim=3.
        q, k, v = torch.split(qkv, [self.num_key_value_groups, 1, 1], dim=3)
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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

        backend = self._resolve_backend(q_len, has_mask=attention_mask is not None)
        # For pure FlashInfer paths we use ``causal=True`` only when we know
        # there's no mask AND we're in a prefill step (q_len == kv_len, no
        # cached prefix). The model uses bidirectional attention inside
        # image blocks via a custom mask, so we only hit the FlashInfer
        # branches in mask-less contexts (decode steps and similar).
        causal = q_len == k.shape[2]

        with _device_ctx(q):
            if backend == "single":
                attn = self._attention_single(q, k, v, causal=causal)
            elif backend == "cudnn":
                attn = self._attention_cudnn(q, k, v, causal=causal)
            elif backend == "decode":
                attn = self._attention_decode(q, k, v)
            else:
                attn = self._attention_sdpa(q, k, v, attn_mask=attention_mask)

        attn = attn.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
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
# Swap pipeline
# ----------------------------------------------------------------------------


def _replace_rmsnorm(parent: nn.Module, name: str, original: nn.Module) -> None:
    """Replace a single ``HunyuanRMSNorm`` attribute with the FlashInfer
    equivalent, copying the affine weight in-place."""
    eps = getattr(original, "variance_epsilon", 1e-5)
    hidden_size = original.weight.shape[0]
    new = FlashInferHunyuanRMSNorm(hidden_size, eps=eps)
    with torch.no_grad():
        new.weight.copy_(original.weight)
    # Move to the same device/dtype as the original parameter, then attach.
    new = new.to(device=original.weight.device, dtype=original.weight.dtype)
    setattr(parent, name, new)


def _replace_mlp(parent: nn.Module, name: str, original: nn.Module,
                 opts: FlashInferBackboneOptions) -> None:
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


def _replace_attention(parent: nn.Module, name: str, original: nn.Module,
                       opts: FlashInferBackboneOptions) -> None:
    new = FlashInferHunyuanImage3Attention(original, opts)
    setattr(parent, name, new)


def _flip_moe_to_flashinfer(moe: nn.Module) -> None:
    """Switch a ``HunyuanMoE`` instance to the FlashInfer fused path.

    The upstream class already implements ``moe_impl='flashinfer'``: setting
    the attribute triggers a check that flashinfer is importable, and the
    first forward call stacks the expert weights and calls
    ``flashinfer.fused_moe.cutlass_fused_moe``. We just flip the flag.
    """
    moe.moe_impl = "flashinfer"


def _prepare_flashinfer_linear_weights(model: nn.Module) -> None:
    """Run ``FlashInferLinear.prepare_weights()`` on every swapped linear.

    Some GEMM backends (FP8 / FP4 / MXFP8 / blockscaled / ...) only quantize
    on the first call. Running ``prepare_weights`` up front means the model
    forward path doesn't pay quantisation cost on the first inference.
    """
    for mod in model.modules():
        if isinstance(mod, FlashInferLinear):
            mod.prepare_weights()


def replace_backbone_with_flashinfer(
    model: nn.Module,
    *,
    gemm_backend: Optional[str] = None,
    online_act_quant: Optional[bool] = None,
    attention_backend: Optional[str] = None,
    moe_impl: Optional[str] = None,
    prepare_weights: bool = True,
) -> FlashInferBackboneOptions:
    """Swap the hot kernels in a loaded HunyuanImage-3 model in-place.

    Targets every ``HunyuanImage3DecoderLayer`` in ``model.model.layers`` and
    swaps:

    1. ``input_layernorm`` / ``post_attention_layernorm`` (HunyuanRMSNorm)
       -> ``FlashInferHunyuanRMSNorm``.
    2. ``self_attn`` (HunyuanImage3SDPAAttention)
       -> ``FlashInferHunyuanImage3Attention``.
    3. ``mlp.shared_mlp`` / each ``mlp.experts[i]`` (HunyuanMLP)
       -> ``_FlashInferHunyuanMLP`` (only when ``hidden_act == 'silu'``).
    4. ``mlp`` (HunyuanMoE) -> upstream class with ``moe_impl='flashinfer'``.

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
            FP8/FP4-family backends consult this flag.
        attention_backend: ``auto`` (default), ``single``, ``cudnn``,
            ``trtllm``, or ``sdpa``. ``sdpa`` always uses PyTorch SDPA.
        moe_impl: ``flashinfer`` (default, calls
            ``flashinfer.fused_moe.cutlass_fused_moe``) or ``eager``.
        prepare_weights: If True, eagerly quantize every swapped
            ``FlashInferLinear`` so the first forward call doesn't pay the
            cost. Set False if you intend to call ``.to(...)`` afterwards.

    Returns:
        The resolved ``FlashInferBackboneOptions`` (for logging / debugging).
    """
    opts = _resolve_options(
        gemm_backend=gemm_backend,
        online_act_quant=online_act_quant,
        attention_backend=attention_backend,
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
    logger.info("Replacing %d HunyuanImage3 decoder layers with FlashInfer kernels.", n_layers)

    for layer in backbone.layers:
        # 1. Norms
        _replace_rmsnorm(layer, "input_layernorm", layer.input_layernorm)
        _replace_rmsnorm(layer, "post_attention_layernorm", layer.post_attention_layernorm)

        # 2. Attention
        _replace_attention(layer, "self_attn", layer.self_attn, opts)

        # 3. MLP / MoE
        mlp = layer.mlp
        if hasattr(mlp, "experts"):  # HunyuanMoE
            # Shared expert and each routed expert: swap to FlashInfer MLP.
            if getattr(mlp.config, "use_mixed_mlp_moe", False):
                _replace_mlp(mlp, "shared_mlp", mlp.shared_mlp, opts)
            for i in range(len(mlp.experts)):
                _replace_mlp(mlp.experts, str(i), mlp.experts[i], opts)
            # Switch to the fused cutlass MoE path.
            if opts.moe_impl == "flashinfer":
                _flip_moe_to_flashinfer(mlp)
        else:  # HunyuanMLP (dense layer)
            # Replace the whole MLP in-place.
            _replace_mlp(layer, "mlp", mlp, opts)

    # 4. Final RMSNorm on the LM-head side.
    if hasattr(backbone, "ln_f"):
        _replace_rmsnorm(backbone, "ln_f", backbone.ln_f)

    if prepare_weights:
        _prepare_flashinfer_linear_weights(model)

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
        "--model", default="tencent/HunyuanImage-3.0-Instruct",
        help="HuggingFace repo id or local path.",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--gemm-backend", default=os.getenv("FLASHINFER_GEMM_BACKEND", "torch"),
        choices=_GEMM_BACKENDS,
    )
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        choices=["auto", "single", "cudnn", "trtllm", "sdpa"],
    )
    parser.add_argument(
        "--moe-impl",
        default=os.getenv("FLASHINFER_MOE_IMPL", "flashinfer"),
        choices=["flashinfer", "eager"],
    )
    parser.add_argument("--offline-act-quant", action="store_true")
    parser.add_argument(
        "--skip-prepare-weights", action="store_true",
        help="Skip the eager FlashInferLinear.prepare_weights() pass.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM
    print(f"Loading {args.model} (trust_remote_code=True) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
    )
    model = model.eval().cuda()

    opts = replace_backbone_with_flashinfer(
        model,
        gemm_backend=args.gemm_backend,
        attention_backend=args.attention_backend,
        moe_impl=args.moe_impl,
        online_act_quant=not args.offline_act_quant,
        prepare_weights=not args.skip_prepare_weights,
    )
    print(f"FlashInfer options applied: {opts}")

    n_flashinfer_linear = sum(
        1 for m in model.modules() if isinstance(m, FlashInferLinear)
    )
    n_flashinfer_rmsnorm = sum(
        1 for m in model.modules() if isinstance(m, FlashInferHunyuanRMSNorm)
    )
    n_flashinfer_attn = sum(
        1 for m in model.modules() if isinstance(m, FlashInferHunyuanImage3Attention)
    )
    print(f"FlashInferLinear modules: {n_flashinfer_linear}")
    print(f"FlashInferHunyuanRMSNorm modules: {n_flashinfer_rmsnorm}")
    print(f"FlashInferHunyuanImage3Attention modules: {n_flashinfer_attn}")
