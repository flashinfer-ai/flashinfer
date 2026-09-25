from flashinfer.norm import (
    fused_dit_gate_residual_layernorm_gamma_beta,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_residual_layernorm_scale_shift,
    fused_qk_rmsnorm_rope,
)
from .cake_minimax_h3_dense_attention import minimax_h3_dense_attention
from .minimax_h3 import minimax_h3_bf16_pre_attention
from .minimax_h3_fc1_swiglu import (
    minimax_h3_fc1_swiglu,
    minimax_h3_fc1_swiglu_mxfp8,
    minimax_h3_fc1_swiglu_nvfp4,
    prepare_minimax_h3_fc1_weight_mxfp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
)
from .cake_minimax_h3_sm120_quant_pre_attention import (
    MiniMaxH3PreAttentionOutput,
    minimax_h3_fp8_pre_attention,
    minimax_h3_nvfp4_pre_attention,
    quantize_minimax_h3_qkv_weight_fp8,
    quantize_minimax_h3_qkv_weight_nvfp4,
)

from .cake_minimax_h3_sm120_quant_out_proj import (
    minimax_h3_fp8_out_proj,
    minimax_h3_nvfp4_out_proj,
    quantize_minimax_h3_o_weight_fp8,
    quantize_minimax_h3_o_weight_nvfp4,
)
from .cake_minimax_h3_mxfp8 import (
    PreparedMiniMaxH3Mxfp8PreAttention,
    prepare_minimax_h3_mxfp8_pre_attention,
)
from .cake_minimax_h3_nvfp4 import (
    PreparedMiniMaxH3Nvfp4PreAttention,
    prepare_minimax_h3_nvfp4_pre_attention,
)
from .cake_minimax_h3_qkv_pack import (
    PreparedMiniMaxH3QkvQuantizePack,
    minimax_h3_qkv_quantize_pack,
    prepare_minimax_h3_qkv_quantize_pack,
)

__all__ = [
    "fused_dit_gate_residual_layernorm_gamma_beta",
    "fused_dit_gate_residual_layernorm_scale_shift",
    "fused_dit_residual_layernorm_scale_shift",
    "fused_qk_rmsnorm_rope",
    "minimax_h3_bf16_pre_attention",
    "minimax_h3_dense_attention",
    "minimax_h3_fc1_swiglu",
    "minimax_h3_fc1_swiglu_mxfp8",
    "minimax_h3_fc1_swiglu_nvfp4",
    "prepare_minimax_h3_fc1_weight_mxfp8",
    "prepare_minimax_h3_fc1_weight_nvfp4",
    "MiniMaxH3PreAttentionOutput",
    "minimax_h3_fp8_pre_attention",
    "minimax_h3_nvfp4_pre_attention",
    "quantize_minimax_h3_qkv_weight_fp8",
    "quantize_minimax_h3_qkv_weight_nvfp4",
    "PreparedMiniMaxH3QkvQuantizePack",
    "minimax_h3_qkv_quantize_pack",
    "prepare_minimax_h3_qkv_quantize_pack",
    "minimax_h3_fp8_out_proj",
    "minimax_h3_nvfp4_out_proj",
    "quantize_minimax_h3_o_weight_fp8",
    "quantize_minimax_h3_o_weight_nvfp4",
]
