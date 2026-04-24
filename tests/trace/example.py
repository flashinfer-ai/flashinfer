"""
fi_trace example: generate flashinfer-bench definition JSON files via auto-dump.

Run:
    python tests/trace/example.py

When FLASHINFER_TRACE_DUMP=1 (set below), every @flashinfer_api(trace=...) decorated
function automatically writes a trace JSON on its first call for each unique input
shape.  Subsequent calls with the same shape are deduplicated (no re-write).

The output directory is controlled by FLASHINFER_TRACE_DUMP_DIR.

Requires a CUDA-capable GPU.

Results:
- We would get these example json files under fi_trace_out directory:
fused_add_rmsnorm_h5120.json
fused_add_rmsnorm_quant_h7168.json
gdn_decode_qk4_v8_d128.json
gdn_mtp_qk4_v8_d128.json
gdn_prefill_qk4_v8_d128.json
gemm_bf16_N256_K7168.json
gemm_bf16_N4096_K4096.json
gemm_fp4_N2048_K7168_block_size16.json
gemm_fp8_N1536_K7168.json
gemm_mxfp8_N4096_K4096.json
gemma_fused_add_rmsnorm_h4608.json
gemma_rmsnorm_h4608.json
gelu_and_mul_h16384.json
gelu_tanh_and_mul_h16384.json
gqa_paged_decode_h32_kv8_d128_ps16.json
gqa_paged_decode_h32_kv8_d128_ps64.json
gqa_paged_prefill_h32_kv8_d128_ps16.json
gqa_ragged_h32_kv8_d128.json
layernorm_h768.json
merge_state_h32_d128.json
merge_state_in_place_h32_d128.json
merge_states_h32_d128.json
mla_paged_decode_h16_ckv512_kpe64_ps1.json
mla_paged_decode_h16_ckv512_kpe64_ps64.json
moe_fp4_block_scale_default_routing_topk8_e32_h7168_i2048.json
moe_fp4_block_scale_ds_routing_topk8_e32_h7168_i2048_ng8_kg4.json
moe_fp4_block_scale_llama4_routing_topk1_e32_h7168_i2048.json
moe_fp4_block_scale_renormalize_naive_routing_topk8_e32_h7168_i2048.json
moe_fp4_block_scale_renormalize_routing_topk8_e32_h7168_i2048.json
moe_fp4_block_scale_topk_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_default_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
moe_fp8_block_scale_llama4_routing_topk1_e32_h7168_i2048.json
moe_fp8_block_scale_renormalize_naive_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_renormalize_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_topk_routing_topk8_e32_h7168_i2048.json
rmsnorm_h4096.json
rmsnorm_h7168.json
rmsnorm_quant_h7168.json
silu_and_mul_h16384.json
top_k_sampling_v128256.json
top_k_top_p_sampling_v128256.json
top_k_top_p_sampling_v151936.json
top_p_sampling_v128256.json
top_p_sampling_v151936.json

Note: top_p_sampling files appear for vocab_size=151936 because
top_k_top_p_sampling calls top_p_sampling internally.
FP4 MoE files are only generated on Blackwell (SM100+) GPUs with fp4_quantize available.
GDN prefill files require SM90+ (Hopper) GPU.
"""

import contextlib
import json
import os
from pathlib import Path

# Must be set before any flashinfer import: template.py reads these at module load time.
os.environ.setdefault(
    "FLASHINFER_TRACE_DUMP_DIR",
    str(Path(__file__).parent / "fi_trace_out"),
)
os.environ.setdefault("FLASHINFER_TRACE_DUMP", "1")

SAVE_DIR = Path(os.environ["FLASHINFER_TRACE_DUMP_DIR"])

import torch

import flashinfer
import flashinfer.norm
import flashinfer.sampling
import flashinfer.gemm
import flashinfer.gdn_decode
import flashinfer.fused_moe
import flashinfer.activation
import flashinfer.cascade
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.mla import BatchMLAPagedAttentionWrapper

device = "cuda"
WORKSPACE = 128 * 1024 * 1024  # 128 MB

print(f"\nAuto-dumping fi_trace JSON files to {SAVE_DIR}/\n")

# ── rmsnorm ───────────────────────────────────────────────────────────────────
# Llama-3.1-8B (hidden=4096) and DeepSeek-V3 (hidden=7168)
for hidden_size in (4096, 7168):
    hidden = torch.randn(32, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    flashinfer.rmsnorm(hidden, weight)

# ── fused_add_rmsnorm (Qwen3-14B, hidden=5120) ───────────────────────────────
x = torch.randn(32, 5120, dtype=torch.bfloat16, device=device)
res = torch.randn(32, 5120, dtype=torch.bfloat16, device=device)
w = torch.ones(5120, dtype=torch.bfloat16, device=device)
flashinfer.fused_add_rmsnorm(x, res, w)

# ── rmsnorm_quant + fused_add_rmsnorm_quant (DeepSeek-V3 down-proj, h=7168) ──
# Quantize to FP8 E4M3 after normalization; scale is per-tensor.
norm_h = 7168
norm_in = torch.randn(32, norm_h, dtype=torch.bfloat16, device=device)
norm_w = torch.ones(norm_h, dtype=torch.bfloat16, device=device)
norm_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
norm_out = torch.empty(32, norm_h, dtype=torch.float8_e4m3fn, device=device)
flashinfer.rmsnorm_quant(norm_out, norm_in, norm_w, norm_scale)

norm_res = torch.randn(32, norm_h, dtype=torch.bfloat16, device=device)
flashinfer.fused_add_rmsnorm_quant(norm_out, norm_in, norm_res, norm_w, norm_scale)

# ── gemma_rmsnorm + gemma_fused_add_rmsnorm (Gemma-2-27B, hidden=4608) ───────
gemma_h = 4608
gemma_in = torch.randn(32, gemma_h, dtype=torch.bfloat16, device=device)
gemma_w = torch.zeros(gemma_h, dtype=torch.bfloat16, device=device)
flashinfer.gemma_rmsnorm(gemma_in, gemma_w)

gemma_res = torch.randn(32, gemma_h, dtype=torch.bfloat16, device=device)
flashinfer.gemma_fused_add_rmsnorm(gemma_in, gemma_res, gemma_w)

# ── layernorm (GPT-2/BERT, hidden=768) ────────────────────────────────────────
ln_h = 768
ln_in = torch.randn(32, ln_h, dtype=torch.bfloat16, device=device)
ln_gamma = torch.ones(ln_h, dtype=torch.float32, device=device)
ln_beta = torch.zeros(ln_h, dtype=torch.float32, device=device)
flashinfer.layernorm(ln_in, ln_gamma, ln_beta)

# ── sampling (Llama vocab=128256) ─────────────────────────────────────────────
probs = torch.rand(64, 128256, dtype=torch.float32, device=device)
top_k = torch.full((64,), 50, dtype=torch.int32, device=device)
top_p = torch.full((64,), 0.9, dtype=torch.float32, device=device)
flashinfer.top_k_sampling_from_probs(probs, top_k)
flashinfer.top_p_sampling_from_probs(probs, top_p)
flashinfer.top_k_top_p_sampling_from_probs(probs, top_k, top_p)

# ── sampling (Qwen3 vocab=151936) ─────────────────────────────────────────────
probs = torch.rand(64, 151936, dtype=torch.float32, device=device)
flashinfer.top_k_top_p_sampling_from_probs(probs, top_k, top_p)

# ── Activation functions (LLaMA/Mistral FFN, hidden=8192 gate+up) ─────────────
# Input shape is [T, 2*H] where H is the output (post-gate) hidden dim.
act_input = torch.randn(128, 2 * 8192, dtype=torch.bfloat16, device=device)
flashinfer.silu_and_mul(act_input)
flashinfer.gelu_tanh_and_mul(act_input)
flashinfer.gelu_and_mul(act_input)

# ── Cascade / merge attention states ─────────────────────────────────────────
# Cascade attention merges partial V/S states from different KV segments.
ms_T, ms_H, ms_D = 128, 32, 128
v_a = torch.randn(ms_T, ms_H, ms_D, dtype=torch.bfloat16, device=device)
s_a = torch.randn(ms_T, ms_H, dtype=torch.float32, device=device)
v_b = torch.randn(ms_T, ms_H, ms_D, dtype=torch.bfloat16, device=device)
s_b = torch.randn(ms_T, ms_H, dtype=torch.float32, device=device)
flashinfer.merge_state(v_a, s_a, v_b, s_b)
flashinfer.merge_state_in_place(v_a, s_a, v_b, s_b)
# merge_states: [T, num_states, H, D]
v_multi = torch.randn(ms_T, 4, ms_H, ms_D, dtype=torch.bfloat16, device=device)
s_multi = torch.randn(ms_T, 4, ms_H, dtype=torch.float32, device=device)
flashinfer.merge_states(v_multi, s_multi)

# ── RoPE (Llama-3.1-8B: h=32/kv=8/d=128, batch=4, seq=128) ────────────────────
rope_B, rope_S, rope_Hq, rope_Hk, rope_D = 4, 128, 32, 8, 128
rope_nnz = rope_B * rope_S
rope_q = torch.randn(rope_nnz, rope_Hq, rope_D, dtype=torch.bfloat16, device=device)
rope_k = torch.randn(rope_nnz, rope_Hk, rope_D, dtype=torch.bfloat16, device=device)
rope_indptr = torch.arange(rope_B + 1, dtype=torch.int32, device=device) * rope_S
rope_offsets = torch.zeros(rope_B, dtype=torch.int32, device=device)
rope_pos_ids = torch.arange(rope_nnz, dtype=torch.int32, device=device) % rope_S
flashinfer.apply_rope(rope_q, rope_k, rope_indptr, rope_offsets)
flashinfer.apply_rope_inplace(rope_q.clone(), rope_k.clone(), rope_indptr, rope_offsets)
flashinfer.apply_rope_pos_ids(rope_q, rope_k, rope_pos_ids)
flashinfer.apply_rope_pos_ids_inplace(rope_q.clone(), rope_k.clone(), rope_pos_ids)
flashinfer.apply_llama31_rope(rope_q, rope_k, rope_indptr, rope_offsets)
flashinfer.apply_llama31_rope_inplace(
    rope_q.clone(), rope_k.clone(), rope_indptr, rope_offsets
)
flashinfer.apply_llama31_rope_pos_ids(rope_q, rope_k, rope_pos_ids)
flashinfer.apply_llama31_rope_pos_ids_inplace(
    rope_q.clone(), rope_k.clone(), rope_pos_ids
)

# ── RoPE with cos/sin cache (SGL/vLLM-compatible) ─────────────────────────────
rope_query = torch.randn(
    rope_nnz, rope_Hq * rope_D, dtype=torch.bfloat16, device=device
)
rope_key = torch.randn(rope_nnz, rope_Hk * rope_D, dtype=torch.bfloat16, device=device)
rope_cos_sin = torch.randn(8192, rope_D, dtype=torch.float32, device=device)
rope_positions = torch.arange(rope_nnz, dtype=torch.int32, device=device) % 8192
flashinfer.apply_rope_with_cos_sin_cache(
    rope_positions, rope_query, rope_key, rope_D, rope_cos_sin
)
flashinfer.apply_rope_with_cos_sin_cache_inplace(
    rope_positions, rope_query.clone(), rope_key.clone(), rope_D, rope_cos_sin
)

# ── Quantization (FP4 / NVFP4 / MXFP4 / MXFP8, SM100+) ────────────────────────
# Kernels are SM100+ only; trace is dumped before kernel launch so JSONs are
# generated on any GPU — runtime failures are suppressed.
from flashinfer.quantization.fp4_quantization import (
    fp4_quantize,
    mxfp4_quantize,
    nvfp4_quantize,
)
from flashinfer.quantization.fp8_quantization import mxfp8_quantize

quant_M, quant_K = 128, 4096
quant_input_bf16 = torch.randn(quant_M, quant_K, dtype=torch.bfloat16, device=device)
quant_global_sf = torch.tensor([1.0], dtype=torch.float32, device=device)

with contextlib.suppress(Exception):
    fp4_quantize(quant_input_bf16, quant_global_sf, sf_vec_size=16)
with contextlib.suppress(Exception):
    nvfp4_quantize(quant_input_bf16, quant_global_sf)
with contextlib.suppress(Exception):
    mxfp4_quantize(quant_input_bf16)
with contextlib.suppress(Exception):
    mxfp8_quantize(quant_input_bf16)

# ── Single-request attention (non-batched) ───────────────────────────────────
sa_Hq, sa_Hk, sa_D, sa_KV = 32, 8, 128, 256
sa_q_dec = torch.randn(sa_Hq, sa_D, dtype=torch.bfloat16, device=device)
sa_k_dec = torch.randn(sa_KV, sa_Hk, sa_D, dtype=torch.bfloat16, device=device)
sa_v_dec = torch.randn(sa_KV, sa_Hk, sa_D, dtype=torch.bfloat16, device=device)
with contextlib.suppress(Exception):
    flashinfer.single_decode_with_kv_cache(sa_q_dec, sa_k_dec, sa_v_dec)

sa_Q = 128
sa_q_pf = torch.randn(sa_Q, sa_Hq, sa_D, dtype=torch.bfloat16, device=device)
sa_k_pf = torch.randn(sa_KV, sa_Hk, sa_D, dtype=torch.bfloat16, device=device)
sa_v_pf = torch.randn(sa_KV, sa_Hk, sa_D, dtype=torch.bfloat16, device=device)
with contextlib.suppress(Exception):
    flashinfer.single_prefill_with_kv_cache(sa_q_pf, sa_k_pf, sa_v_pf, causal=True)

# ── GEMM bf16 ─────────────────────────────────────────────────────────────────
# Llama-3.1-8B o_proj (4096×4096) and DeepSeek-V3 moe.gate (256×7168)
# mm_bf16 expects b in column-major layout with shape [K, N].
# randn(N, K).T gives shape [K, N] with strides (1, N); the kernel transposes
# b back to [N, K] (contiguous) before calling the C++ matmul.
# backend="auto" picks cudnn on SM80/89/90 and cutlass on SM100+.
for N, K in ((4096, 4096), (256, 7168)):
    a = torch.randn(128, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(
        N, K, dtype=torch.bfloat16, device=device
    ).T  # [K, N] column-major; b.T is contiguous
    with contextlib.suppress(Exception):
        flashinfer.mm_bf16(a, b, backend="auto")

# ── GEMM fp8 block-scale (DeepSeek-V3 q_proj: M×7168→1536, block=128) ────────
# Trace is dumped before kernel launch; suppress SM100-only runtime failures.
with contextlib.suppress(Exception):
    M, K, N, BS = 128, 7168, 1536, 128
    a_fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device=device)
    b_fp8 = torch.zeros(K // BS, N, BS, dtype=torch.float8_e4m3fn, device=device)
    alpha_fp8 = torch.tensor(1.0, dtype=torch.float32, device=device)
    flashinfer.mm_fp8(a_fp8, b_fp8, alpha_fp8)

# ── GEMM mxfp8 (Blackwell SM100+: M×4096@4096×4096, block=32) ────────────────
try:
    M, K, N = 128, 4096, 4096
    a_mxfp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device=device)
    b_mxfp8 = torch.zeros(K, N, dtype=torch.float8_e4m3fn, device=device)
    a_ds = torch.ones(M, K // 32, dtype=torch.uint8, device=device)
    b_ds = torch.ones(K // 32, N, dtype=torch.uint8, device=device)
    flashinfer.gemm.mm_mxfp8(a_mxfp8, b_mxfp8, a_ds, b_ds)
except Exception:
    pass  # Requires Blackwell (SM100+)

# ── GEMM fp4 (Blackwell SM100+: M×7168@2048×7168, block=16) ─────────────────
try:
    M, K, N, BS4 = 128, 7168, 2048, 16
    a_fp4 = torch.zeros(M, K, dtype=torch.uint8, device=device)
    b_fp4 = torch.zeros(K, N, dtype=torch.uint8, device=device)
    a_d4 = torch.ones(M, K // BS4, dtype=torch.float8_e4m3fn, device=device)
    b_d4 = torch.ones(K, N // BS4, dtype=torch.float8_e4m3fn, device=device)
    flashinfer.gemm.mm_fp4(a_fp4, b_fp4, a_d4, b_d4, block_size=BS4)
except Exception:
    pass  # Requires Blackwell (SM100+)

# ── GQA paged decode (Llama-3.1-8B, h=32/kv=8/d=128) ────────────────────────
num_qo, num_kv, head_dim, batch_size = 32, 8, 128, 32

for page_size, num_pages in ((16, 128), (64, 32)):
    total = batch_size * num_pages
    kv_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    kv_indices = torch.arange(total, dtype=torch.int32, device=device)
    kv_last = torch.full((batch_size,), page_size, dtype=torch.int32, device=device)

    ws = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
    dec = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    dec.plan(
        kv_indptr,
        kv_indices,
        kv_last,
        num_qo,
        num_kv,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    q_d = torch.randn(batch_size, num_qo, head_dim, dtype=torch.bfloat16, device=device)
    kc = torch.randn(
        total, page_size, num_kv, head_dim, dtype=torch.bfloat16, device=device
    )
    vc = torch.randn(
        total, page_size, num_kv, head_dim, dtype=torch.bfloat16, device=device
    )
    dec.run(q_d, (kc, vc))

# ── GQA paged prefill (Llama-3.1-8B, h=32/kv=8/d=128, page_size=16) ─────────
n_req, total_q, np_pf, page_size = 4, 512, 32, 16
total_pf = n_req * np_pf
qo_indptr = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32, device=device)
kv_indptr_p = torch.arange(n_req + 1, dtype=torch.int32, device=device) * np_pf
kv_idx_p = torch.arange(total_pf, dtype=torch.int32, device=device)
kv_last_p = torch.full((n_req,), page_size, dtype=torch.int32, device=device)

ws_pf = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
pf = BatchPrefillWithPagedKVCacheWrapper(ws_pf, "NHD")
pf.plan(
    qo_indptr,
    kv_indptr_p,
    kv_idx_p,
    kv_last_p,
    num_qo,
    num_kv,
    head_dim,
    page_size,
    causal=True,
    q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16,
)
q_pf = torch.randn(total_q, num_qo, head_dim, dtype=torch.bfloat16, device=device)
kc_pf = torch.randn(
    total_pf, page_size, num_kv, head_dim, dtype=torch.bfloat16, device=device
)
vc_pf = torch.randn(
    total_pf, page_size, num_kv, head_dim, dtype=torch.bfloat16, device=device
)
pf.run(q_pf, (kc_pf, vc_pf))

# ── GQA ragged prefill (Llama-3.1-8B) ────────────────────────────────────────
qo_indptr_r = torch.tensor([0, 64, 128, 192, 256], dtype=torch.int32, device=device)
kv_indptr_r = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32, device=device)

ws_r = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
rag = BatchPrefillWithRaggedKVCacheWrapper(ws_r, "NHD")
rag.plan(
    qo_indptr_r,
    kv_indptr_r,
    num_qo,
    num_kv,
    head_dim,
    causal=True,
    q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16,
)
q_r = torch.randn(256, num_qo, head_dim, dtype=torch.bfloat16, device=device)
k_r = torch.randn(512, num_kv, head_dim, dtype=torch.bfloat16, device=device)
v_r = torch.randn(512, num_kv, head_dim, dtype=torch.bfloat16, device=device)
rag.run(q_r, k_r, v_r)

# ── MLA paged decode (DeepSeek-V3 TP=8, h=16/ckv=512/kpe=64) ─────────────────
mla_b, mla_h, ckv, kpe = 128, 16, 512, 64

for mla_ps, mla_np in ((64, 32), (1, 2048)):
    total_mla = mla_b * mla_np
    mla_qo_indptr = torch.arange(mla_b + 1, dtype=torch.int32, device=device)
    mla_kv_indptr = torch.arange(mla_b + 1, dtype=torch.int32, device=device) * mla_np
    mla_kv_indices = torch.arange(total_mla, dtype=torch.int32, device=device)
    mla_kv_len = torch.full((mla_b,), mla_np * mla_ps, dtype=torch.int32, device=device)

    ws_mla = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
    mla = BatchMLAPagedAttentionWrapper(ws_mla)
    mla.plan(
        mla_qo_indptr,
        mla_kv_indptr,
        mla_kv_indices,
        mla_kv_len,
        mla_h,
        ckv,
        kpe,
        mla_ps,
        causal=False,
        sm_scale=1.0 / (ckv**0.5),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    q_nope = torch.randn(mla_b, mla_h, ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(mla_b, mla_h, kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(total_mla, mla_ps, ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(total_mla, mla_ps, kpe, dtype=torch.bfloat16, device=device)
    mla.run(q_nope, q_pe, ckv_cache, kpe_cache)

# ── GDN prefill (Qwen3-Next TP=4, chunk prefill) ─────────────────────────────
with contextlib.suppress(Exception):
    import flashinfer.gdn_prefill  # noqa: PLC0415

    gp_T, gp_H, gp_HV, gp_K = 256, 4, 8, 128
    cu_seqlens = torch.tensor([0, 64, 128, 192, 256], dtype=torch.int64, device=device)
    gp_q = torch.randn(gp_T, gp_H, gp_K, dtype=torch.bfloat16, device=device)
    gp_k = torch.randn(gp_T, gp_H, gp_K, dtype=torch.bfloat16, device=device)
    gp_v = torch.randn(gp_T, gp_HV, gp_K, dtype=torch.bfloat16, device=device)
    flashinfer.gdn_prefill.chunk_gated_delta_rule(
        gp_q, gp_k, gp_v, cu_seqlens=cu_seqlens
    )

# ── GDN decode (Qwen3-Next TP=4, qk=4/v=8/d=128) ────────────────────────────
B, H, HV, K = 4, 4, 8, 128
q = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device)
k = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device=device)
v = torch.randn(B, 1, HV, K, dtype=torch.bfloat16, device=device)
state = torch.zeros(B, HV, K, K, dtype=torch.float32, device=device)
A_log = torch.zeros(HV, dtype=torch.float32, device=device)
a = torch.zeros(B, 1, HV, dtype=torch.bfloat16, device=device)
dt_bias = torch.zeros(HV, dtype=torch.float32, device=device)
b_ = torch.zeros(B, 1, HV, dtype=torch.bfloat16, device=device)
flashinfer.gdn_decode.gated_delta_rule_decode(q, k, v, state, A_log, a, dt_bias, b_)

# ── GDN MTP (Qwen3-Next TP=4, spec_len=4) ────────────────────────────────────
T_mtp, pool_size = 4, 8
q_m = torch.randn(B, T_mtp, H, K, dtype=torch.bfloat16, device=device)
k_m = torch.randn(B, T_mtp, H, K, dtype=torch.bfloat16, device=device)
v_m = torch.randn(B, T_mtp, HV, K, dtype=torch.bfloat16, device=device)
init_state = torch.zeros(pool_size, HV, K, K, dtype=torch.float32, device=device)
init_idx = torch.arange(B, dtype=torch.int32, device=device)
A_log_m = torch.zeros(HV, dtype=torch.float32, device=device)
a_m = torch.zeros(B, T_mtp, HV, dtype=torch.bfloat16, device=device)
dt_bias_m = torch.zeros(HV, dtype=torch.float32, device=device)
b_m = torch.zeros(B, T_mtp, HV, dtype=torch.bfloat16, device=device)
flashinfer.gdn_decode.gated_delta_rule_mtp(
    q_m, k_m, v_m, init_state, init_idx, A_log_m, a_m, dt_bias_m, b_m
)

# ── MoE FP8 (256 experts, 32 local, h=7168, i=2048) ─────────────────────────
# routing_method_type: 0=Default, 1=Renormalize, 2=DeepSeekV3,
#                      3=Llama4,   4=RenormalizeNaive, 5=TopK
T_moe, H_moe, I_moe, E_tot, E_loc, BS = 128, 7168, 2048, 256, 32, 128
routing_logits = torch.randn(T_moe, E_tot, dtype=torch.float32, device=device)
routing_bias = torch.zeros(E_tot, dtype=torch.bfloat16, device=device)
hs = torch.zeros(T_moe, H_moe, dtype=torch.float8_e4m3fn, device=device)
hs_scale = torch.ones(H_moe // BS, T_moe, dtype=torch.float32, device=device)
w1 = torch.zeros(E_loc, 2 * I_moe, H_moe, dtype=torch.float8_e4m3fn, device=device)
w1s = torch.ones(
    E_loc, (2 * I_moe) // BS, H_moe // BS, dtype=torch.float32, device=device
)
w2 = torch.zeros(E_loc, H_moe, I_moe, dtype=torch.float8_e4m3fn, device=device)
w2s = torch.ones(E_loc, H_moe // BS, I_moe // BS, dtype=torch.float32, device=device)
_moe_common = dict(
    num_experts=E_tot,
    intermediate_size=I_moe,
    local_expert_offset=0,
    local_num_experts=E_loc,
    routed_scaling_factor=2.5,
)
_moe_args = (routing_logits, routing_bias, hs, hs_scale, w1, w1s, w2, w2s)

# Each routing type in its own try/except so a GPU-support failure on one
# variant does not prevent the remaining traces from being dumped.

# 0: Default routing (Softmax -> TopK)
with contextlib.suppress(Exception):
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=8,
        n_group=None,
        topk_group=None,
        routing_method_type=0,
        **_moe_common,
    )

# 1: Renormalize routing (TopK -> Softmax)
with contextlib.suppress(Exception):
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=8,
        n_group=None,
        topk_group=None,
        routing_method_type=1,
        **_moe_common,
    )

# 2: DeepSeekV3 routing (Sigmoid -> group selection -> top_k=8)
with contextlib.suppress(Exception):
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=8,
        n_group=8,
        topk_group=4,
        routing_method_type=2,
        **_moe_common,
    )

# 3: Llama4 routing (Top1 -> Sigmoid)
with contextlib.suppress(Exception):
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=1,
        n_group=None,
        topk_group=None,
        routing_method_type=3,
        **_moe_common,
    )

# 4: RenormalizeNaive routing (Softmax -> TopK -> Renormalize)
with contextlib.suppress(Exception):
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=8,
        n_group=None,
        topk_group=None,
        routing_method_type=4,
        **_moe_common,
    )

# 5: TopK routing (plain TopK, no normalisation)
with contextlib.suppress(Exception):
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=8,
        n_group=None,
        topk_group=None,
        routing_method_type=5,
        **_moe_common,
    )

# ── MoE FP4 (NvFP4, 256 experts, 32 local, h=7168, i=2048) ──────────────────
# routing_method_type: 0=Default, 1=Renormalize, 2=DeepSeekV3,
#                      3=Llama4,   4=RenormalizeNaive, 5=TopK
# NvFP4: block_size=16; hidden_states packed as [T, H//2] uint8,
#        scale as [T, H//16] float8.
try:
    import flashinfer
    from flashinfer import fp4_quantize

    T_fp4, H_fp4, I_fp4, E_tot_fp4, E_loc_fp4 = 128, 7168, 2048, 256, 32
    SF_VEC = 16

    routing_logits_fp4 = torch.randn(
        T_fp4, E_tot_fp4, dtype=torch.bfloat16, device=device
    )
    hs_bf16 = torch.randn(T_fp4, H_fp4, dtype=torch.bfloat16, device=device) * 0.1
    hs_fp4, hs_fp4_scale = fp4_quantize(
        hs_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    hs_fp4_scale = hs_fp4_scale.view(torch.float8_e4m3fn).reshape(T_fp4, -1)

    w13_bf16 = (
        torch.randn(E_loc_fp4, 2 * I_fp4, H_fp4, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    w13_fp4, w13_fp4_scale = fp4_quantize(
        w13_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC,
        sf_use_ue8m0=False,
    )
    w13_fp4_scale = w13_fp4_scale.view(torch.float8_e4m3fn).reshape(
        E_loc_fp4, 2 * I_fp4, -1
    )
    w2_bf16 = (
        torch.randn(E_loc_fp4, H_fp4, I_fp4, dtype=torch.bfloat16, device=device) * 0.1
    )
    w2_fp4, w2_fp4_scale = fp4_quantize(
        w2_bf16,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=SF_VEC,
        sf_use_ue8m0=False,
    )
    w2_fp4_scale = w2_fp4_scale.view(torch.float8_e4m3fn).reshape(E_loc_fp4, H_fp4, -1)

    scale_val = 1.0 / 448.0 / 6.0
    out1_scale = torch.full((E_loc_fp4,), scale_val**2, device=device)
    out1_gate_scale = torch.full((E_loc_fp4,), scale_val**2, device=device)
    out2_scale = torch.full((E_loc_fp4,), scale_val**2, device=device)

    _fp4_moe_common = dict(
        num_experts=E_tot_fp4,
        intermediate_size=I_fp4,
        local_expert_offset=0,
        local_num_experts=E_loc_fp4,
        routed_scaling_factor=None,
    )
    _fp4_moe_args = (
        routing_logits_fp4,
        None,  # routing_bias
        hs_fp4,
        hs_fp4_scale,
        w13_fp4,
        w13_fp4_scale,
        None,  # gemm1_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2_fp4,
        w2_fp4_scale,
        None,  # gemm2_bias
        out1_scale,
        out1_gate_scale,
        out2_scale,
    )
except Exception:
    _fp4_moe_args = None  # fp4_quantize unavailable

if _fp4_moe_args is not None:
    # 0: Default routing (Softmax -> TopK)
    with contextlib.suppress(Exception):
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            *_fp4_moe_args,
            top_k=8,
            n_group=None,
            topk_group=None,
            routing_method_type=0,
            **_fp4_moe_common,
        )

    # 1: Renormalize routing (TopK -> Softmax)
    with contextlib.suppress(Exception):
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            *_fp4_moe_args,
            top_k=8,
            n_group=None,
            topk_group=None,
            routing_method_type=1,
            **_fp4_moe_common,
        )

    # 2: DeepSeekV3 routing (Sigmoid -> group selection -> top_k=8)
    with contextlib.suppress(Exception):
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            *_fp4_moe_args,
            top_k=8,
            n_group=8,
            topk_group=4,
            routing_method_type=2,
            **_fp4_moe_common,
        )

    # 3: Llama4 routing (Top1 -> Sigmoid)
    with contextlib.suppress(Exception):
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            *_fp4_moe_args,
            top_k=1,
            n_group=None,
            topk_group=None,
            routing_method_type=3,
            **_fp4_moe_common,
        )

    # 4: RenormalizeNaive routing (Softmax -> TopK -> Renormalize)
    with contextlib.suppress(Exception):
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            *_fp4_moe_args,
            top_k=8,
            n_group=None,
            topk_group=None,
            routing_method_type=4,
            **_fp4_moe_common,
        )

    # 5: TopK routing (plain TopK, no normalisation)
    with contextlib.suppress(Exception):
        flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            *_fp4_moe_args,
            top_k=8,
            n_group=None,
            topk_group=None,
            routing_method_type=5,
            **_fp4_moe_common,
        )

# ── Summary ───────────────────────────────────────────────────────────────────
files = sorted(SAVE_DIR.glob("*.json"))
print(f"\nWrote {len(files)} definition files:\n")
for f in files:
    defn = json.loads(f.read_text())
    print(f"  {f.name}")
    print(f"    op_type : {defn['op_type']}")
    print(f"    fi_api  : {next(t for t in defn['tags'] if t.startswith('fi_api:'))}")
    const_axes = {
        k: v["value"]
        for k, v in defn["axes"].items()
        if v["type"] == "const" and "value" in v
    }
    if const_axes:
        print(f"    axes    : {const_axes}")
    print()


# ── Extra APIs (category A+B additions) ───────────────────────────────────────
# Many of these require SM100+ kernels; traces dump before the kernel runs so
# the JSONs appear on any GPU. Wrap runtime-only calls in contextlib.suppress.

# append_paged_kv_cache: exercise via a single page write.
with contextlib.suppress(Exception):
    from flashinfer import append_paged_kv_cache

    _pap_B, _pap_H, _pap_D, _pap_PS = 2, 8, 128, 16
    _pap_nnz = 4
    _k_cache = torch.zeros(
        4, _pap_PS, _pap_H, _pap_D, dtype=torch.bfloat16, device=device
    )
    _v_cache = torch.zeros_like(_k_cache)
    _append_k = torch.randn(
        _pap_nnz, _pap_H, _pap_D, dtype=torch.bfloat16, device=device
    )
    _append_v = torch.randn_like(_append_k)
    _bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    _pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    _kv_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    _kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    _last = torch.tensor([2, 2], dtype=torch.int32, device=device)
    append_paged_kv_cache(
        _append_k,
        _append_v,
        _bidx,
        _pos,
        (_k_cache, _v_cache),
        _kv_idx,
        _kv_indptr,
        _last,
    )

# SegmentGEMMWrapper: small per-segment matmul.
with contextlib.suppress(Exception):
    ws = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
    seg = flashinfer.SegmentGEMMWrapper(ws)
    seg_x = torch.randn(256, 128, dtype=torch.bfloat16, device=device)
    seg_w = torch.randn(4, 128, 64, dtype=torch.bfloat16, device=device)
    seg_indptr = torch.tensor([0, 64, 128, 192, 256], dtype=torch.int64, device=device)
    seg.run(
        seg_x,
        seg_w,
        batch_size=4,
        weight_column_major=False,
        seg_indptr=seg_indptr,
    )

# softmax + sampling_from_probs + sampling_from_logits + min_p_sampling.
_sp_probs = torch.rand(64, 32000, dtype=torch.float32, device=device)
_sp_probs = _sp_probs / _sp_probs.sum(dim=-1, keepdim=True)
_sp_logits = torch.randn(64, 32000, dtype=torch.float32, device=device)
with contextlib.suppress(Exception):
    flashinfer.softmax(_sp_logits, temperature=1.0)
with contextlib.suppress(Exception):
    flashinfer.sampling_from_probs(_sp_probs)
with contextlib.suppress(Exception):
    flashinfer.sampling_from_logits(_sp_logits)
with contextlib.suppress(Exception):
    flashinfer.min_p_sampling_from_probs(_sp_probs, 0.1)
with contextlib.suppress(Exception):
    flashinfer.top_p_renorm_probs(_sp_probs, 0.9)
with contextlib.suppress(Exception):
    flashinfer.top_k_renorm_probs(_sp_probs, 50)
with contextlib.suppress(Exception):
    flashinfer.top_k_mask_logits(_sp_logits, 50)
with contextlib.suppress(Exception):
    flashinfer.top_k_top_p_sampling_from_logits(_sp_logits, 50, 0.9)

# chain_speculative_sampling.
with contextlib.suppress(Exception):
    _csd_B, _csd_S, _csd_V = 4, 3, 32000
    _draft_p = torch.softmax(
        torch.randn(_csd_B, _csd_S + 1, _csd_V, dtype=torch.float32, device=device),
        dim=-1,
    )
    _target_p = torch.softmax(
        torch.randn(_csd_B, _csd_S + 1, _csd_V, dtype=torch.float32, device=device),
        dim=-1,
    )
    _draft_ids = torch.randint(
        0,
        _csd_V,
        (_csd_B, _csd_S),
        dtype=torch.int32,
        device=device,
    )
    flashinfer.chain_speculative_sampling(_draft_p, _draft_ids, _target_p)

# rope_quantize_fp8 (GQA layout) + mla_rope_quantize_fp8 (MLA: num_k_heads=1).
with contextlib.suppress(Exception):
    _rqf_nnz = 32
    _rqf_Hq, _rqf_Hk = 8, 2
    _rqf_rope, _rqf_nope = 64, 64
    _rqf_q_rope = torch.randn(
        _rqf_nnz, _rqf_Hq, _rqf_rope, dtype=torch.bfloat16, device=device
    )
    _rqf_k_rope = torch.randn(
        _rqf_nnz, _rqf_Hk, _rqf_rope, dtype=torch.bfloat16, device=device
    )
    _rqf_q_nope = torch.randn(
        _rqf_nnz, _rqf_Hq, _rqf_nope, dtype=torch.bfloat16, device=device
    )
    _rqf_k_nope = torch.randn(
        _rqf_nnz, _rqf_Hk, _rqf_nope, dtype=torch.bfloat16, device=device
    )
    _rqf_t = torch.arange(4096, dtype=torch.float32, device=device)
    _rqf_inv = 1.0 / (
        1e4
        ** (
            torch.arange(0, _rqf_rope, 2, dtype=torch.float32, device=device)
            / _rqf_rope
        )
    )
    _rqf_cache = torch.cat(
        [
            torch.cos(_rqf_t.unsqueeze(-1) * _rqf_inv.unsqueeze(0)),
            torch.sin(_rqf_t.unsqueeze(-1) * _rqf_inv.unsqueeze(0)),
        ],
        dim=-1,
    )
    _rqf_pos = torch.arange(_rqf_nnz, dtype=torch.int32, device=device)
    from flashinfer.rope import rope_quantize_fp8 as _rope_quantize_fp8

    _rope_quantize_fp8(
        _rqf_q_rope,
        _rqf_k_rope,
        _rqf_q_nope,
        _rqf_k_nope,
        _rqf_cache,
        _rqf_pos,
        is_neox=True,
    )

with contextlib.suppress(Exception):
    _mrqf_nnz, _mrqf_Hq = 16, 128
    _mrqf_rope, _mrqf_nope = 64, 512
    _mrqf_q_rope = torch.randn(
        _mrqf_nnz, _mrqf_Hq, _mrqf_rope, dtype=torch.bfloat16, device=device
    )
    _mrqf_k_rope = torch.randn(
        _mrqf_nnz, _mrqf_rope, dtype=torch.bfloat16, device=device
    )
    _mrqf_q_nope = torch.randn(
        _mrqf_nnz, _mrqf_Hq, _mrqf_nope, dtype=torch.bfloat16, device=device
    )
    _mrqf_k_nope = torch.randn(
        _mrqf_nnz, _mrqf_nope, dtype=torch.bfloat16, device=device
    )
    _mrqf_t = torch.arange(4096, dtype=torch.float32, device=device)
    _mrqf_inv = 1.0 / (
        1e4
        ** (
            torch.arange(0, _mrqf_rope, 2, dtype=torch.float32, device=device)
            / _mrqf_rope
        )
    )
    _mrqf_cache = torch.cat(
        [
            torch.cos(_mrqf_t.unsqueeze(-1) * _mrqf_inv.unsqueeze(0)),
            torch.sin(_mrqf_t.unsqueeze(-1) * _mrqf_inv.unsqueeze(0)),
        ],
        dim=-1,
    )
    _mrqf_pos = torch.arange(_mrqf_nnz, dtype=torch.int32, device=device)
    from flashinfer.rope import mla_rope_quantize_fp8 as _mla_rope_quantize_fp8

    _mla_rope_quantize_fp8(
        _mrqf_q_rope,
        _mrqf_k_rope,
        _mrqf_q_nope,
        _mrqf_k_nope,
        _mrqf_cache,
        _mrqf_pos,
        is_neox=True,
    )

# trtllm_batch_decode_with_kv_cache_mla (DeepSeek MLA decode, SM100/103 only).
with contextlib.suppress(Exception):
    import math as _math

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    _tmla_B = 4
    _tmla_num_heads = 128
    _tmla_ckv, _tmla_kpe, _tmla_nope = 512, 64, 512
    _tmla_D_qk = _tmla_ckv + _tmla_kpe  # 576
    _tmla_q_len = 1
    _tmla_page = 64
    _tmla_seq = 128
    _tmla_n_pages = (_tmla_seq + _tmla_page - 1) // _tmla_page
    _tmla_tot = _tmla_n_pages * _tmla_B
    _tmla_query = torch.randn(
        _tmla_B,
        _tmla_q_len,
        _tmla_num_heads,
        _tmla_D_qk,
        dtype=torch.float16,
        device=device,
    )
    _tmla_kv = torch.randn(
        _tmla_tot, _tmla_page, _tmla_D_qk, dtype=torch.float16, device=device
    )
    _tmla_bt = torch.arange(_tmla_tot, dtype=torch.int32, device=device).reshape(
        _tmla_B, _tmla_n_pages
    )
    _tmla_sl = torch.full((_tmla_B,), _tmla_seq, dtype=torch.int32, device=device)
    _tmla_ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    trtllm_batch_decode_with_kv_cache_mla(
        query=_tmla_query,
        kv_cache=_tmla_kv,
        workspace_buffer=_tmla_ws,
        qk_nope_head_dim=_tmla_nope,
        kv_lora_rank=_tmla_ckv,
        qk_rope_head_dim=_tmla_kpe,
        block_tables=_tmla_bt,
        seq_lens=_tmla_sl,
        max_seq_len=_tmla_seq,
        bmm1_scale=1.0 / _math.sqrt(_tmla_D_qk),
        bmm2_scale=1.0,
        is_var_seq=False,
    )

# concat_mla_k (DeepSeek MLA K concat, fixed shape per docstring).
with contextlib.suppress(Exception):
    from flashinfer.concat_ops import concat_mla_k as _concat_mla_k

    _cmk_T, _cmk_H = 2048, 128
    _cmk_nope, _cmk_rope = 128, 64
    _cmk_k = torch.empty(
        _cmk_T, _cmk_H, _cmk_nope + _cmk_rope, dtype=torch.bfloat16, device=device
    )
    _cmk_k_nope = torch.randn(
        _cmk_T, _cmk_H, _cmk_nope, dtype=torch.bfloat16, device=device
    )
    _cmk_k_rope = torch.randn(_cmk_T, 1, _cmk_rope, dtype=torch.bfloat16, device=device)
    _concat_mla_k(_cmk_k, _cmk_k_nope, _cmk_k_rope)

# xqa_batch_decode_with_kv_cache (SM100+ XQA decode wrapper, NHD 5-D cache).
with contextlib.suppress(Exception):
    import math as _math2
    from flashinfer.decode import xqa_batch_decode_with_kv_cache as _xqa_dec

    _xqa_B, _xqa_Hq, _xqa_Hk, _xqa_D, _xqa_PS = 2, 8, 2, 128, 16
    _xqa_MP = 2
    _xqa_NP = _xqa_B * _xqa_MP
    _xqa_kvlen = _xqa_PS * _xqa_MP
    _xqa_kv = torch.randn(
        _xqa_NP, 2, _xqa_PS, _xqa_Hk, _xqa_D, dtype=torch.bfloat16, device=device
    )
    _xqa_q = torch.randn(_xqa_B, _xqa_Hq, _xqa_D, dtype=torch.bfloat16, device=device)
    _xqa_bt = torch.arange(_xqa_NP, dtype=torch.int32, device=device).reshape(
        _xqa_B, _xqa_MP
    )
    _xqa_sl = torch.full((_xqa_B,), _xqa_kvlen, dtype=torch.int32, device=device)
    _xqa_ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _xqa_dec(
        _xqa_q,
        _xqa_kv,
        _xqa_ws,
        _xqa_bt,
        _xqa_sl,
        _xqa_kvlen,
        bmm1_scale=1.0 / _math2.sqrt(_xqa_D),
        bmm2_scale=1.0,
        kv_layout="NHD",
    )

# xqa_batch_decode_with_kv_cache_mla (SM120/121 XQA MLA decode, FP8).
with contextlib.suppress(Exception):
    import math as _math3
    from flashinfer.mla import (
        xqa_batch_decode_with_kv_cache_mla as _xmla_mla_dec,
    )

    _xmla_B, _xmla_H = 2, 128
    _xmla_ckv, _xmla_kpe, _xmla_nope = 512, 64, 512
    _xmla_D = _xmla_ckv + _xmla_kpe
    _xmla_PS = 64
    _xmla_seq = 128
    _xmla_np = (_xmla_seq + _xmla_PS - 1) // _xmla_PS
    _xmla_tot = _xmla_np * _xmla_B
    _xmla_q = torch.randn(
        _xmla_B, 1, _xmla_H, _xmla_D, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    _xmla_kv = torch.randn(
        _xmla_tot, _xmla_PS, _xmla_D, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    _xmla_bt = torch.arange(_xmla_tot, dtype=torch.int32, device=device).reshape(
        _xmla_B, _xmla_np
    )
    _xmla_sl = torch.full((_xmla_B,), _xmla_seq, dtype=torch.int32, device=device)
    _xmla_ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _xmla_mla_dec(
        query=_xmla_q,
        kv_cache=_xmla_kv,
        workspace_buffer=_xmla_ws,
        qk_nope_head_dim=_xmla_nope,
        kv_lora_rank=_xmla_ckv,
        qk_rope_head_dim=_xmla_kpe,
        block_tables=_xmla_bt,
        seq_lens=_xmla_sl,
        max_seq_len=_xmla_seq,
        bmm1_scale=1.0 / _math3.sqrt(_xmla_D),
        bmm2_scale=1.0,
    )

# rope_quantize_fp8_append_paged_kv_cache (fused RoPE+FP8+append, GQA).
with contextlib.suppress(Exception):
    from flashinfer.rope import (
        rope_quantize_fp8_append_paged_kv_cache as _rqfap,
    )

    _rqfap_nnz, _rqfap_Hq, _rqfap_Hk = 16, 8, 2
    _rqfap_rope, _rqfap_nope = 64, 64
    _rqfap_d = _rqfap_rope + _rqfap_nope
    _rqfap_NP, _rqfap_PS = 4, 16
    _rqfap_q_rope = torch.randn(
        _rqfap_nnz, _rqfap_Hq, _rqfap_rope, dtype=torch.bfloat16, device=device
    )
    _rqfap_k_rope = torch.randn(
        _rqfap_nnz, _rqfap_Hk, _rqfap_rope, dtype=torch.bfloat16, device=device
    )
    _rqfap_q_nope = torch.randn(
        _rqfap_nnz, _rqfap_Hq, _rqfap_nope, dtype=torch.bfloat16, device=device
    )
    _rqfap_k_nope = torch.randn(
        _rqfap_nnz, _rqfap_Hk, _rqfap_nope, dtype=torch.bfloat16, device=device
    )
    _rqfap_v = torch.randn(
        _rqfap_nnz, _rqfap_Hk, _rqfap_d, dtype=torch.bfloat16, device=device
    )
    _rqfap_t = torch.arange(4096, dtype=torch.float32, device=device)
    _rqfap_inv = 1.0 / (
        1e4
        ** (
            torch.arange(0, _rqfap_rope, 2, dtype=torch.float32, device=device)
            / _rqfap_rope
        )
    )
    _rqfap_cache = torch.cat(
        [
            torch.cos(_rqfap_t.unsqueeze(-1) * _rqfap_inv.unsqueeze(0)),
            torch.sin(_rqfap_t.unsqueeze(-1) * _rqfap_inv.unsqueeze(0)),
        ],
        dim=-1,
    )
    _rqfap_pos = torch.arange(_rqfap_nnz, dtype=torch.int32, device=device)
    _rqfap_k_cache = torch.zeros(
        _rqfap_NP,
        _rqfap_PS,
        _rqfap_Hk,
        _rqfap_d,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _rqfap_v_cache = torch.zeros_like(_rqfap_k_cache)
    _rqfap_kv_indices = torch.arange(_rqfap_NP, dtype=torch.int32, device=device)
    _rqfap_kv_indptr = torch.tensor(
        [0, _rqfap_NP // 2, _rqfap_NP], dtype=torch.int32, device=device
    )
    _rqfap_batch_indices = torch.cat(
        [
            torch.zeros(_rqfap_nnz // 2, dtype=torch.int32, device=device),
            torch.ones(_rqfap_nnz // 2, dtype=torch.int32, device=device),
        ]
    )
    _rqfap_positions = torch.arange(_rqfap_nnz, dtype=torch.int32, device=device) % (
        _rqfap_nnz // 2
    )
    _rqfap(
        _rqfap_q_rope,
        _rqfap_k_rope,
        _rqfap_q_nope,
        _rqfap_k_nope,
        _rqfap_v,
        _rqfap_cache,
        _rqfap_pos,
        (_rqfap_k_cache, _rqfap_v_cache),
        _rqfap_kv_indices,
        _rqfap_kv_indptr,
        _rqfap_batch_indices,
        _rqfap_positions,
        is_neox=True,
        page_size=_rqfap_PS,
        kv_layout="NHD",
    )
