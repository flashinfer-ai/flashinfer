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
gdn_decode_qk4_v8_d128_k_last.json
gdn_mtp_qk4_v8_d128_k_last.json
gdn_prefill_qk4_v8_d128_k_last.json
gemm_bf16_n256_k7168.json
gemm_bf16_n4096_k4096.json
gemm_fp4_n2048_k7168.json
gemm_fp8_n1536_k7168.json
gemm_mxfp8_n4096_k4096.json
gqa_paged_decode_h32_kv8_d128_ps16.json
gqa_paged_decode_h32_kv8_d128_ps64.json
gqa_paged_prefill_h32_kv8_d128_ps16.json
gqa_ragged_prefill_h32_kv8_d128.json
mla_paged_decode_h16_ckv512_kpe64_ps1.json
mla_paged_decode_h16_ckv512_kpe64_ps64.json
moe_fp8_block_scale_default_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
moe_fp8_block_scale_llama4_routing_topk1_e32_h7168_i2048.json
moe_fp8_block_scale_renormalize_naive_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_renormalize_routing_topk8_e32_h7168_i2048.json
moe_fp8_block_scale_topk_routing_topk8_e32_h7168_i2048.json
rmsnorm_h4096.json
rmsnorm_h7168.json
top_k_sampling_from_probs_v128256.json
top_k_top_p_sampling_from_probs_v128256.json
top_k_top_p_sampling_from_probs_v151936.json
top_p_sampling_from_probs_v128256.json
top_p_sampling_from_probs_v151936.json

Note: top_p_sampling files appear for vocab_size=151936 because
top_k_top_p_sampling (top_k_first order) calls top_p_sampling internally.
"""

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

# ── GEMM bf16 ─────────────────────────────────────────────────────────────────
# Llama-3.1-8B o_proj (4096×4096) and DeepSeek-V3 moe.gate (256×7168)
# Use cutlass backend to avoid cuDNN dependency.
# mm_bf16 expects b in column-major layout with shape [K, N].
# randn(N, K).T gives shape [K, N] with strides (1, N); the kernel transposes
# b back to [N, K] (contiguous) before calling the C++ matmul.
for N, K in ((4096, 4096), (256, 7168)):
    a = torch.randn(128, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(
        N, K, dtype=torch.bfloat16, device=device
    ).T  # [K, N] column-major; b.T is contiguous
    flashinfer.mm_bf16(a, b, backend="cutlass")

# ── GEMM fp8 block-scale (DeepSeek-V3 q_proj: M×7168→1536, block=128) ────────
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
try:
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
    w2s = torch.ones(
        E_loc, H_moe // BS, I_moe // BS, dtype=torch.float32, device=device
    )
    _moe_common = dict(
        num_experts=E_tot,
        intermediate_size=I_moe,
        local_expert_offset=0,
        local_num_experts=E_loc,
        routed_scaling_factor=2.5,
    )
    _moe_args = (routing_logits, routing_bias, hs, hs_scale, w1, w1s, w2, w2s)

    # 0: Default routing (TopK -> no normalisation)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args, top_k=8, routing_method_type=0, **_moe_common
    )

    # 1: Renormalize routing (TopK -> Softmax)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args, top_k=8, routing_method_type=1, **_moe_common
    )

    # 2: DeepSeekV3 routing (Sigmoid -> group selection -> top_k=8)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args,
        top_k=8,
        n_group=8,
        topk_group=4,
        routing_method_type=2,
        **_moe_common,
    )

    # 3: Llama4 routing (Top1 -> Sigmoid)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args, top_k=1, routing_method_type=3, **_moe_common
    )

    # 4: RenormalizeNaive routing (Softmax -> TopK -> Renormalize)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args, top_k=8, routing_method_type=4, **_moe_common
    )

    # 5: TopK routing (plain TopK, no normalisation)
    flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        *_moe_args, top_k=8, routing_method_type=5, **_moe_common
    )
except Exception:
    pass  # May require specific GPU/TRT-LLM support

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
