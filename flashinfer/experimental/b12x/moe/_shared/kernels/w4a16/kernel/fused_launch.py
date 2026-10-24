"""Fused W4A16 MoE compilation and launch bindings."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x._lib.quant.block_codec import IQ2_CODECS
from cutlass.cutlass_dsl import Int32
from cutlass.cutlass_dsl import Int64
from b12x._lib.compiler import KernelCompileSpec
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from cutlass.base_dsl.compiler import OptLevel
from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import compile as b12x_compile
import cuda.bindings.driver as cuda
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
import cutlass
from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut
from b12x.moe._shared.kernels.activations import is_gated_moe_activation
from b12x._lib.utils.pointers import make_ptr
from b12x.moe._shared.kernels.activations import normalize_moe_activation
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.program_cache import register_program_cache
from dataclasses import replace
import torch
from b12x.moe._shared.kernels.w4a16.host import validate_activation
from b12x.moe._shared.kernels.w4a16.kernel.buffers import _rot_scales_dummy
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _MAX_DIRECT_TOPK_ROUTE_M,
    _MODEL_OPT_W13_LAYOUTS,
    _TC_DECODE_MAX_M,
    _TRELLIS256_BITS,
    _TRELLIS256_W13_LAYOUTS,
    _W4A16_SMALL_M_DIRECT_MAX_M,
    _WEIGHT_LAYOUTS,
    _candidate_tile_fits,
    _covering_count,
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _normalize_activation_swiglu_params,
    _normalize_element_dtype,
    _normalize_scale_format,
    _select_tile_config,
    _trellis256_execution_lut,
    _w4a16_small_m_occupancy,
    _w4a16_small_m_splitk_enabled,
)
from b12x.moe._shared.kernels.w4a16.kernel.direct_launch import (
    _compile_w4a16_small_m_direct,
    _small_m_direct_supported,
)
from b12x.moe._shared.kernels.w4a16.kernel.fused import W4A16FusedMoeKernel
from b12x.moe._shared.kernels.w4a16.kernel.types import W4A16FusedMoeCompileResult


_FUSED_CACHE: dict[tuple, W4A16FusedMoeCompileResult] = {}

register_program_cache(_FUSED_CACHE)


def compile_w4a16_fused_moe(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str = "bf16",
    fast_math: bool = True,
    sms: int,
    max_shared_mem: int,
    direct_token_capacity: int | None = None,
    swiglu_limit: float | None = None,
    swiglu_clamp_after_silu: bool = False,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "w13",
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    fc1_trellis_pair_kind: str | None = None,
    fc2_trellis_pair_kind: str | None = None,
    direct_topk_routes: bool = False,
    use_expert_map: bool = False,
    tc_decode_fused_sum: bool = False,
    prefill_fused_sum_fp32: bool = False,
    collect_activation_amax: bool = False,
    force_tile_config: tuple[int, int, int, int] | None = None,
    pipeline_stages: int | None = None,
    skip_empty_m_blocks: bool | None = None,
    intermediate_rotation: bool = False,
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    rotation_input_dtype: str | None = None,
    broadcast_suh: bool = False,
    trellis_decode_table: str = "auto",
    _require_cached: bool = False,
    small_m_occupancy: int | None = None,
    csf_inline_scales: bool = False,
) -> W4A16FusedMoeCompileResult:
    if small_m_occupancy is None:
        small_m_occupancy = _w4a16_small_m_occupancy()
    scale_format = _normalize_scale_format(scale_format)
    intermediate_rotation = bool(intermediate_rotation)
    full_rotation = bool(full_rotation)
    intermediate_hadamard = bool(intermediate_hadamard)
    rotation_input_dtype = (
        element_dtype if rotation_input_dtype is None else str(rotation_input_dtype)
    )
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    device = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    activation = normalize_moe_activation(activation)
    is_gated = validate_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    if weight_layout not in _WEIGHT_LAYOUTS:
        raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
    trellis_bits = int(trellis_bits)
    if weight_layout == "trellis_t256":
        if trellis_bits not in _TRELLIS256_BITS:
            raise ValueError(
                f"trellis_t256 bits must be one of {_TRELLIS256_BITS}, got {trellis_bits}"
            )
    elif trellis_bits != 3:
        raise ValueError("trellis_bits is only valid for trellis_t256 weights")
    # Existing 3-bpw scheduling was conservatively planned as 4 bpw. Keep that
    # grid contract stable for D6; widen only the 5/6-bpw specializations.
    weight_bits = max(4, trellis_bits) if weight_layout == "trellis_t256" else 4
    # GATE 5: the PRODUCTION 256-weight-tile fused-megakernel B-staging is now
    # wired (per-warp native [K/16,N/16,8*bits u32] tile staging + the per-lane
    # bitrate-specialized read) and ADMITTED at 3 bpw against a full-GEMM
    # numeric oracle (megakernel FC1->silu->FC2 vs a torch reference over tiles;
    # tests/test_trellis256_fullgemm_oracle.py, max rel-err <= 2^-6).  The GATE-4
    # NotImplementedError gate is therefore lifted.  See GATE5_RESULT.md.
    if weight_layout == "modelopt":
        if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
            raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
    elif weight_layout == "trellis_t256":
        if w13_layout not in _TRELLIS256_W13_LAYOUTS:
            raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
    else:
        w13_layout = "packed"
    direct_topk_routes = bool(direct_topk_routes)
    use_expert_map = bool(use_expert_map)
    tc_decode_fused_sum = bool(tc_decode_fused_sum)
    prefill_fused_sum_fp32 = bool(prefill_fused_sum_fp32)
    if tc_decode_fused_sum and prefill_fused_sum_fp32:
        raise ValueError(
            "TC-decode and large-M FP32 route reduction are mutually exclusive"
        )
    if use_expert_map and not direct_topk_routes:
        raise ValueError("use_expert_map requires direct_topk_routes")
    collect_activation_amax = bool(collect_activation_amax)
    if collect_activation_amax and (
        direct_topk_routes or tc_decode_fused_sum or prefill_fused_sum_fp32
    ):
        raise ValueError(
            "W4A16 activation amax collection requires the route-packed fused path"
        )
    if collect_activation_amax and intermediate_rotation:
        raise ValueError(
            "W4A16 activation amax collection is incompatible with intermediate rotation"
        )
    if full_rotation:
        if not intermediate_rotation:
            raise ValueError("full_rotation requires intermediate_rotation")
        if weight_layout != "trellis_t256":
            raise ValueError("full_rotation is only supported for trellis_t256")
        if element_dtype != "fp16":
            raise ValueError("full_rotation requires element_dtype='fp16'")
        if rotation_input_dtype not in {"bf16", "fp16"}:
            raise ValueError(
                "rotation_input_dtype must be 'bf16' or 'fp16' for full_rotation"
            )
        if tc_decode_fused_sum or prefill_fused_sum_fp32:
            raise ValueError("full_rotation is incompatible with fused route reduction")
        if apply_router_weight_on_input:
            raise ValueError(
                "full_rotation requires apply_router_weight_on_input=False"
            )
    if intermediate_hadamard and not full_rotation:
        raise ValueError("intermediate_hadamard requires full_rotation")
    if collect_activation_amax and weight_layout == "trellis_t256":
        raise NotImplementedError(
            "trellis_t256 activation-amax collection is not exposed through the "
            "registered launch ABI; refusing to compile a bitrate-ambiguous kernel"
        )
    # The TC-decode path validates M in {1,2,4,8} itself and uses direct-topk
    # routing for the whole {1,2,4,8} range, so it lifts the default decode cap.
    direct_topk_m_cap = (
        _W4A16_SMALL_M_DIRECT_MAX_M
        if tc_decode_fused_sum or use_expert_map
        else _MAX_DIRECT_TOPK_ROUTE_M
    )
    direct_weight_layout_ok = weight_layout in {
        "packed",
        "iq2_xs",
        "iq2_xxs",
        "q8_0",
    } or (full_rotation and weight_layout == "trellis_t256")
    if direct_topk_routes and (
        int(size_m) > direct_topk_m_cap
        or not direct_weight_layout_ok
        or bool(zero_fc2_output)
    ):
        raise ValueError(
            "direct_topk_routes is only valid for supported small-M W4A16 decode"
        )
    fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
    routed_rows = int(size_m) * int(top_k)
    # Logical K/N tails are needed for every shard the tile table can't
    # divide, not just sub-32 ones: 2048/TP6 = 352 and 3072/TP16 = 192 are
    # 32-aligned yet have no dividing tile_k/tile_n. %32 shards are the
    # ceil-scale-grid subset of the same machinery (%32 != 0 implies
    # %128 != 0).
    allow_native_logical_tail = (
        weight_layout == "modelopt"
        and scale_format == "e8m0_k32"
        and int(intermediate_size) % 128 != 0
    )
    fc1_tile_k, fc1_tile_n, fc1_cta_threads, _ = _select_tile_config(
        problem_m=size_m,
        problem_n=fc1_cols,
        problem_k=hidden_size,
        top_k=top_k,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format=scale_format,
        weight_layout=weight_layout,
        weight_bits=weight_bits,
        allow_logical_tail=allow_native_logical_tail,
        small_m_occupancy=small_m_occupancy,
    )
    fc2_tile_k, fc2_tile_n, fc2_cta_threads, _ = _select_tile_config(
        problem_m=routed_rows,
        problem_n=hidden_size,
        problem_k=intermediate_size,
        top_k=1,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format=scale_format,
        weight_layout=weight_layout,
        weight_bits=weight_bits,
        allow_logical_tail=allow_native_logical_tail,
        small_m_occupancy=small_m_occupancy,
    )
    if fc1_cta_threads != fc2_cta_threads:
        common_cta_threads = min(fc1_cta_threads, fc2_cta_threads)
        fc1_tile_k, fc1_tile_n, fc1_cta_threads, _ = _select_tile_config(
            problem_m=size_m,
            problem_n=fc1_cols,
            problem_k=hidden_size,
            top_k=top_k,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            required_cta_threads=common_cta_threads,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=weight_bits,
            allow_logical_tail=allow_native_logical_tail,
            small_m_occupancy=small_m_occupancy,
        )
        fc2_tile_k, fc2_tile_n, fc2_cta_threads, _ = _select_tile_config(
            problem_m=routed_rows,
            problem_n=hidden_size,
            problem_k=intermediate_size,
            top_k=1,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            required_cta_threads=common_cta_threads,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=weight_bits,
            allow_logical_tail=allow_native_logical_tail,
            small_m_occupancy=small_m_occupancy,
        )
        if fc1_cta_threads != fc2_cta_threads:
            raise ValueError(
                "fused W4A16 FC1/FC2 selected different thread counts: "
                f"{fc1_cta_threads} vs {fc2_cta_threads}"
            )
    # TC-decode FC1 wide-N override (single-wave-collapse sizes only): in the m8
    # fused path the host right-sizes grid_x to the FC1 mn-tile count and forces
    # one whole mn-tile per CTA over the full K. With the default fc1_tile_n=128,
    # FC1 (N=fc1_cols=2*intermediate_size) produces size_m*top_k*(fc1_cols/128)
    # mn-tiles. For TP=2 I_tp=1024 (fc1_cols=2048 => 16 n-tiles/route) bs=1 is
    # 96 tiles (<= 188 SMs, one wave); bs=2 is 192 -- JUST over the single-wave
    # SM cap -- forcing a 2-wave launch of grid_x=96 (half the machine idle per
    # wave, a serialized second FC1 wave of pure tail latency on the bandwidth-
    # bound decode). Widening FC1 to tile_n=256 (256-wide N slab per CTA over
    # full K; tile_k=64 keeps cta_threads=256) HALVES FC1's mn-tile count, so
    # bs=2 collapses to 96 tiles = exactly ONE wave, removing that whole second
    # FC1 wave. The narrower tile_k=64 is slower per-tile, so we apply this ONLY
    # where it turns a 2-wave launch into a 1-wave launch: the default 128-wide
    # FC1 spans 2 waves (sms < default_mn_tiles <= 2*sms) AND the wide tile fits
    # in one (default/2 <= sms). bs=1 (one wave already) and bs>=4 (still multi-
    # wave after halving) keep the faster default 128x128 tile.
    # Guarded by fc1_cols%256==0, smem-fit, and 256-thread geometry so the fused
    # FC1/FC2 single-thread-geometry contract is preserved.
    default_fc1_mn_tiles = (
        int(size_m) * int(top_k) * (int(fc1_cols) // int(fc1_tile_n))
        if fc1_tile_n > 0
        else 0
    )
    if (
        bool(tc_decode_fused_sum)
        and weight_layout not in BLOCK_CODECS
        and int(fc1_cols) % 256 == 0
        and fc1_tile_n == 128
        and (fc1_tile_n * fc1_tile_k) // 64 == 256
        and int(sms) < default_fc1_mn_tiles <= 2 * int(sms)
        and (default_fc1_mn_tiles // 2) <= int(sms)
    ):
        wide_fc1_tile_k = 64
        if _candidate_tile_fits(
            problem_n=fc1_cols,
            problem_k=hidden_size,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=256,
            tile_k=wide_fc1_tile_k,
            cta_threads=256,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=weight_bits,
            small_m_occupancy=small_m_occupancy,
        ):
            fc1_tile_n = 256
            fc1_tile_k = wide_fc1_tile_k
            fc1_cta_threads = 256
    # Packed decode FC2 wide-N override: the direct path right-sizes grid_x to
    # the FC1 mn-tile count, while the expert-packed path uses the same
    # persistent cap with a device-known live block count. FC2
    # (N=hidden_size, K=intermediate_size) with the default tile_n=128 produces
    # route_blocks*(hidden_size/128) mn-tiles -- roughly double FC1's count --
    # so FC2 would need ~2 persistent waves while FC1 fits in 1; that second
    # FC2 wave is pure serialized tail latency on the bandwidth-bound decode.
    # Selecting tile_n=256 for FC2 (a 256-wide N slab per CTA over full K)
    # halves FC2's mn-tile count so it also fits one wave. Guarded by smem-fit,
    # hidden_size%256==0, and matching cta_threads so the fused single
    # thread-geometry contract is preserved.
    if (
        weight_layout == "packed"
        and int(moe_block_size) == 8
        and int(size_m) <= _TC_DECODE_MAX_M
        and int(hidden_size) % 256 == 0
        and fc2_tile_n == 128
        and fc1_cta_threads == 256
    ):
        wide_fc2_tile_k = 64
        if _candidate_tile_fits(
            problem_n=hidden_size,
            problem_k=intermediate_size,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=256,
            tile_k=wide_fc2_tile_k,
            cta_threads=256,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=weight_bits,
            small_m_occupancy=small_m_occupancy,
        ):
            fc2_tile_n = 256
            fc2_tile_k = wide_fc2_tile_k
            fc2_cta_threads = 256
    # TC-decode FC2 ultra-wide override (perfect wave-balance with FC1): the
    # persistent grid_x is right-sized to FC1's mn-tile count. After the FC1/FC2
    # wide-N (tile_n=256) overrides, bs=2 still has FC1=route_blocks*8 tiles vs
    # FC2=route_blocks*16 tiles -- FC2 is DOUBLE FC1 and thus needs a second
    # persistent wave at grid_x sized to FC1. That FC2 second wave is pure
    # serialized tail latency on the bandwidth-bound decode. Widening FC2 to
    # tile_n=512 (a 512-wide N slab per CTA over full K) halves FC2's mn-tile
    # count again so FC2 == FC1's tile count and fits the SAME single wave. A
    # 512-wide N tile needs tile_k=32 to keep cta_threads=256 (512*32/64); that
    # is below the generic tile_k>=64 fits-floor, so we validate the footprint
    # directly here. tile_k=32 == the e8m0_k32 scale group, so cta_k_blocks=2
    # with one e8m0 scale group per k-tile -- the existing scale layout is a
    # clean covering. Fire ONLY when it drops FC2 from >1 wave to FC1's wave
    # count (bs=2). Numerically identical: only the FC2 output-tile width changes.
    fc1_mn_after = (
        int(size_m) * int(top_k) * (int(fc1_cols) // int(fc1_tile_n))
        if fc1_tile_n > 0
        else 0
    )
    fc2_mn_after = (
        int(size_m) * int(top_k) * (int(hidden_size) // int(fc2_tile_n))
        if fc2_tile_n > 0
        else 0
    )
    if (
        bool(tc_decode_fused_sum)
        and int(hidden_size) % 512 == 0
        and fc2_tile_n == 256
        and fc1_cta_threads == 256
        and fc1_mn_after > 0
        and fc2_mn_after > fc1_mn_after
        and (fc2_mn_after // 2) <= fc1_mn_after
        and fc1_mn_after <= int(sms)
    ):
        ultra_fc2_tile_k = 32
        if _candidate_tile_fits(
            problem_n=hidden_size,
            problem_k=intermediate_size,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=512,
            tile_k=ultra_fc2_tile_k,
            cta_threads=256,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=weight_bits,
            allow_qualified_fc2_tile=True,
            small_m_occupancy=small_m_occupancy,
        ):
            fc2_tile_n = 512
            fc2_tile_k = ultra_fc2_tile_k
            fc2_cta_threads = 256
    if force_tile_config is not None:
        # Some weight layouts are packed for a specific CTA N-tile. An explicit
        # (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) tuple therefore pins
        # one geometry across every M regime. Tile values are part of the GEMM
        # cache key, so distinct packed layouts cannot collide.
        fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
            int(v) for v in force_tile_config
        )
        fc1_cta_threads = (fc1_tile_n * fc1_tile_k) // 64
        fc2_cta_threads = (fc2_tile_n * fc2_tile_k) // 64
        if fc1_cta_threads != fc2_cta_threads:
            raise ValueError(
                "force_tile_config FC1/FC2 thread counts must match, got "
                f"{fc1_cta_threads} vs {fc2_cta_threads}"
            )
        for name, forced_pn, forced_pk, forced_tn, forced_tk in (
            ("fc1", fc1_cols, hidden_size, fc1_tile_n, fc1_tile_k),
            ("fc2", hidden_size, intermediate_size, fc2_tile_n, fc2_tile_k),
        ):
            if not _candidate_tile_fits(
                problem_n=forced_pn,
                problem_k=forced_pk,
                cta_m_blocks=_covering_count(moe_block_size, 16),
                tile_n=forced_tn,
                tile_k=forced_tk,
                cta_threads=fc1_cta_threads,
                max_shared_mem=int(max_shared_mem) - 512,
                scale_format=scale_format,
                weight_layout=weight_layout,
                weight_bits=weight_bits,
                allow_logical_tail=allow_native_logical_tail,
                allow_qualified_fc2_tile=name == "fc2",
                allow_k32_tile=name == "fc2",
                uses_m_block_8=moe_block_size == 8,
                pipeline_stages=pipeline_stages,
                small_m_occupancy=small_m_occupancy,
            ):
                raise ValueError(
                    f"force_tile_config {name} tile "
                    f"(tile_k={forced_tk}, tile_n={forced_tn}) does not fit "
                    f"problem N/K={forced_pn}/{forced_pk} at "
                    f"moe_block_size={moe_block_size}"
                )
    kernel = W4A16FusedMoeKernel(
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=fast_math,
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        fc1_trellis_pair_kind=fc1_trellis_pair_kind,
        fc2_trellis_pair_kind=fc2_trellis_pair_kind,
        direct_topk_routes=direct_topk_routes,
        use_expert_map=use_expert_map,
        tc_decode_fused_sum=tc_decode_fused_sum,
        prefill_fused_sum_fp32=prefill_fused_sum_fp32,
        collect_activation_amax=collect_activation_amax,
        intermediate_rotation=intermediate_rotation,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        rotation_input_dtype=rotation_input_dtype,
        broadcast_suh=broadcast_suh,
        pipeline_stages=pipeline_stages,
        skip_empty_m_blocks=skip_empty_m_blocks,
        trellis_decode_table=trellis_decode_table,
        small_m_occupancy=small_m_occupancy,
        csf_inline_scales=csf_inline_scales,
    )
    cache_key = (
        "w4a16_fused_moe",
        device,
        kernel.__cache_key__,
    )
    cached = _FUSED_CACHE.get(cache_key)
    if cached is None and _require_cached:
        raise RuntimeError(
            "W4A16 fused MoE launch is not resolved for CUDA graph capture "
            f"(m={size_m}, moe_block_size={moe_block_size}, "
            f"max_m_blocks={max_m_blocks}); run an eager warmup at this token "
            "count before capturing"
        )

    # Packed routes use rounded storage capacity; native direct launches use
    # the exact planned row count, which may be smaller than that bucket.
    direct_m = size_m if direct_token_capacity is None else int(direct_token_capacity)
    if not 0 < direct_m <= size_m:
        raise ValueError("direct token capacity must be within packed capacity")
    small_m_direct_launches = []
    if (not collect_activation_amax) and _small_m_direct_supported(
        m=direct_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=top_k,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        element_dtype=element_dtype,
        weight_layout=weight_layout,
        w13_layout=w13_layout,
        scale_format=scale_format,
    ):
        for ids_dtype in (torch.int32, torch.int64):
            direct = _compile_w4a16_small_m_direct(
                m=direct_m,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                topk=top_k,
                activation=activation,
                fast_math=fast_math,
                topk_ids_dtype=ids_dtype,
                scale_format=scale_format,
                swiglu_limit=swiglu_limit,
                swiglu_clamp_after_silu=swiglu_clamp_after_silu,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                w13_layout=w13_layout,
                device=torch.device("cuda", device) if device is not None else None,
            )
            small_m_direct_launches.append(direct)

    if cached is not None:
        return attach_programs(
            replace(
                cached,
                size_m=size_m,
                num_experts=num_experts,
                max_m_blocks=max_m_blocks,
                blocks_per_sm=kernel.blocks_per_sm,
                small_m_direct_launches=tuple(small_m_direct_launches),
            ),
            cached.compiled,
            *(launch.compiled for launch in small_m_direct_launches),
        )

    compile_size_m = _fake_m_for_specialization(size_m)
    compile_routed_rows = int(compile_size_m) * int(top_k)
    compile_route_blocks = compile_routed_rows if direct_topk_routes else 1
    compile_route_slots = compile_route_blocks * int(moe_block_size)
    packed_route_fake_elements = (
        compile_routed_rows if direct_topk_routes else compile_route_slots
    )
    a_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    rotation_input_fake = make_ptr(
        _cutlass_element_dtype(rotation_input_dtype),
        16,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    weight_cutlass_dtype = (
        cutlass.Uint8 if weight_layout == "modelopt" else cutlass.Int32
    )
    # Weight and E-sized metadata tensors are pointer-only launch arguments.
    # Their runtime views are reconstructed inside W4A16FusedMoeKernel.__call__
    # from weight_num_experts, keeping the compiled ABI independent of a
    # layer's compact tier expert count.
    w13_fake = make_ptr(
        weight_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w2_fake = make_ptr(
        weight_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    fc1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_routed_rows * fc1_cols,),
        assumed_align=16,
    )
    activated_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_routed_rows * intermediate_size,),
        assumed_align=16,
    )
    fc2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32 if kernel.prefill_fused_sum_fp32 else cutlass_dtype,
        (
            (compile_size_m if kernel.prefill_fused_sum_fp32 else compile_routed_rows)
            * hidden_size,
        ),
        assumed_align=16,
    )
    pair_metadata_cutlass_dtype = (
        cutlass.Int64 if fc1_trellis_pair_kind == "P33_P43" else cutlass.Int32
    )
    w13_scales_fake = make_ptr(
        pair_metadata_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w2_scales_fake = make_ptr(
        pair_metadata_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w13_global_fake = make_ptr(
        cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w2_global_fake = make_ptr(
        cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    packed_routes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (packed_route_fake_elements,),
        assumed_align=16,
    )
    block_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (compile_route_blocks,),
        assumed_align=16,
    )
    route_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    activation_amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts * 2,),
        assumed_align=4,
    )
    topk_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    fc1_c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                fc1_cols * compile_route_slots,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    fc2_c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                hidden_size * compile_route_slots,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256 + 2,),
        assumed_align=16,
    )
    rot_scales_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    suh_gate_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    suh_up_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    expert_map_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    trellis_lut_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        a_fake,
        a_fake,
        rotation_input_fake,
        w13_fake,
        w2_fake,
        Int64(1),
        Int64(1),
        fc1_fake,
        activated_fake,
        fc2_fake,
        w13_scales_fake,
        w2_scales_fake,
        w13_global_fake,
        w2_global_fake,
        packed_routes_fake,
        block_experts_fake,
        route_count_fake,
        activation_amax_fake,
        0,
        topk_fake,
        fc1_c_tmp_fake,
        fc2_c_tmp_fake,
        locks_fake,
        rot_scales_fake,
        suh_gate_fake,
        suh_up_fake,
        expert_map_fake,
        trellis_lut_fake,
        trellis_lut_fake,
        Int32(num_experts),
        Int32(num_experts),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.fused_moe",
            8,
            cache_key,
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = W4A16FusedMoeCompileResult(
        compiled=compiled,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        zero_fc2_output=bool(zero_fc2_output),
        element_dtype=element_dtype,
        fast_math=bool(fast_math),
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
        weight_layout=weight_layout,
        w13_layout=w13_layout,
        direct_topk_routes=kernel.direct_topk_routes,
        use_expert_map=kernel.use_expert_map,
        scale_format=scale_format,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        prefill_fused_sum_fp32=bool(prefill_fused_sum_fp32),
        collect_activation_amax=collect_activation_amax,
        schedule_whole_tiles=kernel.schedule_whole_tiles,
        intermediate_rotation=intermediate_rotation,
        dual_a=kernel.dual_a,
        trellis_bits=trellis_bits,
        trellis_codebook=kernel.trellis_codebook,
        fc1_trellis_pair_kind=kernel.fc1_trellis_pair_kind,
        fc2_trellis_pair_kind=kernel.fc2_trellis_pair_kind,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        rotation_input_dtype=rotation_input_dtype,
        cta_threads=kernel.cta_threads,
        shared_memory_bytes=kernel.shared_words * 4,
        broadcast_suh=bool(broadcast_suh),
        small_m_direct_launches=tuple(small_m_direct_launches),
        trellis_direct_lut=kernel.trellis_direct_lut,
    )
    attach_programs(
        result, compiled, *(launch.compiled for launch in small_m_direct_launches)
    )
    _FUSED_CACHE[cache_key] = result
    return result


def _w4a16_fused_moe_launch_flat(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    activation_amax: torch.Tensor | None,
    layer_idx: int,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    direct_topk_routes: bool,
    tc_decode_fused_sum: bool,
    prefill_fused_sum_fp32: bool,
    collect_activation_amax: bool,
    stream_int: int,
    expert_map: torch.Tensor | None = None,
    rot_scales: torch.Tensor | None = None,
    intermediate_rotation: bool = False,
    a_input_up: torch.Tensor | None = None,
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    fc1_trellis_pair_kind: str | None = None,
    fc2_trellis_pair_kind: str | None = None,
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    rotation_input: torch.Tensor | None = None,
    suh_gate_table: torch.Tensor | None = None,
    suh_up_table: torch.Tensor | None = None,
    launcher: W4A16FusedMoeCompileResult | None = None,
) -> None:
    swiglu_limit = float(swiglu_limit_value) if has_swiglu_limit else None
    collect_activation_amax = bool(collect_activation_amax)
    intermediate_rotation = bool(intermediate_rotation)
    full_rotation = bool(full_rotation)
    intermediate_hadamard = bool(intermediate_hadamard)
    use_expert_map = expert_map is not None
    if use_expert_map:
        if not direct_topk_routes:
            raise ValueError("expert_map launch requires direct_topk_routes")
        assert expert_map is not None
        if (
            expert_map.dtype != torch.int32
            or expert_map.device != a_input.device
            or expert_map.ndim != 1
            or not expert_map.is_contiguous()
        ):
            raise ValueError("expert_map must be contiguous int32 on the input device")
    if collect_activation_amax and activation_amax is None:
        raise ValueError("activation_amax is required for calibrated W4A16 launch")
    activation_amax_arg = (
        activation_amax if activation_amax is not None else w13_global_scale
    ).view(-1)
    if intermediate_rotation:
        if rot_scales is None:
            raise ValueError("intermediate_rotation launch requires rot_scales")
        rot_scales_arg = rot_scales.view(-1)
    else:
        rot_scales_arg = _rot_scales_dummy(w13_global_scale.device)
    if full_rotation and a_input_up is None:
        raise ValueError("full_rotation launch requires a distinct up A scratch")
    if a_input_up is None:
        a_input_up = a_input
    if (
        full_rotation
        and not intermediate_hadamard
        and a_input_up.data_ptr() == a_input.data_ptr()
    ):
        raise ValueError("full_rotation gate/up A scratches must not alias")
    if full_rotation and rotation_input is None:
        raise ValueError("full_rotation launch requires the raw rotation input")
    if rotation_input is None:
        rotation_input = a_input
    broadcast_suh = False
    if full_rotation:
        if suh_gate_table is None or suh_up_table is None:
            raise ValueError(
                "full_rotation launch requires suh_gate_table and suh_up_table"
            )
        suh_gate_arg = suh_gate_table.reshape(-1)
        suh_up_arg = suh_up_table.reshape(-1)
        broadcast_suh = suh_gate_arg.numel() == hidden_size
        if broadcast_suh != (suh_up_arg.numel() == hidden_size):
            raise ValueError(
                "suh gate/up tables must both be per-expert or both broadcast"
            )
        rotation_input_dtype = _normalize_element_dtype(rotation_input.dtype)
    else:
        suh_gate_arg = _rot_scales_dummy(w13_global_scale.device)
        suh_up_arg = suh_gate_arg
        rotation_input_dtype = element_dtype
    if launcher is not None and launcher.broadcast_suh != broadcast_suh:
        raise ValueError(
            "prepared W4A16 input rotation layout differs from the bound tables"
        )
    fused = launcher or compile_w4a16_fused_moe(
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        zero_fc2_output=bool(zero_fc2_output),
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=bool(fast_math),
        sms=sms,
        max_shared_mem=max_shared_mem,
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        fc1_trellis_pair_kind=fc1_trellis_pair_kind,
        fc2_trellis_pair_kind=fc2_trellis_pair_kind,
        direct_topk_routes=bool(direct_topk_routes),
        use_expert_map=use_expert_map,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        prefill_fused_sum_fp32=bool(prefill_fused_sum_fp32),
        collect_activation_amax=collect_activation_amax,
        # The custom-op boundary cannot carry the compiled launch object. Re-pin
        # its selected geometry so tile-specific packs resolve the
        # identical cache entry instead of silently recompiling with auto tiles.
        force_tile_config=(fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n),
        intermediate_rotation=intermediate_rotation,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        rotation_input_dtype=rotation_input_dtype,
        broadcast_suh=broadcast_suh,
    )
    weight_cutlass_dtype = (
        cutlass.Uint8 if weight_layout == "modelopt" else cutlass.Int32
    )
    expert_map_addr = (
        packed_route_indices.data_ptr() if expert_map is None else expert_map.data_ptr()
    )
    route_num_experts = 0 if expert_map is None else int(expert_map.numel())
    if weight_layout in IQ2_CODECS:
        iq2_lut = iq2_xs_execution_lut(
            a_input.device, selectors=True, codec=weight_layout
        )
        fc1_trellis_lut_addr = iq2_lut.data_ptr()
        fc2_trellis_lut_addr = iq2_lut.data_ptr()
    elif weight_layout == "trellis_t256" and trellis_codebook != "mcg":
        trellis_rank_lut = _trellis256_execution_lut(
            a_input.device, trellis_codebook, direct_lut=fused.trellis_direct_lut
        )
        fc1_trellis_lut_addr = trellis_rank_lut.data_ptr()
        fc2_trellis_lut_addr = trellis_rank_lut.data_ptr()
    else:
        # Non-trellis and MCG kernels never dereference this ABI slot.
        fc1_trellis_lut_addr = w13_scale_i32.data_ptr()
        fc2_trellis_lut_addr = w13_scale_i32.data_ptr()
    fused.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            a_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            a_input_up.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            _cutlass_element_dtype(rotation_input_dtype),
            rotation_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            weight_cutlass_dtype,
            w13_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            weight_cutlass_dtype,
            w2_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int64(w13_arg.numel()),
        Int64(w2_arg.numel()),
        fc1_out,
        activated,
        fc2_out,
        make_ptr(
            cutlass.Int64 if fc1_trellis_pair_kind == "P33_P43" else cutlass.Int32,
            w13_scale_i32.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int64 if fc2_trellis_pair_kind == "P33_P43" else cutlass.Int32,
            w2_scale_i32.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            w13_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            w2_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        activation_amax_arg,
        int(layer_idx),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        fc1_scratch,
        fc2_scratch,
        workspace,
        make_ptr(
            cutlass.Float16,
            rot_scales_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float16,
            suh_gate_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float16,
            suh_up_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            expert_map_addr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Uint8,
            fc1_trellis_lut_addr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            fc2_trellis_lut_addr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(num_experts),
        Int32(route_num_experts),
        m,
        _w4a16_fused_persistent_grid_x(
            fused=fused,
            m=m,
            topk=topk,
            intermediate_size=intermediate_size,
            activation=activation,
            direct_topk_routes=bool(direct_topk_routes),
            sms=sms,
        ),
        cuda.CUstream(stream_int),
    )


def _w4a16_fused_persistent_grid_x(
    *,
    fused: W4A16FusedMoeCompileResult,
    m: int,
    topk: int,
    intermediate_size: int,
    activation: str,
    direct_topk_routes: bool,
    sms: int,
) -> int:
    """Right-size the persistent grid for the fused FC1+FC2 launch.

    The default over-subscribes the cooperative grid (sms*blocks_per_sm CTAs)
    and forces the larger GEMM into the cross-CTA split-K tail (lock-serialized
    finalize + c_tmp gmem round-trip) whenever its mn-tile count exceeds the
    grid. For the small-M direct-topk decode the host knows the exact FC1
    mn-tile count, so pick the fewest full persistent waves that fit the
    co-residency cap and set grid_x to that wave's tile count. Then every CTA
    owns whole FC1 (and FC2, whose tile count is an integer multiple) mn-tiles
    over the full K -- no split-K reduction, no lock traffic -- with fewer
    grid-barrier participants, while staying <= the cap so the cooperative
    barrier never deadlocks. Falls back to the default for the route-pack path
    where the host cannot know route_blocks ahead of the launch.
    """
    cap = int(sms) * int(fused.blocks_per_sm)
    if (
        fused.weight_layout in BLOCK_CODECS and fused.moe_block_size == 8
    ) or _w4a16_small_m_splitk_enabled():
        # Stripe split-K wants the full persistent grid: each small-M FC1
        # column's K range is fanned across many CTAs, so right-sizing the
        # grid to the mn-tile count would recreate the serialization the
        # flag exists to remove.
        return max(cap, 1)
    if not direct_topk_routes or m <= 0:
        return max(cap, 1)
    is_gated = is_gated_moe_activation(activation)
    fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
    fc1_tile_n = int(getattr(fused, "fc1_tile_n", 0))
    if fc1_tile_n <= 0 or fc1_cols % fc1_tile_n != 0:
        return max(cap, 1)
    n_tiles = fc1_cols // fc1_tile_n
    route_blocks = int(m) * int(topk)
    fc1_mn_tiles = route_blocks * n_tiles
    if fc1_mn_tiles <= 0 or cap <= 0:
        return max(cap, 1)
    if bool(getattr(fused, "schedule_whole_tiles", False)):
        # Whole-tile scheduling tolerates ragged waves, so the whole-cover
        # constraint below does not apply. Minimize the FC1+FC2 critical path
        # in whole-tile waves per CTA; ties go to the smaller grid (fewer
        # grid-barrier participants). Measured on the GLM-5.2 TP4 hybrid
        # shard (m=4: FC2 768 tiles), cap=188 CTAs runs FC2 in 5-deep waves
        # vs 6-deep at the FC1-right-sized 128 CTAs: 95.9us vs 107.1us.
        fc2_tile_n = int(getattr(fused, "fc2_tile_n", 0))
        hidden = int(getattr(fused, "hidden_size", 0))
        if fc2_tile_n <= 0 or hidden <= 0 or hidden % fc2_tile_n != 0:
            return max(cap, 1)
        fc2_mn_tiles = route_blocks * (hidden // fc2_tile_n)

        def whole_tile_critical_path(grid: int) -> int:
            return -(-fc1_mn_tiles // grid) + -(-fc2_mn_tiles // grid)

        candidates = sorted({int(cap), min(fc1_mn_tiles, int(cap))})
        return min(candidates, key=lambda g: (whole_tile_critical_path(g), g))
    waves = (fc1_mn_tiles + cap - 1) // cap
    if waves <= 0:
        return max(cap, 1)
    # Only commit to the right-sized grid when every wave is a whole tile cover
    # (no remainder => no split-K tail); otherwise keep the safe default.
    if fc1_mn_tiles % waves != 0:
        return max(cap, 1)
    grid_x = fc1_mn_tiles // waves
    if grid_x < 1 or grid_x > cap:
        return max(cap, 1)
    return grid_x


@torch.library.custom_op(
    "b12x::w4a16_fused_moe_launch",
    mutates_args="unknown",
)
def _w4a16_fused_moe_launch_op(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    direct_topk_routes: bool,
    tc_decode_fused_sum: bool,
    stream_int: int,
) -> None:
    _w4a16_fused_moe_launch_flat(
        a_input=a_input,
        w13_arg=w13_arg,
        w2_arg=w2_arg,
        fc1_out=fc1_out,
        activated=activated,
        fc2_out=fc2_out,
        w13_scale_i32=w13_scale_i32,
        w2_scale_i32=w2_scale_i32,
        w13_global_scale=w13_global_scale,
        w2_global_scale=w2_global_scale,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        activation_amax=None,
        layer_idx=0,
        topk_weights=topk_weights,
        fc1_scratch=fc1_scratch,
        fc2_scratch=fc2_scratch,
        workspace=workspace,
        m=m,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=fast_math,
        sms=sms,
        max_shared_mem=max_shared_mem,
        has_swiglu_limit=has_swiglu_limit,
        swiglu_limit_value=swiglu_limit_value,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        direct_topk_routes=direct_topk_routes,
        tc_decode_fused_sum=tc_decode_fused_sum,
        prefill_fused_sum_fp32=False,
        collect_activation_amax=False,
        stream_int=stream_int,
    )


@_w4a16_fused_moe_launch_op.register_fake
def _w4a16_fused_moe_launch_fake(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    direct_topk_routes: bool,
    tc_decode_fused_sum: bool,
    stream_int: int,
) -> None:
    return None


@torch.library.custom_op(
    "b12x::w4a16_fused_moe_calibrated_launch",
    mutates_args="unknown",
)
def _w4a16_fused_moe_calibrated_launch_op(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    activation_amax: torch.Tensor,
    layer_idx: int,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    stream_int: int,
) -> None:
    _w4a16_fused_moe_launch_flat(
        a_input=a_input,
        w13_arg=w13_arg,
        w2_arg=w2_arg,
        fc1_out=fc1_out,
        activated=activated,
        fc2_out=fc2_out,
        w13_scale_i32=w13_scale_i32,
        w2_scale_i32=w2_scale_i32,
        w13_global_scale=w13_global_scale,
        w2_global_scale=w2_global_scale,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
        topk_weights=topk_weights,
        fc1_scratch=fc1_scratch,
        fc2_scratch=fc2_scratch,
        workspace=workspace,
        m=m,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=fast_math,
        sms=sms,
        max_shared_mem=max_shared_mem,
        has_swiglu_limit=has_swiglu_limit,
        swiglu_limit_value=swiglu_limit_value,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        direct_topk_routes=False,
        tc_decode_fused_sum=False,
        prefill_fused_sum_fp32=False,
        collect_activation_amax=True,
        stream_int=stream_int,
    )


@_w4a16_fused_moe_calibrated_launch_op.register_fake
def _w4a16_fused_moe_calibrated_launch_fake(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    activation_amax: torch.Tensor,
    layer_idx: int,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_clamp_after_silu: bool,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    stream_int: int,
) -> None:
    return None
