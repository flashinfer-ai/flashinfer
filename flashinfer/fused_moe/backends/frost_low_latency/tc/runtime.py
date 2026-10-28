# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared typed output and launch composition for grouped Tensor Core decode."""

import cutlass
from cutlass import cute

from ..quantization import _mul
from .ampere import AmpereGroupedProjection
from .hopper import HopperGroupedProjection
from .warp import WarpGroupedProjection


@cute.kernel
def combine_kernel(
    routes, ids, scores, output, experts: cutlass.Constexpr, narrow: cutlass.Constexpr
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * 256 + thread
    h = cute.size(output, mode=[1])
    k = cute.size(scores, mode=[1])
    if index < cute.size(output):
        token, column = index // h, index % h
        value = cutlass.Float32(0)
        # Match the existing separate FP32 multiply/add and ascending slot order.
        for slot in cutlass.range_constexpr(k):
            # Invalid slots contribute nothing, even with non-finite padding scores.
            expert = ids[token, slot]
            if expert >= 0 and expert < experts:
                score = scores[token, slot].to(cutlass.Float32)
                if cutlass.const_expr(narrow):
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                partial = routes[token * k + slot, column].to(cutlass.Float32)
                value = value + _mul(partial, score)
        output[token, column] = value.to(output.element_type)


@cute.jit
def launch(
    x,
    w1,
    ids,
    mid,
    w2,
    routes,
    scores,
    output,
    group_size: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    stream,
    architecture: cutlass.Constexpr = 90,
    weight_first: cutlass.Constexpr = False,
):
    top_k = cute.size(ids, mode=[1])
    if cutlass.const_expr(architecture == 80):
        first_ampere = AmpereGroupedProjection(
            True, group_size, stages, tile_k, top_k, activation, gated, parameters
        )
        second_ampere = AmpereGroupedProjection(
            False, group_size, stages, tile_k, top_k
        )
        first_ampere(w1, x, ids, mid, stream)
        second_ampere(w2, mid, ids, routes, stream)
    elif cutlass.const_expr(architecture == 90):
        first_hopper = HopperGroupedProjection(
            True, group_size, stages, tile_k, top_k, activation, gated, parameters
        )
        second_hopper = HopperGroupedProjection(
            False, group_size, stages, tile_k, top_k
        )
        first_hopper(w1, x, ids, mid, stream)
        second_hopper(w2, mid, ids, routes, stream)
    else:
        first_warp = WarpGroupedProjection(
            True,
            group_size,
            stages,
            tile_k,
            top_k,
            activation,
            gated,
            parameters,
            weight_first=weight_first,
        )
        second_warp = WarpGroupedProjection(
            False, group_size, stages, tile_k, top_k, weight_first=weight_first
        )
        first_warp(w1, x, ids, mid, stream)
        second_warp(w2, mid, ids, routes, stream)
    if cutlass.const_expr(finalize):
        combine_kernel(
            routes, ids, scores, output, cute.size(w1, mode=[0]), narrow
        ).launch(
            grid=(cute.ceil_div(cute.size(output), 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


@cute.jit
def launch_fp8(
    x,
    w1,
    ids,
    mid,
    w2,
    routes,
    scores,
    output,
    dq1,
    aq2,
    dq2,
    group_size: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    narrow: cutlass.Constexpr,
    finalize: cutlass.Constexpr,
    stream,
    architecture: cutlass.Constexpr = 90,
    weight_first: cutlass.Constexpr = False,
):
    top_k = cute.size(ids, mode=[1])
    if cutlass.const_expr(
        architecture == 90 and x.element_type == cutlass.Float8E4M3FN
    ):
        first_hopper = HopperGroupedProjection(
            True, group_size, stages, tile_k, top_k, activation, gated, parameters
        )
        second_hopper = HopperGroupedProjection(
            False, group_size, stages, tile_k, top_k
        )
        first_hopper(w1, x, ids, mid, stream, dq1, aq2)
        second_hopper(w2, mid, ids, routes, stream, dq2, aq2)
    elif cutlass.const_expr(
        architecture == 120
        or (
            architecture in (90, 103)
            and w1.element_type == cutlass.Float8E4M3FN
            and x.element_type in (cutlass.BFloat16, cutlass.Float16)
        )
    ):
        # This BF16 T4 schedule reduces producer synchronization. FP16 retains
        # cooperative gathering; the same substitution regresses that path.
        serial_w8_t4 = (
            architecture == 103
            and x.element_type == cutlass.BFloat16
            and w1.element_type == cutlass.Float8E4M3FN
            and cute.size(x, mode=[0]) == 4
            and cute.size(x, mode=[1]) == 2048
            and cute.size(mid, mode=[1]) == 512
            and top_k == 10
            and activation == "geglu"
        )
        first_warp = WarpGroupedProjection(
            True,
            group_size,
            stages,
            tile_k,
            top_k,
            activation,
            gated,
            parameters,
            serial_gather=serial_w8_t4,
            weight_first=weight_first,
        )
        second_warp = WarpGroupedProjection(
            False,
            group_size,
            stages,
            tile_k,
            top_k,
            serial_gather=serial_w8_t4,
            weight_first=weight_first,
        )
        first_warp(w1, x, ids, mid, stream, dq1, aq2)
        second_warp(w2, mid, ids, routes, stream, dq2, aq2)
    else:
        raise ValueError(
            "Grouped scaled projection requires SM90/SM120 or SM103 BF16/FP16 W8A16"
        )
    if cutlass.const_expr(finalize):
        combine_kernel(
            routes, ids, scores, output, cute.size(w1, mode=[0]), narrow
        ).launch(
            grid=(cute.ceil_div(cute.size(output), 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )


_TACTICS = (
    tuple(
        (kind, group, stages, tile_k)
        for kind in ("wgmma", "ampere_mma")
        for group in (2, 4, 8)
        for stages in (2, 4, 8)
        for tile_k in (64, 128)
    )
    + tuple(
        ("warp_mma", group, stages, tile_k)
        for group in (2, 4, 8)
        for tile_k in (64, 128)
        for stages in ((2, 4) if tile_k == 128 else (2, 4, 8))
    )
    + (("warp_mma", 4, 2, 256), ("warp_mma", 8, 2, 256))
)


# Preserve the original four-element tactics and their scheduling. Weight-first
# overlap is an additional dense-only choice: cold weights can benefit while
# cache-resident weights favor the original gather-first order.
_WEIGHT_FIRST_TACTICS = tuple(
    ("warp_mma", *choice, True)
    for choice in ((8, 2, 128), (4, 2, 64), (2, 2, 64), (2, 2, 128))
)


# Keep ordinary FP8's existing scheduling choices. Its cache/reuse boundary
# differs from dense, so only the separately measured FP8 tiles are admitted.
_FP8_WEIGHT_FIRST_TACTICS = tuple(("warp_mma", group, 2, 128, True) for group in (2, 4))


def execute(inputs, config, tactic):
    """Bind live buffers to a compiled projection pair."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from ..fma.activation_config import lower_activation
    from ..runtime import _COMPILED, _COMPILE_LOCK
    from .....utils import get_compute_capability

    if (
        tactic not in _TACTICS
        and tactic not in _WEIGHT_FIRST_TACTICS
        and tactic not in _FP8_WEIGHT_FIRST_TACTICS
    ):
        raise ValueError(
            "Grouped decode requires a valid architecture/group/stage/K tactic"
        )
    x, ids, scores, w1, w2, mid, routes, output, narrow = inputs[:9]
    fp8 = x.dtype == torch.float8_e4m3fn
    weight_only = w1.dtype == torch.float8_e4m3fn and x.dtype in (
        torch.bfloat16,
        torch.float16,
    )
    scaled = fp8 or weight_only
    if scaled and (tactic[0] not in ("wgmma", "warp_mma") or len(inputs) != 12):
        raise ValueError("Grouped FP8 requires SM90/SM120 and per-tensor scales")
    capability = get_compute_capability(x.device)
    if weight_only and (
        tactic[0] != "warp_mma" or capability not in ((9, 0), (10, 3), (12, 0))
    ):
        raise ValueError("Grouped W8A16 requires SM90, SM103 or SM120 warp MMA")
    architecture = {"wgmma": 90, "warp_mma": 120, "ampere_mma": 80}[tactic[0]]
    if tactic[0] == "warp_mma" and capability == (10, 3) and weight_only:
        architecture = 103
    if weight_only and tactic[0] == "warp_mma" and capability == (9, 0):
        architecture = 90
    expected = (architecture // 10, architecture % 10)
    if capability != expected:
        raise ValueError(f"Grouped {tactic[0]} decode requires SM{architecture}")
    if (
        x.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
        or w1.dtype != w2.dtype
        or (not weight_only and w1.dtype != x.dtype)
    ):
        raise ValueError(
            "Grouped TC requires matching BF16/FP16/E4M3 or supported E4M3 weights with 16-bit inputs"
        )
    if x.shape[0] > 8 or ids.shape[1] > 32 or mid.shape[1] % 128 or x.shape[1] % 128:
        raise ValueError("Grouped TC decode requires T<=8, top-k<=32 and aligned H/I")
    group_size, stages, tile_k = tactic[1:4]
    weight_first = len(tactic) == 5
    if weight_first:
        from ....api import ReLU2, SiTU, SwiGLU

        if fp8:
            if (
                tactic not in _FP8_WEIGHT_FIRST_TACTICS
                or x.shape[0] not in (4, 8)
                or (x.shape[1], mid.shape[1], ids.shape[1]) != (2048, 512, 10)
                or output.dtype != torch.bfloat16
                or not (
                    config.activation in (SwiGLU(), SiTU())
                    or (x.shape[0] == 8 and config.activation == ReLU2())
                )
                or not config.finalize.do_finalize
            ):
                raise ValueError(
                    "Weight-first FP8 decode requires measured SM120 SwiGLU/SiTU or T8 ReLU2 tiles, "
                    "T4/T8/H2048/I512/top-k10, BF16 output and finalization"
                )
        elif (
            weight_only
            or tactic not in _WEIGHT_FIRST_TACTICS
            or (x.shape[0], x.shape[1], mid.shape[1], ids.shape[1])
            != (8, 2048, 512, 10)
            or output.dtype != x.dtype
            or config.activation != SwiGLU()
            or not config.finalize.do_finalize
        ):
            raise ValueError(
                "Weight-first grouped decode requires dense BF16/FP16 SwiGLU, "
                "T8/H2048/I512/top-k10, matching output dtype and finalization"
            )
    finalize = config.finalize.do_finalize
    if tile_k == 256:
        from ....api import GeGLU

        if not (
            architecture == 90
            and weight_only
            and x.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and x.shape == (group_size, 2048)
            and mid.shape[1] == 512
            and ids.shape[1] == 10
            and config.activation == GeGLU()
            and finalize
        ):
            raise ValueError("K256 W8A16 requires SM90 BF16 GeGLU T4/T8 H2048/I512/K10")
    tensors = (x, w1, ids, mid, w2, routes, scores, output)
    if scaled:
        tensors += tuple(inputs[9:])
    key = (
        "grouped-tc-ffi-w8a16-hopper-k256-v13",
        architecture,
        tuple((v.dtype, tuple(v.shape), tuple(v.stride()), v.device) for v in tensors),
        config.activation,
        tactic,
        narrow,
        finalize,
    )
    with torch.cuda.device(x.device):
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up frost_low_latency before CUDA Graph capture"
                )
            with _COMPILE_LOCK:
                compiled = _COMPILED.get(key)
                if compiled is None:
                    name, parameters = lower_activation(config.activation)
                    compiled = cute.compile(
                        launch_fp8 if scaled else launch,
                        # pack_inputs validates weight alignment. Ampere's
                        # 128-bit cp.async atom needs that fact in its IR.
                        *(
                            from_dlpack(v, assumed_align=16)
                            if architecture == 80 and j in (1, 4)
                            else from_dlpack(v)
                            for j, v in enumerate(tensors)
                        ),
                        group_size,
                        stages,
                        tile_k,
                        name,
                        config.activation.is_gated,
                        parameters,
                        narrow,
                        finalize,
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        architecture,
                        **({"weight_first": True} if weight_first else {}),
                        options=f"--gpu-arch sm_{architecture}{'' if architecture == 80 else 'a'} --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(*tensors)
        if finalize:
            return output
        mapping = torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
        return [routes, scores, mapping]
