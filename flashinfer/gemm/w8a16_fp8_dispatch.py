"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""W8A16 FP8 GEMM, CuTe-DSL kernels (Blackwell and Hopper).

``out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])``; the public
entry with its argument checks is ``flashinfer.gemm.w8a16_fp8_gemm``.  This module
holds the variant selection (``algo`` codes, cost models, tile walk) and the
launch of the Hopper and Blackwell CuTe-DSL kernels under ``kernels/cute_dsl/``.
The selection rules follow those of the CUTLASS C++ mixed-input W8A16 kernels.

Workspace: a split-K call writes fp32 partials there (Blackwell: the slices'
partials stacked as one output of ``splits x M`` rows, reduced by a second
kernel; Hopper: one block per tail (tile, slice) plus one arrival counter per
tail tile, the slices reducing the tile among themselves in the kernel).  The
public query ``mm_bf16_fp8_workspace_bytes`` reports the resolved need.
"""

import functools
import math
from typing import Dict, NamedTuple, Optional, Tuple

import torch

from ..utils import get_compute_capability, get_device_sm_count


class _Sm100Tile(NamedTuple):
    """A Blackwell kernel variant."""

    mma_tiler: Tuple[int, int, int]  # (weight rows, activation rows, k) per MMA tile
    two_sm: bool  # 2SM MMA
    cluster: Tuple[int, int]


class _Sm100Plan(NamedTuple):
    """How a Blackwell call runs: the cached decision of :func:`_sm100_plan`."""

    algo: int  # a key of _SM100_TILES (the variant actually launched)
    splits: int  # K-slices (1 = plain; > 1 = partials + reduce)
    swizzle_size: int
    raster_along_m: bool
    max_active_clusters: int  # persistent grid size, in clusters


class _Sm90Plan(NamedTuple):
    """How a Hopper call runs: the cached decision of :func:`_sm90_plan`."""

    tile_algo: int  # a key of _SM90_TILES (the tile actually launched)
    body_items: int  # tiles run whole; the tiles after them are the split tail
    tail_splits: int  # K-slices of each tail tile (1 = no tail)
    tail_tiles: int  # 0 = no tail, no workspace
    group: int  # activation tiles per L2 group (1 = ungrouped)


# Blackwell tile configurations by algo code.  Code 3 is reserved (a retired
# 256x256 tile with its transformed weight in smem).
_SM100_TILES: Dict[int, _Sm100Tile] = {
    1: _Sm100Tile((256, 128, 128), True, (2, 1)),  # 2sm256x128
    2: _Sm100Tile((128, 128, 128), False, (1, 1)),  # 1sm128x128
    4: _Sm100Tile((256, 192, 128), True, (2, 1)),  # 2sm256x192
    7: _Sm100Tile((256, 160, 128), True, (2, 1)),  # 2sm256x160
}
_SM100_SPLITK = 5
_SM100_STREAMK = 6
# The plain 2SM tiles the cost models choose among, in tie-break order, with the
# measured MMA efficiency of each relative to the 256x128 tile.
_SM100_PLAIN_CANDIDATES: Tuple[Tuple[int, float], ...] = (
    (1, 1.00),
    (7, 0.96),
    (4, 1.02),
)


def _sm100_plain_waves(m: int, n: int, algo: int, clusters: int) -> int:
    """Waves of ``clusters`` the plain tile ``algo`` needs for the output: weight
    tiles of 256 rows times activation tiles of the variant's rows."""
    tile_a = _SM100_TILES[algo].mma_tiler[1]
    return _tiles(_tiles(n, 256) * _tiles(m, tile_a), clusters)


# Largest splits x M the split-K kernel's 32-bit row coordinates represent.
_SM100_MAX_STACKED_ROWS = 2**31 - 1


def _tiles(x: int, t: int) -> int:
    return -(-x // t)


def _sm100_split_k(m: int, n: int, k: int, num_sms: int) -> int:
    """slices for the 256x128 tile (1 = do not split)."""
    if k % 128 != 0:
        return 1
    clusters = num_sms // 2 if num_sms > 1 else 1
    units = _tiles(n, 256) * _tiles(m, 128)
    ktiles = k // 128
    if units * 2 <= clusters:
        best = 1
        for s in range(2, 9):
            if ktiles % s != 0 or ktiles // s < 2 or units * s > clusters:
                continue
            best = s
        return best
    if ktiles >= 100:
        waves1 = _tiles(units, clusters)
        for s in range(2, 9):
            if ktiles % s != 0 or ktiles // s < 50:
                continue
            waves = _tiles(units * s, clusters)
            if waves * 100 <= waves1 * s * 85:
                return s
    return 1


def _sm100_use_stream_k(m: int, n: int, k: int, num_sms: int) -> bool:
    """Whether CUTLASS's heuristic would pick stream-K here (a long K, a grid of two
    or three waves with a short last one), which routes to _sm100_streamk_substitute."""
    if k % 128 != 0 or k // 128 < 64:
        return False
    clusters = num_sms // 2 if num_sms > 1 else 1
    units = _tiles(n, 256) * _tiles(m, 128)
    if units * 2 <= clusters:
        return False
    waves = _tiles(units, clusters)
    tail = units - (waves - 1) * clusters
    return 2 <= waves <= 3 and tail * 3 <= clusters


def sm100_splits_for(m: int, n: int, k: int, num_sms: int) -> int:
    """the split-K slice count actually used: the model's
    choice, or, when the variant is forced on a grid the model would not split,
    the smallest s >= 2 dividing the k-tile count.  0 = K cannot be split."""
    s = _sm100_split_k(m, n, k, num_sms)
    if s >= 2:
        return s
    ktiles = k // 128
    if k % 128 != 0 or ktiles < 2:
        return 0
    for c in range(2, 9):
        if ktiles % c == 0:
            return c
    return 0


def _sm100_streamk_substitute(m: int, n: int, k: int, num_sms: int) -> Tuple[int, int]:
    """(algo, splits) for a problem whose grid would suit stream-K: there is no
    stream-K kernel, so the plain tiles and the split-K slice counts are scored by wave
    quantization (whole waves of k-tiles, the last wave counted whole) plus the
    split-K reduce (its fp32 partial traffic in k-tile-waves), and the cheapest wins.
    When K cannot be sliced only the plain candidates compete."""
    clusters = num_sms // 2 if num_sms > 1 else 1
    ktiles = k // 128

    def plain(algo: int, eff: float) -> float:
        # Whole waves of k-tiles, each tile's work in units of the 128-row tile.
        tile_a = _SM100_TILES[algo].mma_tiler[1]
        return _sm100_plain_waves(m, n, algo, clusters) * ktiles * (tile_a / 128) / eff

    # The first strict winner in candidate order (all costs 0 on K < 128: the
    # 256x128 tile).
    best_algo, best_cost = 1, plain(*_SM100_PLAIN_CANDIDATES[0])
    for algo, eff in _SM100_PLAIN_CANDIDATES[1:]:
        c = plain(algo, eff)
        if c < best_cost:
            best_algo, best_cost = algo, c
    best_splits = 1
    units = _tiles(n, 256) * _tiles(m, 128)
    # One k-tile-wave: the whole grid advancing one k-tile.  The reduce streams
    # (splits + 0.5) * m * n fp32 words; the ratio to one k-tile-wave's MMA work is
    # a constant of the machine (about 1/400 per byte-per-FLOP here).
    reduce_per_split = 4.0 * m * n / (clusters * 2.0 * 128 * 128 * 128 * 2.0) * 400.0
    # Slicing needs K to be whole k-tiles: a slice covers k // splits elements, and a
    # K that is not a multiple of 128 would lose its tail (the plain tiles handle it).
    for s in range(2, 9) if k % 128 == 0 else ():
        if ktiles % s != 0 or ktiles // s < 8:
            continue
        c = _tiles(units * s, clusters) * (ktiles // s) + reduce_per_split * (s + 0.5)
        if c < best_cost:
            best_algo, best_cost, best_splits = _SM100_SPLITK, c, s
    if best_algo == _SM100_SPLITK:
        return 1, best_splits
    return best_algo, 1


def sm100_pick_algo(m: int, n: int, k: int, num_sms: int) -> int:
    """the launcher's variant for a problem."""
    if n <= 128 and m >= 16384:
        return 2
    if _sm100_use_stream_k(m, n, k, num_sms):
        return _SM100_STREAMK
    if _sm100_split_k(m, n, k, num_sms) > 1:
        return _SM100_SPLITK
    clusters = num_sms // 2 if num_sms > 1 else 1

    def cost(algo: int, eff: float) -> float:
        tile_a = _SM100_TILES[algo].mma_tiler[1]
        return _sm100_plain_waves(m, n, algo, clusters) * tile_a / eff

    # 192 rows when it is no costlier than either other width; else 160 rows when
    # strictly cheaper than 128; else 128.
    c128, c160, c192 = (cost(algo, eff) for algo, eff in _SM100_PLAIN_CANDIDATES)
    if c192 <= c128 and c192 <= c160:
        return 4
    if c160 < c128:
        return 7
    return 1


def sm100_pick_swizzle(m: int, n: int, k: int) -> int:
    """L2 tile group in clusters (1 = plain raster)."""
    weight_bytes = n * k
    if weight_bytes < (64 << 20) or m < 4096:
        return 1
    s = ((24 << 20) + 128 * k) // (256 * k)
    return max(2, min(16, s))


def sm100_raster_along_m(m: int, n: int) -> bool:
    """Walk the weight rows (the kernel's M) fast when the fp8 weight (n * k
    bytes) is the smaller operand, else the activation rows (2 * m * k bytes).

    A wave of clusters then shares one tile of the operand walked slowly and
    streams the other operand whole, so the operand walked fast is the one that
    must stay L2-resident across the run: the smaller one.  CUTLASS's tile-count
    heuristic (more activation tiles than weight tiles) ignores the 2x byte
    difference between the operands and walks the activation for N >= 1.5 M;
    at M=8192, N=10240, K=5120 on a B200 that is an L2 hit rate of 59% against
    84%, 1.27 GB of DRAM reads against 0.32 GB and 576 us against 536 us.  Over
    twenty Cosmos layer shapes at a locked clock the byte rule is never slower
    by more than 0.7% and faster by 4-20% wherever M >= 7296."""
    return n < 2 * m


def _compile_pointers(specs: Tuple):
    """CuTe pointer objects for ``cute.compile`` from (dtype, address, alignment)
    triples.  Built only when a kernel is compiled: the compiled function takes the
    raw addresses."""
    from ..cute_dsl.utils import make_ptr

    return tuple(
        make_ptr(dtype, addr, cute_gmem(), assumed_align=align)
        for dtype, addr, align in specs
    )


def _addresses(specs: Tuple) -> Tuple[int, ...]:
    """The raw addresses of (dtype, address, alignment) triples, in order: what a
    compiled kernel takes, read once per call from the same specs that would
    compile it."""
    return tuple(addr for _, addr, _ in specs)


def _raw_stream(device: torch.device):
    """The current Torch stream of ``device`` as a driver stream handle (the public
    torch.cuda.current_stream() path costs ~20 us per call, most of a small GEMM)."""
    import cuda.bindings.driver as cuda

    index = device.index if device.index is not None else torch.cuda.current_device()
    return cuda.CUstream(torch._C._cuda_getCurrentRawStream(index))


# In-process layer of the kernel cache.  Below it sits the shared on-disk cache of
# flashinfer/jit/cute_dsl_core.py: one module directory per compile arch (the
# sm_100a and sm_103a builds of the Blackwell kernel never share an artifact), one
# exported object per specialization, invalidated by the DSL version or a change
# to any file of _kernel_source_files().  A new process reloads the object instead
# of recompiling; FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 turns the disk layer off.
_KERNEL_CACHE: Dict[Tuple, object] = {}
_CUTE_DSL_MODULE = "w8a16_fp8_gemm"


def _kernel_source_files() -> Tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernel cache."""
    from .kernels.cute_dsl import (
        w8a16_fp8_gemm_sm90,
        w8a16_fp8_gemm_sm100,
        w8a16_fp8_gemm_sm100_utils,
        w8a16_fp8_splitk_reduce,
    )

    return (
        __file__,
        w8a16_fp8_gemm_sm90.__file__,
        w8a16_fp8_gemm_sm100.__file__,
        w8a16_fp8_gemm_sm100_utils.__file__,
        w8a16_fp8_splitk_reduce.__file__,
    )


def _build_kernel(kernel_name: str, compile_fn):
    """``compile_fn()`` through the shared CuTe-DSL disk cache."""
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    return build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        kernel_name,
        compile_fn,
        extra_key_files=_kernel_source_files(),
    )


def _dtype_tag(dtype: torch.dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp16"


def _target(device: torch.device) -> Tuple[int, int]:
    """The compile target of ``device`` as its (major, minor) capability: part of
    every in-process cache key, as the arch is of the on-disk module name, so a
    process driving two different GPUs (sm_100a and sm_103a, say) never hands one
    device a kernel built for the other."""
    return get_compute_capability(device)


def _get_sm100_kernel(
    target: Tuple[int, int],
    algo: int,
    act_dtype: torch.dtype,
    has_bias: bool,
    swizzle_size: int,
    raster_along_m: bool,
    max_active_clusters: int,
    pointer_specs: Tuple,
    n: int,
    m: int,
    k: int,
    stream,
    enable_pdl: bool = False,
    partial_output: bool = False,
    splits: int = 1,
):
    mma_tiler, two_sm, cluster = _SM100_TILES[algo]
    # Bytes of weight each transform thread converts per step: a whole stage when K
    # is one k-tile (no pipelining to overlap), 32 otherwise (as the W4A16 kernel).
    transform_fragment_size = 128 if k == mma_tiler[2] else 32
    key = (
        "sm100",
        target,
        algo,
        act_dtype,
        has_bias,
        swizzle_size,
        raster_along_m,
        transform_fragment_size,
        max_active_clusters,
        enable_pdl,
        partial_output,
        splits,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    import cutlass.cute as cute

    from .kernels.cute_dsl.w8a16_fp8_gemm_sm100 import Sm100W8A16Fp8GemmKernel

    kernel = Sm100W8A16Fp8GemmKernel(
        mma_tiler_mnk=mma_tiler,
        use_2cta_instrs=two_sm,
        cluster_shape_mn=cluster,
        has_bias=has_bias,
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
        transform_fragment_size=transform_fragment_size,
        enable_pdl=enable_pdl,
        partial_output=partial_output,
    )
    kernel_name = (
        f"sm100_algo{algo}_{_dtype_tag(act_dtype)}_bias{int(has_bias)}"
        f"_swz{swizzle_size}_rm{int(raster_along_m)}_tf{transform_fragment_size}"
        f"_mac{max_active_clusters}_pdl{int(enable_pdl)}"
        f"_po{int(partial_output)}_sp{splits}"
    )
    compiled = _build_kernel(
        kernel_name,
        lambda: cute.compile(
            kernel.wrapper,
            *_compile_pointers(pointer_specs),
            n,
            m,
            k,
            splits=splits,
            max_active_clusters=max_active_clusters,
            stream=stream,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _get_reduce_kernel(
    target: Tuple[int, int],
    act_dtype: torch.dtype,
    has_bias: bool,
    splits: int,
    pointer_specs: Tuple,
    stream,
):
    """The split-K reduce (sum of slices, scale, bias, cast), compiled and cached
    once per compile target, activation dtype, bias presence and slice count.
    ``pointer_specs`` is not part of that key: every call of one specialization
    must supply the same operand dtypes and alignment guarantees (the caller's
    invariant), while the addresses themselves are passed at run time."""
    key = ("reduce", target, act_dtype, has_bias, splits)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    import cutlass.cute as cute

    from .kernels.cute_dsl.w8a16_fp8_splitk_reduce import W8A16SplitKReduce

    kernel_name = f"reduce_{_dtype_tag(act_dtype)}_bias{int(has_bias)}_sp{splits}"
    compiled = _build_kernel(
        kernel_name,
        lambda: cute.compile(
            W8A16SplitKReduce(has_bias, splits).wrapper,
            *_compile_pointers(pointer_specs),
            1,
            8,
            1,
            stream=stream,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


# Resolved plans by (device / SM count, algo, shape): the cost models run once per
# distinct call signature.
_SM100_PLAN_CACHE: Dict[Tuple, "_Sm100Plan"] = {}
_SM90_PLAN_CACHE: Dict[Tuple, "_Sm90Plan"] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _max_active_clusters(device_index: int, cluster_size: int) -> int:
    """Clusters of ``cluster_size`` CTAs resident at once on device ``device_index``
    (the persistent grid size), asked of the DSL's occupancy probe for that device.
    Per device: the shared helper of ``flashinfer.cute_dsl.utils`` probes device 0 and
    caches by cluster size alone.  Falls back to one CTA per SM when the probe fails
    (a spawned process without a current driver context, say).  The fallback is
    safe because clusters are independent of one another: the CTAs of a 2SM
    cluster do synchronize among themselves, but no cluster waits on another, so
    a grid beyond the resident capacity only has whole clusters run in turn."""
    import cutlass.utils

    try:
        with torch.cuda.device(device_index):
            return cutlass.utils.HardwareInfo(device_index).get_max_active_clusters(
                cluster_size
            )
    except Exception:  # noqa: BLE001 -- the probe is best effort, the bound is safe
        return max(
            get_device_sm_count(torch.device("cuda", device_index)) // cluster_size, 1
        )


def _sm100_plan(algo: int, m: int, n: int, k: int, device: torch.device) -> _Sm100Plan:
    """(algo, splits, swizzle, raster_along_m, max active clusters) of a Blackwell
    call.  A pure function of the shape and the device, cached: the cost model is
    most of the host cost of a call otherwise."""
    device_index = _device_index(device)
    key = (device_index, algo, m, n, k)
    plan = _SM100_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    num_sms = get_device_sm_count(device)
    if algo == 0:
        algo = sm100_pick_algo(m, n, k, num_sms)
    splits = 1
    if algo == _SM100_STREAMK:
        # No stream-K here: the best of what this kernel has for a grid of two or
        # three partly filled waves (a plain tile or split-K).
        algo, splits = _sm100_streamk_substitute(m, n, k, num_sms)
    elif algo == _SM100_SPLITK:
        splits = sm100_splits_for(m, n, k, num_sms)
        if splits < 2:
            raise ValueError(
                f"w8a16_fp8_gemm split-K cannot slice K={k}: K must be a multiple of "
                "128 whose k-tile count has a divisor in 2..8"
            )
        algo = 1
    if splits > 1 and splits * m > _SM100_MAX_STACKED_ROWS:
        # The kernel writes the slices' partials as one output of splits x M rows
        # and derives its row coordinates in 32-bit arithmetic: a stacked extent
        # past that runs the 256x128 tile plain (no workspace), as the query reports.
        splits = 1
    tile = _SM100_TILES[algo]
    plan = _Sm100Plan(
        algo,
        splits,
        sm100_pick_swizzle(m, n, k),
        sm100_raster_along_m(m, n),
        _max_active_clusters(device_index, tile.cluster[0] * tile.cluster[1]),
    )
    _SM100_PLAN_CACHE[key] = plan
    return plan


def w8a16_fp8_gemm_workspace_bytes(
    m: int, n: int, k: int, device: torch.device, algo: int = 0
) -> int:
    """Scratch bytes a call needs (0 = none): the fp32 partials of a Blackwell
    split-K run (``splits * m * n * 4``) or of a Hopper split tail (one
    ``tile_a x tile_w`` fp32 block per tail slice plus the tail tiles' arrival
    counters, :func:`_sm90_workspace_layout`).  A pure function of the shape and
    the device's SM count; ``algo`` as :func:`launch_w8a16_fp8_gemm`."""
    num_sms = get_device_sm_count(device)
    if get_compute_capability(device)[0] == 9:
        plan = _sm90_plan(algo, m, n, k, num_sms)
        tile_w, tile_a = _SM90_TILES[plan.tile_algo].rows
        if plan.tail_tiles == 0:
            return 0
        return _sm90_workspace_layout(
            plan.tail_tiles, plan.tail_splits, tile_a, tile_w
        )[2]
    splits = _sm100_plan(algo, m, n, k, device).splits
    return splits * m * n * 4 if splits > 1 else 0


def launch_w8a16_fp8_gemm(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    algo: int = 0,
    workspace: Optional[torch.Tensor] = None,
) -> None:
    """Launch the W8A16 GEMM (``out = act @ dequant(weight).T + bias``) on
    validated 2-D tensors: ``act`` [M, K] fp16 / bf16, ``weight`` [N, K] e4m3,
    ``weight_scale`` one fp32, ``bias`` [N] or None, ``out`` [M, N], all
    contiguous and on one Hopper or Blackwell device; ``act``, ``weight``,
    ``bias``, ``out`` and ``workspace`` 16-byte aligned, ``weight_scale`` only
    4-byte aligned (a one-element view at any fp32 offset is fine; the public
    :func:`flashinfer.gemm.w8a16_fp8_gemm` checks all of that).  ``algo``: a code of the
    architecture's table (0 = the model's choice).  ``workspace``: a buffer of at
    least :func:`mm_bf16_fp8_workspace_bytes` bytes; None only when that is 0.
    The public op owns allocation; this layer never allocates."""
    # The DSL resolves its compile target from the current CUDA device: run the
    # call under act's device, which need not be the current one.
    with torch.cuda.device(act.device):
        if _target(act.device)[0] == 9:
            _w8a16_fp8_gemm_sm90(act, weight, weight_scale, bias, out, algo, workspace)
        else:
            _w8a16_fp8_gemm_sm100(act, weight, weight_scale, bias, out, algo, workspace)


def _require_workspace(workspace: Optional[torch.Tensor], nbytes: int) -> torch.Tensor:
    """The caller's workspace for a plan that needs ``nbytes``: the launcher's
    invariant is that the public op (or a direct caller) allocated it already."""
    have = 0 if workspace is None else workspace.numel() * workspace.element_size()
    if have < nbytes:
        raise ValueError(
            f"launch_w8a16_fp8_gemm needs a workspace of {nbytes} bytes for this "
            f"plan, got {have} (see mm_bf16_fp8_workspace_bytes)"
        )
    return workspace


def _w8a16_fp8_gemm_sm100(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    algo: int,
    workspace: Optional[torch.Tensor],
) -> None:
    """The Blackwell path of ``launch_w8a16_fp8_gemm`` (inputs already validated,
    act's device current)."""
    import cutlass

    m, k = act.shape
    n = weight.shape[0]
    target = _target(act.device)
    plan = _sm100_plan(algo, m, n, k, act.device)
    algo, splits, swizzle_size, raster_along_m, max_active_clusters = plan
    # Split-K: the 256x128 tile over `splits` K-slices writing fp32 partials, then
    # the reduce.  The GEMM's `out` is then the partials buffer and the kernel does
    # no scale / bias.
    partials = None
    if splits > 1:
        workspace = _require_workspace(workspace, splits * m * n * 4)
        partials = _scratch(workspace, (splits, m, n), torch.float32)
    k_slice = k // splits
    has_bias = bias is not None
    act_ctype = cutlass.BFloat16 if act.dtype == torch.bfloat16 else cutlass.Float16
    # Without a bias the stand-in pointer is the scale, which the contract only
    # promises 4-byte aligned: every spec below declares no more than that (make_ptr
    # asserts the declared alignment when the kernel compiles).
    bias_tensor = bias if has_bias else weight_scale  # any valid pointer when unused
    bias_ctype = act_ctype if has_bias else cutlass.Float32
    bias_align = 16 if has_bias else 4
    gemm_out = partials if partials is not None else out
    gemm_out_ctype = cutlass.Float32 if partials is not None else act_ctype
    pointer_specs = (
        (cutlass.Float8E4M3FN, weight.data_ptr(), 16),
        (act_ctype, act.data_ptr(), 16),
        (cutlass.Float32, weight_scale.data_ptr(), 4),
        (bias_ctype, bias_tensor.data_ptr(), bias_align),
        (gemm_out_ctype, gemm_out.data_ptr(), 16),
    )
    stream = _raw_stream(act.device)
    compiled = _get_sm100_kernel(
        target,
        algo,
        act.dtype,
        has_bias,
        swizzle_size,
        raster_along_m,
        max_active_clusters,
        pointer_specs,
        n,
        m,
        k_slice,
        stream,
        partial_output=partials is not None,
        splits=splits,
    )
    compiled(*_addresses(pointer_specs), n, m, k_slice, stream)
    if partials is not None:
        reduce_specs = (
            (cutlass.Float32, partials.data_ptr(), 16),
            (cutlass.Float32, weight_scale.data_ptr(), 4),
            (bias_ctype, bias_tensor.data_ptr(), bias_align),
            (act_ctype, out.data_ptr(), 16),
        )
        reduce = _get_reduce_kernel(
            target, act.dtype, has_bias, splits, reduce_specs, stream
        )
        reduce(*_addresses(reduce_specs), m, n, m * n // 8, stream)


# ======================================================================= Hopper
class _Sm90Tile(NamedTuple):
    """A Hopper kernel tile and its cost-model entry."""

    algo: int
    weight_rows: int
    act_rows: int
    pingpong: bool
    # ns per unit of K of one tile alone on an SM; None = not a candidate of the
    # cost model (selectable by name only).
    ns_per_k: Optional[float]

    @property
    def rows(self) -> Tuple[int, int]:
        return self.weight_rows, self.act_rows


# Tile configurations by algo code.
# Codes 1 and 4 are reserved (a retired hand-written kernel and a retired 128x160
# tile).
_SM90_TILES: Dict[int, _Sm90Tile] = {
    2: _Sm90Tile(2, 128, 256, False, 11.37),
    3: _Sm90Tile(3, 128, 192, False, 9.00),
    5: _Sm90Tile(5, 128, 128, False, 7.45),
    6: _Sm90Tile(6, 64, 256, True, 6.15),
    9: _Sm90Tile(9, 64, 128, True, 4.50),
}
# The cost model's candidates, in tie-break order.
_SM90_TILE_CANDIDATES: Tuple[_Sm90Tile, ...] = tuple(
    _SM90_TILES[a] for a in (2, 3, 5, 6, 9)
)
# The policy codes map to the tile they run.  Stream-K codes run the whole waves of
# that tile plain and cut only the last partial wave into K-slices (sm90_tail_plan;
# CUTLASS's stream-K scheduler itself is not ported); split-K codes cut every tile.
_SM90_STREAMK_TILE: Dict[int, int] = {7: 2, 8: 3}
_SM90_SPLITK_TILE: Dict[int, int] = {10: 5, 11: 2, 12: 9}
_SM90_TILE_K = 64
_SM90_MAX_SPLITS = 8
_SM90_MIN_KTILES_PER_SPLIT = 8
_SM90_MAX_PARTIALS_BYTES = 64 << 20


def _sm90_tile_of(algo: int) -> _Sm90Tile:
    """The cost-model tile a code runs on (policies map to their base tile); an
    unknown code is costed as the first candidate."""
    base = _SM90_STREAMK_TILE.get(algo, _SM90_SPLITK_TILE.get(algo, algo))
    t = _SM90_TILES.get(base)
    if t is None or t.ns_per_k is None:
        return _SM90_TILE_CANDIDATES[0]
    return t


def _sm90_is_legal_split(k: int, s: int) -> bool:
    if s < 2 or s > _SM90_MAX_SPLITS or k % _SM90_TILE_K != 0:
        return False
    ktiles = k // _SM90_TILE_K
    return ktiles % s == 0 and ktiles // s >= _SM90_MIN_KTILES_PER_SPLIT


def _sm90_tile_us(t: _Sm90Tile, k_len: float) -> float:
    epi = 0.0 if t.pingpong else 2.0 * t.act_rows / 256
    return t.ns_per_k * 1e-3 * k_len + epi


def _sm90_tiles_of(t: _Sm90Tile, m: int, n: int) -> float:
    return float(_tiles(n, t.weight_rows)) * float(_tiles(m, t.act_rows))


def _sm90_split_us(t, m: int, n: int, k: int, num_sms: int, s: int) -> float:
    if 4.0 * m * n * s > _SM90_MAX_PARTIALS_BYTES:
        return 1e30
    return (
        math.ceil(_sm90_tiles_of(t, m, n) * s / max(num_sms, 1))
        * _sm90_tile_us(t, k / s)
        + 2.0
        + 2e-6 * m * n * s
    )


def _sm90_best_split(t, m: int, n: int, k: int, num_sms: int) -> tuple[int, float]:
    """The cheapest legal split count of tile ``t`` and its modelled cost:
    ``(1, 1e30)`` when no legal split has a usable cost."""
    best, best_us = 1, 1e30
    for s in range(2, _SM90_MAX_SPLITS + 1):
        if not _sm90_is_legal_split(k, s):
            continue
        us = _sm90_split_us(t, m, n, k, num_sms, s)
        if us < best_us:
            best_us, best = us, s
    return best, best_us


def sm90_splits_for(algo: int, m: int, n: int, k: int, num_sms: int) -> int:
    """the model's split, or the smallest legal one when the
    variant is forced; 1 = K cannot be sliced."""
    t = _sm90_tile_of(algo)
    s, _ = _sm90_best_split(t, m, n, k, num_sms)
    if s >= 2:
        return s
    for c in range(2, _SM90_MAX_SPLITS + 1):
        if _sm90_is_legal_split(k, c):
            return c
    return 1


def _sm90_plain_us(t, m: int, n: int, k: int, num_sms: int) -> float:
    return math.ceil(_sm90_tiles_of(t, m, n) / max(num_sms, 1)) * _sm90_tile_us(t, k)


def _sm90_streamk_us(t, m: int, n: int, k: int, num_sms: int) -> float:
    sms = float(max(num_sms, 1))
    ktiles = k // _SM90_TILE_K
    tiles = _sm90_tiles_of(t, m, n)
    tile_us = _sm90_tile_us(t, k)
    full_waves, total_waves = math.floor(tiles / sms), math.ceil(tiles / sms)
    tail_tiles = tiles - full_waves * sms
    if (
        full_waves == total_waves
        or ktiles <= 8
        or (full_waves >= 1 and 2 * tail_tiles >= sms)
    ):
        return total_waves * tile_us
    dp_waves = full_waves - 1 if full_waves > 1 else 0
    sk_tiles = tiles - dp_waves * sms
    units = max(1.0, min(sms, math.floor(sk_tiles * ktiles / 8.0)))
    return (
        dp_waves * tile_us
        + sk_tiles * ktiles / units * (tile_us / ktiles) * 1.1
        + 8.0
        + 2.0 * units / sk_tiles
    )


def _sm90_pick_by_cost(m: int, n: int, k: int, num_sms: int) -> int:
    best, best_us = 2, 1e30

    def consider(a: int, us: float) -> None:
        nonlocal best, best_us
        if us < best_us:
            best_us, best = us, a

    for t in _SM90_TILE_CANDIDATES:
        consider(t.algo, _sm90_plain_us(t, m, n, k, num_sms))
    for a in (7, 8):
        consider(a, _sm90_streamk_us(_sm90_tile_of(a), m, n, k, num_sms))
    for a in (10, 11, 12):
        consider(a, _sm90_best_split(_sm90_tile_of(a), m, n, k, num_sms)[1])
    return best


def sm90_pick_algo(m: int, n: int, k: int, num_sms: int) -> int:
    """the cost model up to M = 8192, wave-count rules above."""
    units = _tiles(n, 128) * _tiles(m, 256)
    waves = units / max(num_sms, 1)
    frac = waves - math.floor(waves)
    if m <= 8192:
        return _sm90_pick_by_cost(m, n, k, num_sms)
    if n <= 512:
        if waves < 0.75:
            return 5
        if m >= 16384:
            return 3
        return 2
    if k <= 1024 and m >= 16384:
        return 6
    if waves <= 1.5:
        if k >= 4096 and waves >= 0.7:
            return 7
        if n <= 2048 or k <= 3200:
            return 6
        return 2
    if frac >= 0.8:
        return 2
    if frac < 0.15 and k >= 2048 and waves <= 16.0:
        return 7
    if waves <= 3.0:
        return 6
    return 2


def sm90_pick_group(m: int, n: int, k: int, tile_w: int) -> int:
    """activation tiles per L2 group (1 = ungrouped)."""
    if n * k < 40 * 1000 * 1000 or m < 16384:
        return 1
    return 4 if _tiles(n, tile_w) >= 64 else 2


def _get_sm90_kernel(
    target: Tuple[int, int],
    tile_algo: int,
    act_dtype: torch.dtype,
    has_bias: bool,
    group: int,
    num_persistent: int,
    pointer_specs: Tuple,
    n: int,
    m: int,
    k: int,
    tail_blocks: int,
    stream,
    has_tail: bool = False,
):
    key = (
        "sm90",
        target,
        tile_algo,
        act_dtype,
        has_bias,
        group,
        num_persistent,
        has_tail,
        _SM90_TAIL_WAIT_NS,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    import cutlass.cute as cute

    from .kernels.cute_dsl.w8a16_fp8_gemm_sm90 import Sm90W8A16Fp8GemmKernel

    from .kernels.cute_dsl.w8a16_fp8_gemm_sm90 import TAIL_WAIT_NS

    kernel = Sm90W8A16Fp8GemmKernel(
        _SM90_TILES[tile_algo].rows,
        has_bias=has_bias,
        group_size=group,
        has_tail=has_tail,
        tail_wait_ns=(
            TAIL_WAIT_NS if _SM90_TAIL_WAIT_NS is None else _SM90_TAIL_WAIT_NS
        ),
    )
    kernel_name = (
        f"sm90_algo{tile_algo}_{_dtype_tag(act_dtype)}_bias{int(has_bias)}"
        f"_grp{group}_np{num_persistent}_tail{int(has_tail)}"
        + ("" if _SM90_TAIL_WAIT_NS is None else f"_wait{_SM90_TAIL_WAIT_NS}")
    )
    compiled = _build_kernel(
        kernel_name,
        lambda: cute.compile(
            kernel.wrapper,
            *_compile_pointers(pointer_specs),
            n,
            m,
            k,
            tail_blocks,
            1,  # body_items: dynamic
            1,  # tail_splits: dynamic
            num_persistent=num_persistent,
            stream=stream,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


_SM90_TAIL_WORKSPACE_CAP = 64 << 20
_SM90_TAIL_MAX_SPLITS = 16
# Test hook: how long a tail slice waits for the others before withdrawing from the
# fix-up, in ns (None: the kernel's default, TAIL_WAIT_NS).  Part of the kernel key.
_SM90_TAIL_WAIT_NS: Optional[int] = None


def _sm90_workspace_layout(
    tail_tiles: int, tail_splits: int, tile_a: int, tile_w: int
) -> Tuple[int, int, int]:
    """(partials bytes, counters offset, total bytes) of a Hopper call with a tail:
    the fp32 partial blocks first, then one 64-bit arrival counter per tail tile at
    the next 16-byte boundary.  The counters belong to the call's workspace, not to
    the device: two calls in flight on different streams (or two graphs replayed
    together) with their own buffers never count on each other's tiles."""
    partial_bytes = tail_tiles * tail_splits * tile_a * tile_w * 4
    counters_offset = (partial_bytes + 15) // 16 * 16
    return partial_bytes, counters_offset, counters_offset + tail_tiles * 8


def sm90_tail_plan(
    algo: int, m: int, n: int, k: int, num_sms: int
) -> Tuple[int, int, int]:
    """The work decomposition of a Hopper variant: (tile algo, body tiles, tail
    splits).  Plain tiles: every tile whole.  Split-K codes: no body, every tile cut
    into ``sm90_splits_for`` slices.  Stream-K codes: the whole waves plain and the
    last partial wave cut into the slice count that best fills the SMs (fewest rounds
    of the shortest slices under the workspace cap), or plain when CUTLASS's own
    stream-K would run data-parallel (no tail, or a tail of at least half a wave)."""
    tile_algo = _SM90_STREAMK_TILE.get(algo, _SM90_SPLITK_TILE.get(algo, algo))
    tile_w, tile_a = _SM90_TILES[tile_algo].rows
    tiles = _tiles(n, tile_w) * _tiles(m, tile_a)
    k_tiles = k // _SM90_TILE_K
    if algo in _SM90_SPLITK_TILE:
        return tile_algo, 0, sm90_splits_for(algo, m, n, k, num_sms)
    if algo not in _SM90_STREAMK_TILE:
        return tile_algo, tiles, 1
    full = tiles // num_sms
    tail = tiles - full * num_sms
    if tail == 0 or 2 * tail >= num_sms or k_tiles < 4:
        return tile_algo, tiles, 1
    # The most slices that still run in one round (the tail is sized to the idle
    # SMs of the last wave; the fix-up itself needs no co-residency), at least two
    # k-tiles each, under the workspace cap.
    block_bytes = tile_w * tile_a * 4
    best_s = 1
    for s in range(2, _SM90_TAIL_MAX_SPLITS + 1):
        if tail * s > num_sms or k_tiles // s < 2:
            break
        if tail * s * block_bytes > _SM90_TAIL_WORKSPACE_CAP:
            break
        best_s = s
    if best_s == 1:
        return tile_algo, tiles, 1
    return tile_algo, full * num_sms, best_s


def _sm90_plan(algo: int, m: int, n: int, k: int, num_sms: int) -> _Sm90Plan:
    """(tile algo, body items, tail splits, tail tiles, L2 group) of a Hopper call:
    the variant choice and the tail plan, pure functions of the shape, cached."""
    key = (num_sms, algo, m, n, k)
    plan = _SM90_PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    if algo == 0:
        algo = sm90_pick_algo(m, n, k, num_sms)
    if algo == 1 or algo > 12:
        raise ValueError(f"w8a16_fp8_gemm: unknown Hopper algo code {algo}")
    if algo in _SM90_SPLITK_TILE and k % _SM90_TILE_K != 0:
        raise ValueError("w8a16_fp8_gemm split-K: K must be a multiple of 64")
    tile_algo, body_items, tail_splits = sm90_tail_plan(algo, m, n, k, num_sms)
    if algo in _SM90_SPLITK_TILE and tail_splits < 2:
        raise ValueError(
            f"w8a16_fp8_gemm split-K cannot slice K={k}: K must be a multiple of 64 "
            "with at least 8 k-tiles per slice"
        )
    tile_w, tile_a = _SM90_TILES[tile_algo].rows
    tiles = _tiles(n, tile_w) * _tiles(m, tile_a)
    tail_tiles = tiles - body_items
    if tail_splits < 2 or tail_tiles * tail_splits > num_sms:
        body_items, tail_splits, tail_tiles = tiles, 1, 0
    plan = _Sm90Plan(
        tile_algo,
        body_items,
        tail_splits,
        tail_tiles,
        sm90_pick_group(m, n, k, tile_w),
    )
    _SM90_PLAN_CACHE[key] = plan
    return plan


def _w8a16_fp8_gemm_sm90(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    algo: int,
    workspace: Optional[torch.Tensor],
) -> None:
    """The Hopper path of ``launch_w8a16_fp8_gemm`` (inputs already validated,
    act's device current)."""
    import cutlass

    m, k = act.shape
    n = weight.shape[0]
    target = _target(act.device)
    num_sms = get_device_sm_count(act.device)
    tile_algo, body_items, tail_splits, tail_tiles, group = _sm90_plan(
        algo, m, n, k, num_sms
    )
    tile_w, tile_a = _SM90_TILES[tile_algo].rows
    has_tail = tail_tiles > 0
    tail_blocks = tail_tiles * tail_splits
    if has_tail:
        _partial_bytes, counters_offset, total_bytes = _sm90_workspace_layout(
            tail_tiles, tail_splits, tile_a, tile_w
        )
        workspace = _require_workspace(workspace, total_bytes)
        partials = _scratch(workspace, (tail_blocks, tile_a, tile_w), torch.float32)
        counters = _scratch(workspace, (tail_tiles,), torch.int64, counters_offset)
        # The kernel's precondition: the arrival counters are zero at launch.  One
        # memset on the call's stream, so the counters of a call are ordered with its
        # kernel and a caller's reused buffer holds nothing the kernel trusts.
        counters.zero_()
    else:
        # Unused by the kernel, which still builds its workspace TMA descriptor and
        # declares 16- and 8-byte alignment: ``out`` is always 16-byte aligned (a
        # scalar ``weight_scale`` view need only be 4-byte aligned).
        partials = counters = out
    has_bias = bias is not None
    act_ctype = cutlass.BFloat16 if act.dtype == torch.bfloat16 else cutlass.Float16
    # Stand-in bias pointer and its declared alignment: see the Blackwell launch.
    bias_tensor = bias if has_bias else weight_scale  # any valid pointer when unused
    bias_ctype = act_ctype if has_bias else cutlass.Float32
    bias_align = 16 if has_bias else 4
    pointer_specs = (
        (cutlass.Float8E4M3FN, weight.data_ptr(), 16),
        (act_ctype, act.data_ptr(), 16),
        (cutlass.Float32, weight_scale.data_ptr(), 4),
        (bias_ctype, bias_tensor.data_ptr(), bias_align),
        (act_ctype, out.data_ptr(), 16),
        (cutlass.Float32, partials.data_ptr(), 16),
        (cutlass.Int64, counters.data_ptr(), 8),
    )
    stream = _raw_stream(act.device)
    compiled = _get_sm90_kernel(
        target,
        tile_algo,
        act.dtype,
        has_bias,
        group,
        num_sms,
        pointer_specs,
        n,
        m,
        k,
        max(tail_blocks, 1),
        stream,
        has_tail=has_tail,
    )
    compiled(
        *_addresses(pointer_specs),
        n,
        m,
        k,
        max(tail_blocks, 1),
        body_items,
        tail_splits,
        stream,
    )


def _scratch(
    workspace: torch.Tensor, shape: Tuple, dtype: torch.dtype, offset: int = 0
) -> torch.Tensor:
    """A ``shape`` tensor of ``dtype`` carved from ``workspace`` at byte ``offset``
    (the caller sized the buffer with :func:`mm_bf16_fp8_workspace_bytes`)."""
    numel = 1
    for d in shape:
        numel *= d
    nbytes = numel * dtype.itemsize
    flat = workspace.view(torch.uint8).flatten()
    return flat[offset : offset + nbytes].view(dtype).view(shape)


def cute_gmem():
    import cutlass.cute as cute

    return cute.AddressSpace.gmem
