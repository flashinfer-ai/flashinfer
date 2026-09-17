"""SM120 CUTLASS backend for NVFP4 SVDQuant GEMM and fused linear layers.

The fused linear path combines CUDA smooth quantization and LoRA-down
projection with a CUTLASS GEMM that applies the LoRA-up correction and bias.
Select it explicitly with ``backend="cutlass-sm120"``; ``"auto"`` keeps the
upstream backend selection.
"""

import functools
import weakref
from collections import OrderedDict
from typing import Final, List, Optional

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..autotuner.initializers import autotuner_initializer_empty
from ..jit.gemm.svdquant_sm120 import gen_gemm_sm120_module_cutlass_nvfp4_svdquant
from ..utils import (
    _get_cache_buf,
    device_support_pdl,
    get_compute_capability,
)
from . import svdquant_sm120_routes as _sm120_routes
from .gemm_svdquant import (
    DEFAULT_WORKSPACE_SIZE,
    _NVFP4_SVDQUANT_GEMM_TUNING_CONFIG,
    SVDQUANT_LORA_RANK_GRANULARITY,
    _swizzled_sf_size,
    get_nvfp4_svdquant_module,
)


_SM120_FUSED_LINEAR_MK = _sm120_routes.SM120_FUSED_LINEAR_MK
_sm120_producer_variants = _sm120_routes.sm120_producer_variants
_sm120_decode_producer_variant = _sm120_routes.sm120_decode_producer_variant
_sm120_variant_packs_l2t = _sm120_routes.sm120_variant_packs_l2t
_sm120_pack_tactic = _sm120_routes.sm120_pack_tactic
_sm120_unpack_tactic = _sm120_routes.sm120_unpack_tactic
_SM120_LINEAR_ROUTE_ABI_VERSION = _sm120_routes.SM120_LINEAR_ROUTE_ABI_VERSION
_SM120_LINEAR_ROUTE_V6_MKR = _sm120_routes.SM120_LINEAR_ROUTE_V6_MKR
_SM120_LINEAR_ROUTE_V7_MKR = _sm120_routes.SM120_LINEAR_ROUTE_V7_MKR
_SM120_LINEAR_ROUTE_V8_MKR = _sm120_routes.SM120_LINEAR_ROUTE_V8_MKR
_SM120_LINEAR_ROUTE_V9_MKR = _sm120_routes.SM120_LINEAR_ROUTE_V9_MKR
_SM120_TACTIC_ABI_VERSION = _sm120_routes.SM120_TACTIC_ABI_VERSION
_sm120_linear_route_abi_version = _sm120_routes.sm120_linear_route_abi_version
_sm120_tactic_abi_version = _sm120_routes.sm120_tactic_abi_version


@functools.cache
def get_nvfp4_svdquant_sm120_module(lora_rank: int = SVDQUANT_LORA_RANK_GRANULARITY):
    """JIT-build and load the SM120 CUTLASS NVFP4 SVDQuant module.

    One module serves one LoRA rank, the way upstream's cute-dsl path compiles a
    specialization per rank. ``lora_rank=32`` is the production build and passes
    no extra flag, so it stays byte-identical to a build that never knew about
    this argument -- and keeps its tuned cache.

    A rank a given tile cannot stage still compiles that tile at rank 32; the
    collective's ``can_implement`` then refuses it, so the autotuner simply sees
    a smaller candidate plane rather than a build failure. Today only the K256
    tiles (11 of 31 configs) stage rank 64, because the byte-exact path overlays
    the rank tile on a residual stage worth TileK/4 bf16 columns.
    """
    return gen_gemm_sm120_module_cutlass_nvfp4_svdquant(
        lora_rank=lora_rank
    ).build_and_load()


def _svdquant_backend_for_capability(major: int, minor: int) -> str:
    """Map a compute capability to the SVDQuant kernel backend key.

    SM100 (10.0) and SM103 (10.3) share the tcgen05/TMEM module; SM120 (12.0) uses
    its own register-accumulator module with a separate JIT cache and tactic space.
    Every other capability -- including other 10.x and 12.x parts -- is rejected
    explicitly: none of them has a validated SVDQuant backend.
    """
    if (major, minor) in ((10, 0), (10, 3)):
        return "sm100"
    if (major, minor) == (12, 0):
        return "sm120"
    raise ValueError(
        f"NVFP4 SVDQuant is not supported on compute capability {major}{minor}; "
        "supported: 100, 103 (SM100 backend) and 120 (SM120 backend)"
    )


_SM120_M537_PACKED_L2T_K = frozenset({3072, 5120, 5376, 7168})


_SM120_M537_PACKED_L2T_CACHE_CAPACITY = 64


_SM120_M537_PACKED_L2T_CACHE: OrderedDict[
    int, tuple[weakref.ReferenceType, int, torch.Tensor]
] = OrderedDict()


class _SM120M537L2TPackingError(ValueError):
    """Report an unsupported matrix shape at the shared L2T packing boundary."""

    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        super().__init__(
            "SM120 L2T packing requires rank 32 and K in "
            f"{sorted(_SM120_M537_PACKED_L2T_K)}, got {shape}"
        )


def _pack_sm120_m537_l2t(l2t_smoothed: torch.Tensor) -> torch.Tensor:
    """Pack L2T for one load per lane in M537 and packed small-M producers."""
    shape = tuple(l2t_smoothed.shape)
    if len(shape) != 2 or shape[0] not in _SM120_M537_PACKED_L2T_K or shape[1] != 32:
        raise _SM120M537L2TPackingError(shape)
    if (
        l2t_smoothed.is_cuda
        and l2t_smoothed.dtype == torch.bfloat16
        and l2t_smoothed.is_contiguous()
    ):
        packed = torch.empty_like(l2t_smoothed)
        get_nvfp4_svdquant_sm120_module().nvfp4_svdquant_pack_l2t_sm120(
            l2t_smoothed, packed
        )
        return packed
    k = shape[0]
    return (
        l2t_smoothed.reshape(k // 16, 2, 4, 2, 2, 2, 8)
        .permute(0, 4, 5, 6, 2, 1, 3)
        .contiguous()
        .reshape(shape)
    )


def _cached_sm120_m537_l2t(
    l2t_smoothed: torch.Tensor,
) -> torch.Tensor:
    """Cache a shared L2T pack and invalidate it on mutation."""
    # Capture the pack so graph replay observes weight updates without Python
    # cache lookups. Keep graph-owned buffers out of the eager cache as well.
    # Inference tensors have no version counter and must also be repacked.
    if l2t_smoothed.is_inference() or (
        l2t_smoothed.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        return _pack_sm120_m537_l2t(l2t_smoothed)

    source_id = id(l2t_smoothed)
    source_version = int(l2t_smoothed._version)
    cached = _SM120_M537_PACKED_L2T_CACHE.get(source_id)
    if (
        cached is not None
        and cached[0]() is l2t_smoothed
        and cached[1] == source_version
    ):
        _SM120_M537_PACKED_L2T_CACHE.move_to_end(source_id)
        return cached[2]

    packed = _pack_sm120_m537_l2t(l2t_smoothed)

    def discard(dead_ref: weakref.ReferenceType, *, key: int = source_id) -> None:
        current = _SM120_M537_PACKED_L2T_CACHE.get(key)
        if current is not None and current[0] is dead_ref:
            _SM120_M537_PACKED_L2T_CACHE.pop(key, None)

    source_ref = weakref.ref(l2t_smoothed, discard)
    _SM120_M537_PACKED_L2T_CACHE[source_id] = (
        source_ref,
        source_version,
        packed,
    )
    _SM120_M537_PACKED_L2T_CACHE.move_to_end(source_id)
    while len(_SM120_M537_PACKED_L2T_CACHE) > _SM120_M537_PACKED_L2T_CACHE_CAPACITY:
        _SM120_M537_PACKED_L2T_CACHE.popitem(last=False)
    return packed


def _sm120_fused_linear_supported(m: int, k: int, rank: int) -> bool:
    """Return whether a shape may offer the fused K12 route to the autotuner.

    Computed, not looked up. A shape can offer the fused route when a producer
    can launch it -- when its geometry ladder is non-empty -- and the fused
    producer stages the module's compile-time rank. Rank 64 uses the row-major
    producer families; packed-L2T families remain rank 32 only.

    This replaced a 51-entry table whose contents were exactly the (M, K)
    projection of the 71 benchmark shapes, so on everything ever measured the
    two agree. Off that matrix they do not, and the fused route was checked
    there before this changed: on 2048x3072x3072, 3000x3072x3072 and
    1024x5120x5120 the fused K12+K3 launch is bit-identical to separate
    launches of the same producer and tactic, all four outputs, with an
    in-matrix shape carried alongside as the control.

    Offering is not choosing. A shape nobody benchmarked now reaches the tuner
    as a candidate instead of being excluded by a list, and the tuner drops it
    if it loses.
    """
    if rank not in (32, 64):
        return False
    return bool(_sm120_producer_variants(m, k, rank))


def _svdquant_op_name(
    backend_key: str, lora_rank: int = SVDQUANT_LORA_RANK_GRANULARITY
) -> str:
    """Autotune-cache namespace per backend (and per tactic ABI on SM120).

    SM100 keeps the original name so existing tuning caches stay valid; other
    backends are suffixed so tactic ids can never cross architectures or
    tactic-table versions. The benchmark's tactic read-back must derive the
    name the same way.
    """
    if backend_key == "sm100":
        return "nvfp4_svdquant_gemm"
    if backend_key == "sm120":
        # Tactic ids are stable across ranks by construction, but which of them
        # can_implement admits is not, so a record must not cross ranks. Rank 32
        # keeps the historical name and its existing cache.
        rank_tag = (
            "" if lora_rank == SVDQUANT_LORA_RANK_GRANULARITY else f"_r{lora_rank}"
        )
        return f"nvfp4_svdquant_gemm_sm120_v{_SM120_TACTIC_ABI_VERSION}{rank_tag}"
    return f"nvfp4_svdquant_gemm_{backend_key}"


def _get_nvfp4_svdquant_module_for_device(
    device: torch.device, lora_rank: int = SVDQUANT_LORA_RANK_GRANULARITY
):
    backend = _svdquant_backend_for_capability(*get_compute_capability(device))
    if backend == "sm120":
        return get_nvfp4_svdquant_sm120_module(lora_rank)
    return get_nvfp4_svdquant_module()


_SM120_WORKSPACE_BUDGET_FRACTION: Final = 0.25


def _sm120_workspace_budget_bytes(device_index: Optional[int]) -> int:
    """Largest per-tactic workspace this card can back for one shape, right now.

    The autotuner provisions one buffer per shape covering every tactic it may
    select, so a single Split-K row asking for O(splits*m*n*4B) sets the size
    for the whole shape.

    Against FREE memory, not total, and not cached. Sizing this as a fraction
    of total was measured wrong on 82752x28672x5376: the widest request there
    is 8.84 GiB, comfortably under a quarter of a 71 GiB card, but by the time
    the buffer is provisioned the problem's own operands hold 65 GiB and 5.74
    GiB is left. A total-memory budget admits the tactic and the allocation
    fails -- which is the OOM the shape-keyed exposure filter used to prevent,
    reintroduced by pricing the wrong quantity.

    Reading free memory makes the admitted set depend on what else is resident.
    That is the honest dependency: whether a buffer can be backed is a fact
    about the moment it is allocated, not about the part number.
    """
    if device_index is None:
        return 0
    free, _total = torch.cuda.mem_get_info(device_index)
    return int(free * _SM120_WORKSPACE_BUDGET_FRACTION)


_SM120_WORKSPACE_BUDGET_CACHE: dict = {}


def _sm120_shape_workspace_budget(device: Optional[torch.device], key) -> int:
    """The budget this shape was sized against, decided once and then held.

    Sizing and selection are separate calls, and free memory moves between
    them. If selection re-read it, the autotuner could admit a tactic whose
    workspace the already-provisioned buffer cannot hold, which the C++
    ICHECK_GE then refuses at launch. Deciding once per shape keeps the two
    halves talking about the same number.
    """
    if device is None:
        return 0
    cached = _SM120_WORKSPACE_BUDGET_CACHE.get(key)
    if cached is None:
        cached = _sm120_workspace_budget_bytes(device.index)
        _SM120_WORKSPACE_BUDGET_CACHE[key] = cached
    return cached


def _nvfp4_svdquant_valid_tactics(
    module,
    m: int,
    n: int,
    k: int,
    rank: int,
    device: Optional[torch.device] = None,
) -> List[int]:
    """Tactics this shape can run, bounded by what the card can back.

    `can_implement` is the correctness boundary and the workspace budget is a
    legality boundary on top of it. There is deliberately no profitability
    filter here: which of these candidates is fastest is a measurement, and the
    autotuner is the thing that measures. Passing ``device=None`` asks for the
    unbounded set, which is what the CPU-only tests want.
    """
    all_tactics = range(module.nvfp4_svdquant_gemm_tactic_num())
    can_implement = getattr(module, "nvfp4_svdquant_gemm_can_implement", None)
    if can_implement is None:
        return list(all_tactics)
    tactics = [t for t in all_tactics if can_implement(m, n, k, rank, t)]
    workspace_size = getattr(module, "nvfp4_svdquant_gemm_workspace_size", None)
    if device is None or workspace_size is None or not tactics:
        return tactics
    budget = _sm120_shape_workspace_budget(device, (device.index, m, n, k, rank))
    affordable = [t for t in tactics if int(workspace_size(m, n, k, t)) <= budget]
    if affordable:
        return affordable
    # A shape whose every candidate overruns the budget still has to run.
    return [min(tactics, key=lambda t: int(workspace_size(m, n, k, t)))]


def _nvfp4_svdquant_gemm_runners(
    enable_pdl: bool,
    device: torch.device,
    m: int,
    n: int,
    k: int,
    rank: int,
) -> List[TunableRunner]:
    """Build the per-shape candidate set; first runner is the eager fallback."""
    return [_nvfp4_svdquant_gemm_runner(enable_pdl, device, rank)]


def _sm120_linear_tactics(module, inputs: List[torch.Tensor]) -> List[int]:
    """Consumer rows crossed with the producer configurations this shape admits.

    The producer half indexes ``sm120_producer_variants``, which is computed from
    the launch geometry's own constraints rather than read from a table -- so a
    shape nobody measured still gets the fused prefix, and a card nobody measured
    on ranks the candidates itself.
    """
    m, n, k, rank = _sm120_linear_inputs_shape(inputs)
    rows = _nvfp4_svdquant_valid_tactics(module, m, n, k, rank, inputs[0].device)
    variants = range(len(_sm120_producer_variants(m, k, rank)))
    return [_sm120_pack_tactic(row, v) for v in variants for row in rows]


def _sm120_linear_inputs_shape(inputs: List[torch.Tensor]) -> tuple[int, int, int, int]:
    x, weight_fp4, _, _, _, l2t_smoothed = inputs[:6]
    return x.shape[0], weight_fp4.shape[0], x.shape[1], l2t_smoothed.shape[1]


def _sm120_run_unfused_linear_prefix(
    module,
    inputs: List[torch.Tensor],
    enable_pdl: bool,
) -> None:
    (
        x,
        _,
        _,
        _,
        pre_quant_scale,
        l2t_smoothed,
        _,
        global_scale,
        _,
        xq,
        x_sf,
        down,
        _,
        _,
    ) = inputs
    module.nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        xq,
        x_sf,
        enable_pdl,
    )
    torch.mm(x, l2t_smoothed, out=down)


def _sm120_fused_linear_runner(
    enable_pdl: bool, device: torch.device, rank: int = 32
) -> TunableRunner:
    module = get_nvfp4_svdquant_sm120_module(rank)

    class Sm120FusedLinearRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return _sm120_linear_tactics(module, inputs)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (
                x,
                weight_fp4,
                weight_sf,
                alpha,
                pre_quant_scale,
                l2t_smoothed,
                l1_scaled,
                global_scale,
                bias,
                xq,
                x_sf,
                down,
                out,
                workspace_buffer,
            ) = inputs
            # The tactic names a consumer row and a producer configuration.
            # The geometry and address policy go down explicitly, so the C++
            # side needs no table to turn an index back into a launch.
            m_rt, k_rt = x.shape[0], x.shape[1]
            _, producer_variant = _sm120_unpack_tactic(tactic)
            family, tiling, address_policy = _sm120_decode_producer_variant(
                m_rt, k_rt, producer_variant, rank
            )
            # M537 and packed small-M producers share the packed L2T layout;
            # the remaining producers read the original row-major matrix.
            fused_l2t_smoothed = (
                _cached_sm120_m537_l2t(l2t_smoothed)
                if _sm120_variant_packs_l2t(m_rt, k_rt, producer_variant, rank)
                else l2t_smoothed
            )
            module.nvfp4_svdquant_linear_sm120(
                x,
                pre_quant_scale,
                global_scale,
                fused_l2t_smoothed,
                weight_fp4,
                weight_sf,
                alpha,
                l1_scaled,
                bias,
                xq,
                x_sf,
                down,
                out,
                workspace_buffer,
                tactic,
                enable_pdl,
                family,
                tiling[0],
                tiling[1],
                tiling[2],
                address_policy,
            )
            return out

    return Sm120FusedLinearRunner()


def _sm120_cutlass_linear_runner(
    enable_pdl: bool, device: torch.device, rank: int
) -> TunableRunner:
    module = get_nvfp4_svdquant_sm120_module(rank)

    class Sm120CutlassLinearRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return _nvfp4_svdquant_valid_tactics(
                module, *_sm120_linear_inputs_shape(inputs), device
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            return _sm120_run_cutlass_linear(
                module, inputs, tactic=tactic, enable_pdl=enable_pdl
            )

    return Sm120CutlassLinearRunner()


def _sm120_run_cutlass_linear(
    module,
    inputs: List[torch.Tensor],
    *,
    tactic: int,
    enable_pdl: bool,
):
    """Run the generic quantize/down prefix followed by one fixed CUTLASS K3."""
    _sm120_run_unfused_linear_prefix(module, inputs, enable_pdl)
    (
        _,
        weight_fp4,
        weight_sf,
        alpha,
        _,
        _,
        l1_scaled,
        _,
        bias,
        xq,
        x_sf,
        down,
        out,
        workspace_buffer,
    ) = inputs
    module.nvfp4_svdquant_gemm(
        xq,
        weight_fp4,
        x_sf,
        weight_sf,
        alpha,
        down,
        l1_scaled,
        bias,
        out,
        workspace_buffer,
        tactic,
        enable_pdl,
    )
    return out


def _sm120_linear_runners(
    enable_pdl: bool,
    device: torch.device,
    m: int,
    n: int,
    k: int,
    rank: int,
) -> List[TunableRunner]:
    """Return every full-linear route valid for one exact SM120 shape."""
    runners = []
    if _sm120_fused_linear_supported(m, k, rank):
        runners.append(_sm120_fused_linear_runner(enable_pdl, device, rank))
    runners.append(_sm120_cutlass_linear_runner(enable_pdl, device, rank))
    # Keep rank 64's existing untuned fallback; autotuning can select fusion.
    if rank == 64:
        runners.reverse()
    return runners


@functools.cache
def _cached_sm120_linear_runners(
    enable_pdl: bool,
    device: torch.device,
    m: int,
    n: int,
    k: int,
    rank: int,
) -> List[TunableRunner]:
    """Reuse stateless runner objects for repeated calls to one exact shape."""
    return _sm120_linear_runners(enable_pdl, device, m, n, k, rank)


def _exact_num_tokens_buckets(x, *args):
    """Exact-shape tuning: the only bucket for M is M itself."""
    return (x,)


def _map_to_exact_bucket(x, *args):
    return x


_SVDQUANT_CONSTRAINT_SPECS = (
    ConstraintSpec(
        2,  # a_sf tensor index: 1-D 128x4-swizzled scale buffer sized by (m, k/16)
        0,
        lambda shapes: _swizzled_sf_size(shapes[0][0], shapes[0][1] * 2 // 16),
    ),
    ConstraintSpec(
        5,  # d tensor index: [m, r] LoRA-down output (r kept from the real input)
        0,
        lambda shapes: shapes[0][0],
    ),
    ConstraintSpec(
        8,  # out tensor index
        0,
        lambda shapes: shapes[0][0],
    ),
    ConstraintSpec(
        9,  # workspace_buffer index: scratch; exclude its (resizable) size from the
        0,  # cache key so a mid-tune resize never causes a silent cache miss.
        lambda shapes: shapes[9][0],
    ),
)


_NVFP4_SVDQUANT_GEMM_TUNING_CONFIG_EXACT = TuningConfig(
    use_cuda_graph=True,
    use_cold_l2_cache=True,
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            _exact_num_tokens_buckets,
            _map_to_exact_bucket,
        ),
    ),
    constraint_specs=_SVDQUANT_CONSTRAINT_SPECS,
    tensor_initializers=((9, autotuner_initializer_empty),),
)


_SM120_LINEAR_TUNING_CONFIG_COLD = TuningConfig(
    use_cuda_graph=False,
    use_cold_l2_cache=True,
    tensor_initializers=((13, autotuner_initializer_empty),),
    constraint_specs=(
        ConstraintSpec(
            13,  # workspace size is scratch state, not part of the problem key
            0,
            lambda shapes: shapes[13][0],
        ),
    ),
)


def _sm120_linear_op_name(m: int, k: int, rank: int, *, enable_pdl: bool) -> str:
    """Name the autotune record for one exact SM120 linear shape and PDL mode.

    PDL changes the measured cost of every runner (the launch overlaps the
    preceding kernel), so a profile taken without it must not be replayed with
    it. The in-process dispatch key always separated the two; the persistent
    record did not, so a PDL=False winner could be replayed for PDL=True.

    The route version tracks changes to the admitted candidate set. The `_pdl`
    suffix independently separates PDL-enabled measurements. `enable_pdl` is
    required at every call site so the scheduling mode cannot be omitted.
    """
    name = (
        f"svdquant_linear_sm120_routes_v{_sm120_linear_route_abi_version(m, k, rank)}"
        f"_tactics_v{_sm120_tactic_abi_version(m, k)}"
    )
    return f"{name}_pdl" if enable_pdl else name


_SM120_LINEAR_DISPATCH_CACHE: dict[tuple, tuple[int, int]] = {}


_SM120_TUNING_BUDGET_US: Final = 20_000.0


_SM120_TUNING_REPEAT_FLOOR: Final = 15


_SM120_TUNING_REPEAT_CEIL: Final = 100


_SM120_TUNING_LAUNCH_FLOOR_US: Final = 15.0


_SM120_TUNING_FLOP_PER_US: Final = 8.0e8


_SM120_SELECT_ROUNDS: Final = 3


_SM120_SELECT_FINALISTS: Final = 6


_SM120_SELECT_FINALIST_MARGIN: Final = 0.05


_SM120_SELECT_CAPTURE_WARMUP: Final = 3


def _choose_sm120_linear_runner(
    tuner: AutoTuner,
    runners: List[TunableRunner],
    tuning_config: TuningConfig,
    inputs: List[torch.Tensor],
    *,
    device: torch.device,
    m: int,
    n: int,
    k: int,
    rank: int,
    enable_pdl: bool,
    has_bias: bool,
) -> tuple[TunableRunner, int]:
    """Choose once per exact shape, then bypass repeated shape-key construction."""
    # One computed name for both seams: the persistent autotune record below and
    # the in-process dispatch cache. If they ever diverged, a shape could profile
    # under one name and replay under another.
    op_name = _sm120_linear_op_name(m, k, rank, enable_pdl=enable_pdl)
    cache_key = (
        op_name,
        device,
        m,
        n,
        k,
        rank,
        enable_pdl,
        has_bias,
    )
    if not tuner.is_tuning_mode:
        cached = _SM120_LINEAR_DISPATCH_CACHE.get(cache_key)
        if cached is not None:
            runner_id, tactic = cached
            return runners[runner_id], tactic

    runner, tactic = tuner.choose_one(
        op_name,
        runners,
        tuning_config,
        inputs,
    )
    if tactic >= 0:
        runner_id = next(
            index for index, candidate in enumerate(runners) if candidate is runner
        )
        _SM120_LINEAR_DISPATCH_CACHE[cache_key] = (runner_id, tactic)
    return runner, tactic


def _svdquant_tuning_config(backend_key: str) -> TuningConfig:
    return (
        _NVFP4_SVDQUANT_GEMM_TUNING_CONFIG_EXACT
        if backend_key == "sm120"
        else _NVFP4_SVDQUANT_GEMM_TUNING_CONFIG
    )


_SM120_WORKSPACE_BYTES_CACHE: dict = {}


def _sm120_max_workspace_bytes(
    device: torch.device, m: int, n: int, k: int, rank: int
) -> int:
    key = (device.index, m, n, k, rank)
    cached = _SM120_WORKSPACE_BYTES_CACHE.get(key)
    if cached is not None:
        return cached
    module = _get_nvfp4_svdquant_module_for_device(device, rank)
    required = 0
    for t in _nvfp4_svdquant_valid_tactics(module, m, n, k, rank, device):
        required = max(required, module.nvfp4_svdquant_gemm_workspace_size(m, n, k, t))
    _SM120_WORKSPACE_BYTES_CACHE[key] = required
    return required


def _nvfp4_svdquant_gemm_runner(
    enable_pdl: bool,
    device: torch.device,
    lora_rank: int = SVDQUANT_LORA_RANK_GRANULARITY,
):
    module = _get_nvfp4_svdquant_module_for_device(device, lora_rank)

    class Nvfp4SvdquantGemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            if not hasattr(module, "nvfp4_svdquant_gemm_can_implement"):
                return list(range(module.nvfp4_svdquant_gemm_tactic_num()))
            a, b, _, _, _, d = inputs[:6]
            m, k = a.shape[0], a.shape[1] * 2
            n = b.shape[0]
            rank = d.shape[1]
            return _nvfp4_svdquant_valid_tactics(module, m, n, k, rank, device)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_sf, b_sf, alpha, d, l1, bias, out, workspace_buffer) = inputs
            module.nvfp4_svdquant_gemm(
                a,
                b,
                a_sf,
                b_sf,
                alpha,
                d,
                l1,
                bias,
                out,
                workspace_buffer,
                tactic,
                enable_pdl,
            )
            return out

    return Nvfp4SvdquantGemmRunner()


def mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    enable_pdl: Optional[bool],
) -> torch.Tensor:
    """``backend="cutlass-sm120"`` for :func:`flashinfer.gemm.mm_nvfp4_svdquant`.

    The GEMM half only: the caller has already quantized and projected. The
    tactic space is the CUTLASS module's, and the tuner picks over it under the
    SM120 tactic-ABI namespace so a record can never be read back by a build
    whose tactic ids mean something else.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)
    if out is None:
        out = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=a.device)

    m, n, k, rank = a.shape[0], b.shape[0], a.shape[1] * 2, d.shape[1]
    # Sized to the maximum over candidate tactics so the launcher never has to
    # allocate; the first eager call for a shape provisions the cache buffer,
    # and CUDA-graph capture afterwards reuses it allocation-free.
    workspace_bytes = max(
        DEFAULT_WORKSPACE_SIZE,
        _sm120_max_workspace_bytes(a.device, m, n, k, rank),
    )
    workspace_buffer = _get_cache_buf(
        "nvfp4_svdquant_gemm_workspace", workspace_bytes, a.device
    )

    tuner = AutoTuner.get()
    runners = _nvfp4_svdquant_gemm_runners(enable_pdl, a.device, m, n, k, rank)
    inputs = [a, b, a_sf, b_sf, alpha, d, l1, bias, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        _svdquant_op_name("sm120", rank),
        runners,
        _svdquant_tuning_config("sm120"),
        inputs,
    )
    runner(inputs=inputs, tactic=tactic)
    return out


def svdquant_linear(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    l2t_smoothed: torch.Tensor,
    l1_scaled: torch.Tensor,
    global_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    enable_pdl: Optional[bool],
) -> torch.Tensor:
    """``backend="cutlass-sm120"`` for :func:`flashinfer.gemm.svdquant_linear`.

    The whole chain in one tuned unit, which is what separates this backend from
    upstream's: the prefix that smooth-quantizes and projects LoRA-down
    is a route the tuner selects over -- one fused launch where the composed
    path spends two -- and the tactic it picks names the prefix and the GEMM
    together, so the pair is chosen on measured cost rather than independently.
    Both ranks offer fused and separate preprocessing to the autotuner.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(x.device)

    m, k = x.shape if x.ndim == 2 else (-1, -1)
    n = weight_fp4.shape[0] if weight_fp4.ndim == 2 else -1
    rank = l2t_smoothed.shape[1] if l2t_smoothed.ndim == 2 else -1
    if rank not in (32, 64):
        raise ValueError(
            "the SM120 CUTLASS SVDQuant linear backend supports LoRA ranks "
            f"32 and 64, got {rank}"
        )

    xq = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty(
        (_swizzled_sf_size(m, k // 16),), dtype=torch.uint8, device=x.device
    )
    down = torch.empty((m, rank), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    # Sized before the tactic is known, so it must cover every tactic the
    # tuner can select for this shape.
    workspace_bytes = max(
        DEFAULT_WORKSPACE_SIZE, _sm120_max_workspace_bytes(x.device, m, n, k, rank)
    )
    workspace_buffer = _get_cache_buf(
        "nvfp4_svdquant_gemm_workspace", workspace_bytes, x.device
    )
    inputs = [
        x,
        weight_fp4,
        weight_sf,
        alpha,
        pre_quant_scale,
        l2t_smoothed,
        l1_scaled,
        global_scale,
        bias,
        xq,
        x_sf,
        down,
        out,
        workspace_buffer,
    ]
    tuner = AutoTuner.get()
    runners = _cached_sm120_linear_runners(enable_pdl, x.device, m, n, k, rank)
    runner, tactic = _choose_sm120_linear_runner(
        tuner,
        runners,
        _SM120_LINEAR_TUNING_CONFIG_COLD,
        inputs,
        device=x.device,
        m=m,
        n=n,
        k=k,
        rank=rank,
        enable_pdl=enable_pdl,
        has_bias=bias is not None,
    )
    runner(inputs=inputs, tactic=tactic)
    return out
