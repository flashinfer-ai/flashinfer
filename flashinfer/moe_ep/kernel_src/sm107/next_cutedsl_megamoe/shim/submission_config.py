# SPDX-License-Identifier: Apache-2.0
"""Token buckets, kernel settings and environment knobs of the NVFP4 MegaMoE plugin.

The kernel's ``max_tokens_per_rank`` is compile-time, and its shared expert and
router always cover all of them, so the plugin compiles one variant per token
bucket and pads every call to the smallest bucket that holds it. Every EP rank
must pick the same bucket (the variants differ in their symmetric layouts).

Environment:

``VLLM_NVFP4_MEGAMOE_KERNEL_DIR``
    Checkout of NVIDIA cutedsl_megamoe with the shared-expert fusion (the
    ``sources`` package). Required for NVFP4 layers.
``VLLM_NVFP4_MEGAMOE_BUCKETS``
    Comma-separated token buckets (multiples of 128). The largest bucket is
    always ``round_up(max_num_batched_tokens, 128)``.
``VLLM_NVFP4_MEGAMOE_IMPL``
    JSON object of kernel-setting overrides applied to every bucket (keys of
    ``KernelSettings``).
``VLLM_NVFP4_MEGAMOE_IMPL_BY_BUCKET``
    JSON object ``{"<bucket>": {overrides}}``: overrides for single buckets, applied
    after ``VLLM_NVFP4_MEGAMOE_IMPL`` (tuning).
``VLLM_NVFP4_MEGAMOE_RUBIN_TABLE``
    Per-bucket settings measured on Rubin (VR200) for DeepSeek-V4.1-Flash's MoE
    shape, used instead of the checkout's GB200 table for the buckets it covers:
    ``1`` (default) the fastest entry of each bucket, ``bitwise`` only entries whose
    outputs are bit-identical to the GB200 table's, ``0`` off.
``VLLM_NVFP4_MEGAMOE_RUBIN_DEDUP_MIN``
    Smallest bucket that may use a ``combine_dedup`` entry of that table (default 0:
    all); smaller buckets take their bit-identical entry. The dedup combine needs a
    larger local workspace (e.g. 4.0 instead of 1.2 GiB at 12288), which vLLM then
    no longer has for the KV cache.
``VLLM_NVFP4_MEGAMOE_DECODE_KERNEL``
    ``genphase``: a server whose token budget is at most 1024 per rank (disaggregated decode) runs cutedsl_megamoe
    main's generation-phase kernel (genphase.py) instead of the general one; ``general`` (default) keeps it.
``VLLM_NVFP4_MEGAMOE_DECODE_BUCKETS``
    Its buckets: powers of two up to 1024 (default 8,16,32,64,128,256,512,1024; the smallest one holding the
    token budget is always added).
``VLLM_NVFP4_MEGAMOE_DECODE_ROUTE``
    Routing key of the kernel checkout's tuned decode tactics: balanced (default) or power_law.
``VLLM_NVFP4_MEGAMOE_DECODE_TOPK_POSITION``
    ``pre_fc1_quant`` (default) applies the routing weights at FC1, before the FC2-input quantization, as vLLM and
    the general kernel do; ``tuned`` takes each tuned entry's position (see genphase.py).
``VLLM_NVFP4_MEGAMOE_L2_MARGIN``
    Headroom factor on the FC2 input global scale (default 1.0).
``VLLM_NVFP4_MEGAMOE_ALLOW_UNVALIDATED_GEOMETRY``
    ``1`` admits a (physical experts, EP size) pair with redundant (EPLB) experts
    that is not marked validated in ``NVFP4_GEOMETRIES`` (for the GPU validation
    runs themselves). Layouts without redundant experts never need it.

The tuning tables are keyed by the model's LOGICAL expert count: with EPLB the
kernel runs ``logical + redundant`` physical experts (e.g. 400 for 384 logical),
but its settings (tile, cluster, schedule hint, combine mode) do not depend on
the expert count, so the tables of the logical model apply unchanged.
"""

import dataclasses
import json
import logging
import os

logger = logging.getLogger("vllm.plugins.nvfp4_megamoe")

DEFAULT_BUCKETS = (256, 1024, 2048, 3072, 4096, 4608, 5632, 6144, 8192)
TOKEN_ROW_ALIGN = 128  # shared-expert scale atom rows


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def token_buckets(max_num_tokens: int) -> tuple[int, ...]:
    top = round_up(max_num_tokens, TOKEN_ROW_ALIGN)
    text = os.environ.get("VLLM_NVFP4_MEGAMOE_BUCKETS", "")
    chosen = [int(v) for v in text.split(",") if v.strip()] if text else list(DEFAULT_BUCKETS)
    for value in chosen:
        if value <= 0 or value % TOKEN_ROW_ALIGN:
            raise ValueError(f"NVFP4 MegaMoE bucket {value} is not a positive multiple of {TOKEN_ROW_ALIGN}")
    return tuple(sorted({v for v in chosen if v < top} | {top}))


DECODE_KERNEL_ENV = "VLLM_NVFP4_MEGAMOE_DECODE_KERNEL"
DECODE_BUCKETS_ENV = "VLLM_NVFP4_MEGAMOE_DECODE_BUCKETS"
GENPHASE_MAX_TOKENS = 1024  # the generation-phase communication protocol's limit per rank
DEFAULT_DECODE_BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024)


def decode_kernel(max_num_tokens: int) -> str:
    """``genphase`` when this server runs the generation-phase kernel, else ``general``.

    ``VLLM_NVFP4_MEGAMOE_DECODE_KERNEL=genphase`` selects it, and only for a token budget the generation-phase
    kernel covers (at most 1024 tokens per rank): a prefill or aggregated server with the same environment keeps
    the general kernel.
    """
    choice = os.environ.get(DECODE_KERNEL_ENV, "general").strip().lower()
    if choice in ("", "general", "0", "off"):
        return "general"
    if choice != "genphase":
        raise ValueError(f"{DECODE_KERNEL_ENV} must be general or genphase, got {choice!r}")
    return "genphase" if max_num_tokens <= GENPHASE_MAX_TOKENS else "general"


def genphase_buckets(max_num_tokens: int) -> tuple[int, ...]:
    """Generation-phase buckets: powers of two (the tuned tables' keys) up to the first one that holds the budget."""
    text = os.environ.get(DECODE_BUCKETS_ENV, "")
    chosen = [int(v) for v in text.split(",") if v.strip()] if text else list(DEFAULT_DECODE_BUCKETS)
    for value in chosen:
        if value <= 0 or value & (value - 1) or value > GENPHASE_MAX_TOKENS:
            raise ValueError(f"NVFP4 MegaMoE decode bucket {value} is not a power of two up to {GENPHASE_MAX_TOKENS}")
    top = next((v for v in DEFAULT_DECODE_BUCKETS if v >= max_num_tokens), None)
    if top is None:
        raise ValueError(f"{max_num_tokens} tokens exceed the gen-phase limit of {GENPHASE_MAX_TOKENS} per rank")
    return tuple(sorted({v for v in chosen if v < top} | {top}))


def layer_buckets(max_num_tokens: int) -> tuple[int, ...]:
    """The buckets of the kernel this server runs."""
    if decode_kernel(max_num_tokens) == "genphase":
        return genphase_buckets(max_num_tokens)
    return token_buckets(max_num_tokens)


def bucket_for(buckets: tuple[int, ...], num_tokens: int) -> int:
    for bucket in buckets:
        if num_tokens <= bucket:
            return bucket
    raise ValueError(f"{num_tokens} tokens exceed the largest NVFP4 MegaMoE bucket {buckets[-1]}")


@dataclasses.dataclass(frozen=True)
class KernelSettings:
    """Per-bucket implementation settings of BlockScaledSwapAbMegaMoeKernel."""

    mma_tiler_mnk: tuple[int, int, int] = (256, 256, 256)
    cluster_shape_mn: tuple[int, int] = (4, 1)
    fallback_cluster_shape_mn: tuple[int, int] | None = (2, 1)
    schedule_policy: tuple[str, int | None] = ("phase_interleave", 2)
    fc2_tma_stages: int = 1
    epi_flag_batches: tuple[int, int] = (2, 4)
    token_in_flag_batch: int = 1
    fc2_use_bulk: bool = True
    token_back_mode: str = "epi_warps"
    work_id_mode: str = "atomic_counter"
    token_padding_block: int = 64
    sf_padding_block: int = 128
    # Rank-deduplicated combine: the top-k terms of a token that land on one rank are summed there and
    # returned as one BF16 row (needs token_back_mode reuse_dispatch_warps or standalone_warps). Not
    # bit-identical to the default combine: each rank's partial sum is rounded to BF16 once more.
    combine_dedup: bool = False
    # Fused shared expert's FC2 before ("early") or after ("tail") the routed work; None: the kernel decides.
    shared_l2_position: str | None = None
    # Options of cutedsl_megamoe main (None: not passed, the kernel's default). epilogue_warpgroups=2: two epilogue
    # warpgroups share every accumulator tile (token_back_mode epi_warps, no combine dedup). The dedup_* knobs apply to
    # combine_dedup only: copies per fill (1-4), the combine's poll back-off cap, and the epilogue warps reducing
    # leftover combine blocks in their FC2 staging SMEM. Outputs equal the same kernel without them, except
    # epilogue_warpgroups and the tail combine change only the work order (checked by the plugin's GPU check).
    epilogue_warpgroups: int | None = None
    dedup_copies_per_fill: int | None = None
    dedup_poll_max_sleep_ns: int | None = None
    dedup_epilogue_tail_combine: bool | None = None


# Rubin (VR200) settings for DeepSeek-V4.1-Flash NVFP4 (H 5120, I 2304, 384 experts, top-6) at EP4, measured on
# rhea02 with the plugin's full layer forward (staging + kernel + top-k reduce, CUDA graph) on 2026-11-11 (campaign
# dsv4.1_dep4_max_throughput_campaign/analysis/moe-tune/REPORT.md). Each row: (first bucket, last bucket, fastest
# settings, fastest settings with outputs bit-identical to the GB200 table's); the last bucket of every row is the
# measured one. Buckets outside every row keep the GB200 table.
_RUBIN_BASE = dict(mma_tiler_mnk=(256, 256, 256), cluster_shape_mn=(4, 1), fallback_cluster_shape_mn=(2, 1),
                   fc2_tma_stages=1, fc2_use_bulk=True, work_id_mode="atomic_counter", token_padding_block=64,
                   sf_padding_block=128)


def _ew(hint: int, epi: tuple[int, int], tin: int, early: bool) -> dict:
    """Combine from the epilogue warps (the GB200 table's path): outputs bit-identical to it."""
    return dict(schedule_policy=("phase_interleave", hint), epi_flag_batches=epi, token_in_flag_batch=tin,
                token_back_mode="epi_warps", combine_dedup=False, shared_l2_position="early" if early else None)


def _dedup(hint: int, epi: tuple[int, int], tin: int) -> dict:
    """Rank-deduplicated combine from the dispatch warps (one more BF16 rounding, see KernelSettings)."""
    return dict(schedule_policy=("phase_interleave", hint), epi_flag_batches=epi, token_in_flag_batch=tin,
                token_back_mode="reuse_dispatch_warps", combine_dedup=True, shared_l2_position=None)


_RUBIN_DSV41_FLASH = (
    # first, last, fastest, bit-identical
    (3072, 3072, _ew(3, (1, 4), 1, True), _ew(3, (1, 4), 1, True)),
    (3200, 4096, _ew(3, (1, 4), 1, True), _ew(3, (1, 4), 1, True)),
    (4224, 4608, _ew(3, (2, 4), 1, True), _ew(3, (2, 4), 1, True)),
    (4736, 5632, _ew(3, (2, 4), 1, True), _ew(3, (2, 4), 1, True)),
    (5760, 6144, _dedup(2, (1, 1), 1), _ew(3, (2, 4), 1, True)),
    (6272, 8192, _dedup(3, (1, 1), 1), _ew(2, (2, 4), 1, False)),  # bit-identical: the GB200 entry itself
    (8320, 10240, _dedup(3, (1, 1), 1), _ew(3, (1, 1), 1, False)),
    (10368, 12288, _dedup(2, (1, 1), 1), _ew(3, (1, 1), 1, False)),
)


# Two epilogue warpgroups (cutedsl_megamoe main's epilogue_warpgroups=2) on every non-dedup bucket up to 5632: bit-identical
# outputs (work order only), -1.1 to -2.7% at 3072-5632 tokens and within noise at 256-2048 (hecate job 1121188: plugin
# layer forward, CUDA graph, L2 flushed, slowest rank, 3 interleaved rounds, uniform and zipf routing). Above 5632 the
# dedup entries win and two warpgroups need the plain combine (+8 to +20% there, job 1120649). Kernels without the
# option (441ca70) do not receive it (runtime.make_kernel).
_RUBIN_TWO_EPILOGUE_MAX_BUCKET = 5632


def rubin_epilogue_warpgroups(bucket: int, settings: "KernelSettings") -> "KernelSettings":
    """``settings`` with two epilogue warpgroups where the Rubin table is on and the bucket qualifies (see above)."""
    table_on = os.environ.get("VLLM_NVFP4_MEGAMOE_RUBIN_TABLE", "1").strip().lower() not in ("0", "off", "false", "no")
    if (table_on and bucket <= _RUBIN_TWO_EPILOGUE_MAX_BUCKET and not settings.combine_dedup
            and settings.token_back_mode == "epi_warps" and settings.epilogue_warpgroups is None):
        return dataclasses.replace(settings, epilogue_warpgroups=2)
    return settings


def _rubin_settings(bucket: int) -> "KernelSettings | None":
    mode = os.environ.get("VLLM_NVFP4_MEGAMOE_RUBIN_TABLE", "1").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return None
    if mode not in ("1", "on", "true", "yes", "bitwise"):
        raise ValueError(f"VLLM_NVFP4_MEGAMOE_RUBIN_TABLE must be 1, bitwise or 0, got {mode!r}")
    dedup_min = int(os.environ.get("VLLM_NVFP4_MEGAMOE_RUBIN_DEDUP_MIN", "0"))
    for first, last, fastest, bitwise in _RUBIN_DSV41_FLASH:
        if first <= bucket <= last:
            use_fastest = mode != "bitwise" and not (fastest["combine_dedup"] and bucket < dedup_min)
            return KernelSettings(**_RUBIN_BASE, **(fastest if use_fastest else bitwise))
    return None


def _ints(text) -> tuple[int, ...]:
    if isinstance(text, (tuple, list)):
        return tuple(int(v) for v in text)
    return tuple(int(v) for v in str(text).split(","))


def _schedule(text) -> tuple[str, int | None]:
    if isinstance(text, (tuple, list)):
        return (str(text[0]), None if len(text) < 2 or text[1] is None else int(text[1]))
    mode, _, hint = str(text).partition(",")
    return mode.strip(), (int(hint) if hint.strip() not in ("", "none", "None") else None)


def _from_published(params: dict) -> KernelSettings:
    """KernelSettings from one entry of the checkout's EP4 parameter tables."""
    fallback = params.get("fallback_cluster_shape_mnk")
    return KernelSettings(
        mma_tiler_mnk=_ints(params["mma_tiler_mnk"]),
        cluster_shape_mn=_ints(params["cluster_shape_mnk"])[:2],
        fallback_cluster_shape_mn=None if fallback is None else _ints(fallback)[:2],
        schedule_policy=_schedule(params["schedule_policy"]),
        fc2_tma_stages=int(params.get("fc2_tma_stages") or 1),
        epi_flag_batches=_ints(params["epi_flag_batches"]),
        token_in_flag_batch=int(params.get("token_in_flag_batch") or 1),
        fc2_use_bulk=bool(params.get("fc2_use_bulk", True)),
        token_back_mode=str(params.get("token_back_mode", "epi_warps")),
        work_id_mode=str(params.get("work_id_mode", "atomic_counter")),
    )


def kernel_settings(bucket: int, hidden: int, intermediate: int, logical_experts: int, topk: int) -> KernelSettings:
    """Settings of the ``bucket`` variant for a MoE of ``logical_experts`` (NOT the physical count).

    With EPLB the kernel runs more physical experts (redundant replicas), but the settings do not depend on
    the expert count: the schedule hint's minimum depends on the FC1/FC2 blocks and the cluster count, and
    combine_dedup only needs EP <= 32. Matching the tables on the physical count (e.g. 400) would silently
    fall back to ``KernelSettings()`` defaults for every bucket.
    """
    experts = logical_experts
    settings = KernelSettings()
    from ..submission_6cb79ec1.best_params import get_params

    if (hidden, intermediate, experts, topk) == (5120, 2304, 384, 6):
        table_tokens = next((t for t in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if bucket <= t), None)
        if table_tokens is not None:
            try:
                settings = _from_published(get_params("deepseek_v4_1_flash", "nvfp4", table_tokens))
            except Exception as error:  # noqa: BLE001 - fall back to defaults
                logger.warning("NVFP4 MegaMoE: tuned parameters for %d tokens unusable (%s)", table_tokens, error)
    if (hidden, intermediate, experts, topk) == (5120, 2304, 384, 6):
        rubin = _rubin_settings(bucket)
        if rubin is not None:
            settings = rubin
        settings = rubin_epilogue_warpgroups(bucket, settings)
    overrides = os.environ.get("VLLM_NVFP4_MEGAMOE_IMPL")
    if overrides:
        settings = _apply_overrides(settings, json.loads(overrides))
    by_bucket = os.environ.get("VLLM_NVFP4_MEGAMOE_IMPL_BY_BUCKET")
    if by_bucket:
        values = json.loads(by_bucket).get(str(bucket))
        if values:
            settings = _apply_overrides(settings, values)
    return settings


ALLOW_UNVALIDATED_GEOMETRY_ENV = "VLLM_NVFP4_MEGAMOE_ALLOW_UNVALIDATED_GEOMETRY"

# (physical experts, EP size) of NVFP4 layers per (hidden, intermediate, top-k): (validated on GPUs, evidence).
# Enforced only for layouts WITH redundant experts (physical != logical, i.e. EPLB), which are new with EPLB support:
# 25 experts per rank (400 at EP16) takes kernel paths that 24 or 96 per rank never ran (the router's scalar
# per-expert size copy, 25 push CTAs), so every such pair needs its own GPU run before serving. Layouts without
# redundant experts ran before EPLB support at any EP size and stay admitted (a one-time warning when unlisted), so
# EPLB-off behaviour is unchanged.
# Flip a pending pair to True once, at that pair, tests/nvfp4_eplb_gpu_check.py passes ``geometry`` (bitwise table),
# ``dedup`` (production Rubin table, tolerance) and ``move``, and record the job here.
NVFP4_GEOMETRIES: dict[tuple[int, int, int], dict[tuple[int, int], tuple[bool, str]]] = {
    (5120, 2304, 6): {  # DeepSeek-V4.1-Flash
        (384, 4): (True, "tests/nvfp4_gpu_check.py; DEP4 max-throughput campaign, 2026-11-11"),
        (384, 16): (True, "deployed DEP16 decode without EPLB"),
        (388, 4): (False, "pending GPU validation: EPLB with 4 redundant experts at DEP4 (97 per rank)"),
        (400, 4): (True, "tests/nvfp4_eplb_gpu_check.py all + dedup at EP4 (100 per rank), job 1044262, 2026-11-12"),
        (400, 8): (False, "pending GPU validation: EPLB with 16 redundant experts at DEP8 (50 per rank)"),
        (400, 16): (True, "tests/nvfp4_eplb_gpu_check.py all + dedup at EP16 on 4 nodes (25 per rank), job 1044263, "
                          "2026-11-12"),
    },
}
_geometry_warned: set[tuple] = set()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _warn_once(key: tuple, message: str, *args) -> None:
    if key not in _geometry_warned:  # once per process, not once per layer (43 layers per rank)
        _geometry_warned.add(key)
        logger.warning(message, *args)


def geometry_unsupported_reason(num_experts: int, ep_size: int, hidden: int, intermediate: int, topk: int, *,
                                logical_experts: int) -> str | None:
    """Why NVFP4 layers must not run ``num_experts`` physical experts at EP ``ep_size`` (None: admitted).

    Only layouts with redundant experts (``num_experts != logical_experts``) are refused when not validated; see
    ``NVFP4_GEOMETRIES``.
    """
    table = NVFP4_GEOMETRIES.get((hidden, intermediate, topk), {})
    entry = table.get((num_experts, ep_size))
    if entry is not None and entry[0]:
        return None
    key = (num_experts, ep_size, hidden, intermediate, topk)
    if num_experts == logical_experts:
        _warn_once(key, "NVFP4 MegaMoE: %d experts at EP%d (H=%d, I=%d, top-%d) are not in the validated "
                   "geometries; admitted (no redundant experts, as before EPLB support)", *key)
        return None
    if _env_flag(ALLOW_UNVALIDATED_GEOMETRY_ENV):
        _warn_once(key, "NVFP4 MegaMoE: running %d physical experts at EP%d (H=%d, I=%d, top-%d), which is not "
                   "validated (%s)", *key, ALLOW_UNVALIDATED_GEOMETRY_ENV)
        return None
    validated = sorted(pair for pair, (ok, _) in table.items() if ok)
    status = entry[1] if entry is not None else "never ran on GPUs"
    return (
        f"NVFP4 MegaMoE has not been validated with {num_experts} physical experts at EP{ep_size} "
        f"(H={hidden}, I={intermediate}, top-{topk}): {status}. Validated (experts, EP) pairs: {validated}. "
        f"Set {ALLOW_UNVALIDATED_GEOMETRY_ENV}=1 to run it anyway (for GPU validation)."
    )


def _apply_overrides(settings: KernelSettings, values: dict) -> KernelSettings:
    """``VLLM_NVFP4_MEGAMOE_IMPL`` / ``_IMPL_BY_BUCKET`` values over a bucket's settings."""
    values = dict(values)
    for key in ("mma_tiler_mnk", "cluster_shape_mn", "epi_flag_batches"):
        if key in values:
            values[key] = _ints(values[key])
    if "fallback_cluster_shape_mn" in values and values["fallback_cluster_shape_mn"] is not None:
        values["fallback_cluster_shape_mn"] = _ints(values["fallback_cluster_shape_mn"])
    if "schedule_policy" in values:
        values["schedule_policy"] = _schedule(values["schedule_policy"])
    return dataclasses.replace(settings, **values)


def l2_margin() -> float:
    value = float(os.environ.get("VLLM_NVFP4_MEGAMOE_L2_MARGIN", "1.0"))
    if not value >= 1.0:
        raise ValueError("VLLM_NVFP4_MEGAMOE_L2_MARGIN must be >= 1")
    return value
