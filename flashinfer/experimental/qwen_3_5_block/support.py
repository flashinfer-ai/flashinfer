# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight admission for the Qwen 3.5 gated attention block: the geometry, the device / DSL gates, the version probe.

Imports torch and FlashInfer's own probes only -- never the kernels, the CuTe DSL or the vendored cuDNN Frontend tree --
so a serving stack's init-time dispatch question (:func:`qwen_3_5_block_support`) costs nothing when the answer is no.
The per-form decline table is read from the vendored tree only once every lighter gate has passed, and the geometry rules this
module mirrors from that tree are re-read against it then (:func:`check_vendored_rules`).

ONE geometry gate, :meth:`Qwen35BlockGeometry.unsupported_reason`, serves both callers -- the probe and ``api.prepare`` (through
:func:`require_served`) -- so a geometry the probe advertises is one every form's declaration admits, and a geometry it declines is
a typed ``NotImplementedError`` at plan time before any vendored code runs.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from flashinfer.cute_dsl.availability import (
    is_cute_dsl_available,
    is_rubin_cute_dsl_available,
)

FEATURE = "Qwen 3.5 gated attention block (bf16)"
SUPPORTED_CC: Tuple[int, int] = (10, 7)
HEAD_DIM = 256
CUTEDSL_REQUIRED = "4.8.0"
FORMS: Tuple[str, ...] = ("prefill", "extend", "decode")
PAGE_SIZES: Tuple[int, ...] = (16, 32, 64, 128)
PAGE_SIZE_ALIGN = 16
KV_LAYOUTS: Tuple[str, ...] = ("NHD", "HND")
DECODE_MAX_Q_LEN = 4  # the claimed envelope: the packed Rubin d256 decode tile
DECODE_MAX_ROWS = (
    32  # routed (the unpacked tile) and declared up to here; a typed decline past it
)
DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16,)
KV_CACHE_DTYPES: Tuple[torch.dtype, ...] = (torch.bfloat16,)
# The forced 256-wide projection GEMM tile: every projected width must be a multiple of it (the adapter's capture-safe route runs
# the vendored GEMM on that tile only; ``kernels/proj_gemm.py`` ``_forced_tile_config``).
_PROJ_TILE = 256

# The rope_dim envelope: the INTERSECTION of the two renderings stage (1) has in the vendored tree, so one geometry verdict holds
# for every form and every row bucket.  The serving forms declare ``fuse_norm_rope=True``, and the vendored declaration LOWERS the
# DECODE form to the unfused chain at decode row counts (``batch x q_len <= kernels.proj_gemm.SMALL_M_MAX``), so both rules bind:
#   * the fused norm + RoPE projection fork (``_cudnn_frost/gated_attention_block/api.py``, ``_FusedQkvProjection.check_support``):
#     ``rope_dim`` a positive multiple of ``2 * _SUBTILE_N`` = 64 and below ``d_head`` -- each rotate_half pair spans whole epilogue
#     subtiles;
#   * the norm + RoPE kernels of the lowered chain (``_cudnn_frost/gated_attention_block/kernels/qk_norm_rope{,_tma}.py``,
#     ``validate_shape``): ``rope_dim`` a multiple of ``2 * ELEMS_PER_ACCESS`` = 16 whose quotient is a power of two (the
#     rotate_half partner is a whole-lane butterfly shuffle) and within the lanes one head row spans (the stage takes the TMA kernel
#     where its tiling holds and the LDG kernel otherwise, so the LDG bound -- a warp of 16-byte accesses, 256 elements at
#     head_dim 256 -- is the chain's).
# Served at head_dim 256: 64 (Qwen3.5's value) and 128.  The constants are the vendored tree's, mirrored here so the init-time probe
# stays free of the kernels and the DSL; :func:`check_vendored_rules` re-reads them from the tree wherever it is imported (the
# probe's vendored read, ``api.prepare``) and ``test_admission.py`` pins the gate against the real stage checks, so a re-vendor
# that moves a rule fails loudly instead of advertising what the tree declines.
_ROPE_SUBTILE_PAIR = 64  # 2 * _FusedQkvProjection._SUBTILE_N: the fused fork rotates whole epilogue-subtile pairs
_ROPE_ACCESS_ELEMS = (
    8  # ELEMS_PER_ACCESS of both norm + RoPE kernels: bf16 elements per 16-byte access
)
_ROPE_ROW_LANES = 32  # kernels/qk_norm_rope.py lanes_per_row: a head row spans at most one warp of accesses


def _rope_fork_ok(rope_dim: int, head_dim: int) -> bool:
    """The fused norm + RoPE projection fork's rule."""
    return rope_dim % _ROPE_SUBTILE_PAIR == 0 and 0 < rope_dim < head_dim


def _rope_chain_ok(rope_dim: int, head_dim: int) -> bool:
    """The lowered chain's norm + RoPE kernels' rule (the LDG kernel's, the one the stage falls back to)."""
    pair = 2 * _ROPE_ACCESS_ELEMS
    lanes = min(_ROPE_ROW_LANES, head_dim // _ROPE_ACCESS_ELEMS)
    half = rope_dim // pair
    return (
        rope_dim % pair == 0
        and half > 0
        and half & (half - 1) == 0
        and rope_dim <= lanes * _ROPE_ACCESS_ELEMS
    )


#: the rope_dim values the block serves at ``HEAD_DIM`` (both stage-(1) renderings admit them): ``(64, 128)``
ROPE_DIMS: Tuple[int, ...] = tuple(
    r
    for r in range(2, HEAD_DIM + 1, 2)
    if _rope_fork_ok(r, HEAD_DIM) and _rope_chain_ok(r, HEAD_DIM)
)


def rope_dim_reason(rope_dim: int, head_dim: int = HEAD_DIM) -> Optional[str]:
    """Why the kernels do not serve this ``rope_dim`` at this ``head_dim`` (``None`` = served): the fused projection fork's rule,
    then the lowered decode chain's -- the two renderings of stage (1) a geometry has to pass (see the module constants)."""
    served = f"the Qwen 3.5 block serves rope_dim in {ROPE_DIMS} at head_dim {HEAD_DIM}"
    if not _rope_fork_ok(rope_dim, head_dim):
        return (
            f"rope_dim must be a positive multiple of {_ROPE_SUBTILE_PAIR} below head_dim={head_dim} (the fused norm + RoPE "
            f"projection rotates whole epilogue-subtile pairs); got rope_dim={rope_dim}; {served}"
        )
    if not _rope_chain_ok(rope_dim, head_dim):
        pair = 2 * _ROPE_ACCESS_ELEMS
        bound = (
            min(_ROPE_ROW_LANES, head_dim // _ROPE_ACCESS_ELEMS) * _ROPE_ACCESS_ELEMS
        )
        return (
            f"rope_dim must be {pair} x a power of two and at most {bound} (the norm + RoPE kernels of the lowered decode chain "
            f"shuffle the rotate_half partner across whole lanes); got rope_dim={rope_dim}; {served}"
        )
    return None


_VENDORED = Path(__file__).resolve().parent / "_cudnn_frost"


@dataclass(frozen=True)
class Qwen35BlockGeometry:
    """The sub-layer's shape at THIS tensor-parallel rank (Qwen3.5-397B at TP 1: ``d_model=4096, num_qo_heads=32,
    num_kv_heads=2``; ``24 / 2`` at ``d_model=2560`` for the 27B-class shards).  ``is_causal`` with the bottom-right
    diagonal, no window and no soft-cap are fixed, not parameters.  The constructor checks the FORM of the fields (positive
    ints, whole GQA groups, an even ``rope_dim`` within the head); :meth:`unsupported_reason` is the kernels' envelope
    (``head_dim`` 256, ``d_model`` a multiple of the projection tile, ``rope_dim`` in :data:`ROPE_DIMS`)."""

    d_model: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int = HEAD_DIM
    rope_dim: int = 64
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    sm_scale: Optional[float] = None

    def __post_init__(self):
        for name in ("d_model", "num_qo_heads", "num_kv_heads", "head_dim", "rope_dim"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"Qwen35BlockGeometry.{name} must be a positive int, got {v!r}"
                )
        if self.num_qo_heads % self.num_kv_heads:
            raise ValueError(
                f"Qwen35BlockGeometry: num_qo_heads={self.num_qo_heads} must be a multiple of num_kv_heads={self.num_kv_heads} (GQA groups)"
            )
        if self.rope_dim % 2 or self.rope_dim > self.head_dim:
            raise ValueError(
                f"Qwen35BlockGeometry: rope_dim={self.rope_dim} must be even and at most head_dim={self.head_dim}"
            )
        if not self.qk_norm_eps > 0.0:
            raise ValueError(
                f"Qwen35BlockGeometry: qk_norm_eps must be > 0, got {self.qk_norm_eps}"
            )
        if self.sm_scale is not None and not (float(self.sm_scale) > 0.0):
            raise ValueError(
                f"Qwen35BlockGeometry: sm_scale must be positive or None, got {self.sm_scale}"
            )

    @property
    def group(self) -> int:
        """Query heads per KV head."""
        return self.num_qo_heads // self.num_kv_heads

    @property
    def n_qkvg(self) -> int:
        """Columns of the fused ``W_qkvg`` (``Q | GATE | K | V`` bands)."""
        return (2 * self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def scale(self) -> float:
        return (
            float(self.sm_scale)
            if self.sm_scale is not None
            else float(self.head_dim) ** -0.5
        )

    def unsupported_reason(self) -> Optional[str]:
        """Why the kernels do not serve this geometry (``None`` = served): the geometry envelope, independent of the device."""
        if self.head_dim != HEAD_DIM:
            return f"the Qwen 3.5 block kernels serve head_dim={HEAD_DIM} only (the Rubin d256 flavor); got head_dim={self.head_dim}"
        if self.d_model % _PROJ_TILE:
            return f"d_model must be a multiple of {_PROJ_TILE} (the forced projection GEMM tile); got d_model={self.d_model}"
        return rope_dim_reason(self.rope_dim, self.head_dim)

    def frost(self):
        """The vendored cuDNN Frontend geometry (imports the vendored tree)."""
        from ._cudnn_frost.gated_attention_block import GatedAttentionBlockGeometry

        return GatedAttentionBlockGeometry(
            d_model=self.d_model,
            h_q=self.num_qo_heads,
            h_kv=self.num_kv_heads,
            d_head=self.head_dim,
            rope_dim=self.rope_dim,
            qk_norm_eps=self.qk_norm_eps,
            attn_scale=self.sm_scale,
            is_causal=True,
            causal_bottom_right=True,
            qk_norm=self.qk_norm,
        )


@dataclass(frozen=True)
class Qwen35BlockSource:
    """Where the kernels come from: the vendored cuDNN Frontend commit and version (``VENDOR_MANIFEST.json``), the CuTe DSL and
    compute capability they were validated with, and the manifest's own digest."""

    origin: str
    commit: str
    version: str
    files: int
    manifest_sha256: str
    validated_cutedsl: str = CUTEDSL_REQUIRED
    validated_compute_capability: str = "10.7"


@functools.lru_cache(maxsize=1)
def source() -> Qwen35BlockSource:
    """The version probe (cached): read once from the vendored manifest, no kernel import."""
    path = _VENDORED / "VENDOR_MANIFEST.json"
    raw = path.read_bytes()
    m = json.loads(raw)
    return Qwen35BlockSource(
        origin=str(m.get("origin", "")),
        commit=str(m.get("commit", "")),
        version=str(m.get("cudnn_frontend_version", "")),
        files=len(m.get("files", [])),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def cutedsl_version() -> Optional[str]:
    """The installed ``nvidia-cutlass-dsl`` version, or ``None``."""
    for dist in ("nvidia-cutlass-dsl", "nvidia_cutlass_dsl"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _version_tuple(v: str) -> Tuple[int, ...]:
    out = []
    for part in v.split("+")[0].split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def cutedsl_reason() -> Optional[str]:
    """Why the installed CuTe DSL cannot serve the block (``None`` = it can): absent, below the floor, or without the Rubin target."""
    if not is_cute_dsl_available():
        return f"nvidia-cutlass-dsl is not installed; the Qwen 3.5 block kernels need nvidia-cutlass-dsl >= {CUTEDSL_REQUIRED} (the sm_107a target)"
    version = cutedsl_version()
    if version is not None and _version_tuple(version) < _version_tuple(
        CUTEDSL_REQUIRED
    ):
        return f"nvidia-cutlass-dsl {version} is below the {CUTEDSL_REQUIRED} the Qwen 3.5 block kernels need (the first wheel with the sm_107a target)"
    if not is_rubin_cute_dsl_available():
        return (
            f"the installed nvidia-cutlass-dsl ({version or 'unknown version'}) has no Rubin target (cutlass.utils.rubin_helpers); "
            f"the Qwen 3.5 block kernels need nvidia-cutlass-dsl >= {CUTEDSL_REQUIRED}"
        )
    return None


def device_cc(device) -> Optional[Tuple[int, int]]:
    """``(major, minor)`` of a CUDA device, ``None`` for anything else (never raises)."""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 - a probe never raises
        return None


def device_reason(device) -> Optional[str]:
    cc = device_cc(device)
    if cc != SUPPORTED_CC:
        found = "no CUDA device" if cc is None else f"SM{cc[0]}{cc[1]}"
        return f"the Qwen 3.5 block targets SM{SUPPORTED_CC[0]}{SUPPORTED_CC[1]} (Rubin, cc 10.7) only; found {found}"
    return None


def dtype_reason(dtype: torch.dtype) -> Optional[str]:
    if dtype not in DTYPES:
        return f"the bf16 entry points serve torch.bfloat16 activations, weights and pools only; got {dtype}"
    return None


@dataclass(frozen=True)
class Qwen35BlockFormSupport:
    form: str
    served: bool
    reason: Optional[str] = None
    #: every typed decline the form can raise at plan time, by name, with its message (the FE decline table, verbatim)
    declines: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Qwen35BlockSupport:
    """The static answer to "can this process run the block for this geometry / dtype on this device" -- read once at init by a
    serving stack's dispatch predicate instead of catching exceptions.  CPU-constructible; never raises for an unsupported device."""

    served: bool
    reason: Optional[str]
    device_cc: Optional[Tuple[int, int]]
    cutedsl_version: Optional[str]
    cutedsl_ok: bool
    forms: Tuple[str, ...]
    prefill: Qwen35BlockFormSupport
    extend: Qwen35BlockFormSupport
    decode: Qwen35BlockFormSupport
    source: Qwen35BlockSource
    cutedsl_required: str = CUTEDSL_REQUIRED
    page_sizes: Tuple[int, ...] = PAGE_SIZES
    kv_layouts: Tuple[str, ...] = KV_LAYOUTS
    decode_max_q_len: int = DECODE_MAX_Q_LEN
    decode_max_rows: int = DECODE_MAX_ROWS
    dtypes: Tuple[torch.dtype, ...] = DTYPES
    kv_cache_dtypes: Tuple[torch.dtype, ...] = KV_CACHE_DTYPES
    #: the rope_dim values served at head_dim 256 (appended; the geometry gate's envelope)
    rope_dims: Tuple[int, ...] = ROPE_DIMS

    def form(self, name: str) -> Qwen35BlockFormSupport:
        if name not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, got {name!r}")
        return getattr(self, name)


def _frost_form_table(
    geometry: Qwen35BlockGeometry, dtype: torch.dtype, device
) -> Dict[str, Any]:
    """The vendored tree's own envelope for the two paged forms (imports the vendored tree: called after the light gates passed)."""
    from ._cudnn_frost.gated_attention_block import GatedAttentionBlockFwd

    check_vendored_rules()  # the mirrored geometry rules against the tree's: a drift is a loud refusal, never a false verdict
    env = GatedAttentionBlockFwd.serving_envelope(
        geometry.frost(), dtype, torch.device(device), device_cc=SUPPORTED_CC
    )
    return {
        "extend": (env.extend.served, env.extend.reason, dict(env.extend.declines)),
        "decode": (env.decode.served, env.decode.reason, dict(env.decode.declines)),
    }


def qwen_3_5_block_support(
    device, geometry: Qwen35BlockGeometry, *, dtype: torch.dtype = torch.bfloat16
) -> Qwen35BlockSupport:
    """See :func:`flashinfer.qwen_3_5_block.qwen_3_5_block_support` (the public, decorated entry)."""
    if not isinstance(geometry, Qwen35BlockGeometry):
        raise TypeError(
            f"geometry must be a Qwen35BlockGeometry, got {type(geometry).__name__}"
        )
    cc = device_cc(device)
    version = cutedsl_version()
    dsl_reason = cutedsl_reason()
    common = (
        geometry.unsupported_reason()
        or dtype_reason(dtype)
        or device_reason(device)
        or dsl_reason
    )
    forms: Dict[str, Qwen35BlockFormSupport] = {}
    if common is not None:
        for name in FORMS:
            forms[name] = Qwen35BlockFormSupport(name, False, common)
    else:
        try:
            table = _frost_form_table(geometry, dtype, device)
            prefill_reason = None
        except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
            why = f"the vendored cuDNN Frontend tree could not answer: {type(exc).__name__}: {exc}"
            table = {name: (False, why, {}) for name in ("extend", "decode")}
            prefill_reason = why
        # The dense prefill form has no row in the vendored envelope (its declaration meets no table row the gates above did not
        # already settle: the device, the DSL, the dtype, head_dim, d_model and rope_dim are the geometry gate's), so it is served
        # exactly when the vendored tree answered -- never advertised when the tree could not be read.
        forms["prefill"] = Qwen35BlockFormSupport(
            "prefill", prefill_reason is None, prefill_reason
        )
        for name in ("extend", "decode"):
            served, reason, declines = table[name]
            forms[name] = Qwen35BlockFormSupport(name, bool(served), reason, declines)
    served_forms = tuple(name for name in FORMS if forms[name].served)
    return Qwen35BlockSupport(
        served=bool(served_forms),
        reason=common
        if common is not None
        else (None if served_forms else forms["decode"].reason),
        device_cc=cc,
        cutedsl_version=version,
        cutedsl_ok=dsl_reason is None,
        forms=served_forms,
        prefill=forms["prefill"],
        extend=forms["extend"],
        decode=forms["decode"],
        source=source(),
    )


def _passes(check, *args) -> bool:
    try:
        check(*args)
    except (ValueError, NotImplementedError):
        return False
    return True


@functools.lru_cache(maxsize=1)
def check_vendored_rules() -> Dict[str, Any]:
    """Re-read the geometry rules this module mirrors from the vendored tree (imports it) and refuse a drift.

    A re-vendor that moves a rule must fail HERE, loudly -- before the probe advertises, or ``prepare`` declines, a geometry the
    tree reads the other way.  Compared: the fused fork's tile constants (``_FusedQkvProjection._TILE_N`` = head_dim, ``2 *
    _SUBTILE_N`` = the rope_dim pair), the lowered chain's rope rule as BEHAVIOUR (the TMA and the LDG norm + RoPE kernels'
    ``validate_shape`` over every even ``rope_dim`` up to the head dim, at a geometry the TMA tiling fits), the decode row cap,
    the validated page sizes and the target cc.  Once per process (the probe's vendored read, ``api.prepare``); returns the
    tree's constants.  Raises ``RuntimeError`` naming every drifted fact."""
    from ._cudnn_frost.gated_attention_block import api as vendored
    from ._cudnn_frost.gated_attention_block.kernels import (
        qk_norm_rope,
        qk_norm_rope_tma,
    )

    fork = vendored._FusedQkvProjection
    facts: Dict[str, Any] = {
        "head_dim": int(fork._TILE_N),
        "rope_subtile_pair": 2 * int(fork._SUBTILE_N),
        "rope_access_elems": int(qk_norm_rope.ELEMS_PER_ACCESS),
        "rope_row_lanes": int(qk_norm_rope.lanes_per_row(HEAD_DIM)),
        "decode_max_rows": int(vendored.DECODE_MAX_ROWS),
        "page_sizes": tuple(int(p) for p in vendored.SERVING_PAGE_SIZES),
        "supported_cc": tuple(int(x) for x in vendored._SM107_CC),
    }
    mirrored: Dict[str, Any] = {
        "head_dim": HEAD_DIM,
        "rope_subtile_pair": _ROPE_SUBTILE_PAIR,
        "rope_access_elems": _ROPE_ACCESS_ELEMS,
        "rope_row_lanes": _ROPE_ROW_LANES,
        "decode_max_rows": DECODE_MAX_ROWS,
        "page_sizes": PAGE_SIZES,
        "supported_cc": SUPPORTED_CC,
    }
    drift = [
        f"{k}: support.py has {mirrored[k]!r}, the vendored tree {facts[k]!r}"
        for k in facts
        if mirrored[k] != facts[k]
    ]
    # The chain's rule as the stage applies it: the TMA kernel where its tiling holds (32 / 2 heads, 16-row tiles, the stage's
    # thread count), else the LDG kernel -- so a rope_dim is served by the chain iff one of the two validators takes it.
    threads = int(vendored._QK_NORM_ROPE_THREADS)
    tile_rows = int(vendored._QK_NORM_ROPE_TILE_ROWS)
    for rope in range(2, HEAD_DIM + 1, 2):
        tree = _passes(
            qk_norm_rope_tma.validate_shape, HEAD_DIM, rope, 32, 2, tile_rows, threads
        ) or _passes(qk_norm_rope.validate_shape, HEAD_DIM, rope, threads)
        if tree != _rope_chain_ok(rope, HEAD_DIM):
            drift.append(
                f"rope_dim={rope}: the lowered chain's norm + RoPE kernels {'take' if tree else 'decline'} it, support.py reads the other way"
            )
    if drift:
        raise RuntimeError(
            "qwen_3_5_block: the geometry rules support.py mirrors differ from the vendored cuDNN Frontend tree's; the gate "
            "(Qwen35BlockGeometry.unsupported_reason) must be re-derived with the re-vendored tree: "
            + "; ".join(drift)
        )
    return facts


def require_served(
    device, geometry: Qwen35BlockGeometry, dtype: torch.dtype, form: str
) -> None:
    """The plan-time gate: a typed ``NotImplementedError`` naming the device, the DSL version, the dtype or the geometry."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    reason = (
        geometry.unsupported_reason()
        or dtype_reason(dtype)
        or device_reason(device)
        or cutedsl_reason()
    )
    if reason is not None:
        raise NotImplementedError(f"qwen_3_5_block ({form}): {reason}")
