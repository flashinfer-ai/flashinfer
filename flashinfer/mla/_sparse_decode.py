# SPDX-License-Identifier: Apache-2.0
"""Generalized sparse-MLA decode front over the same-op per-arch entries.

``sparse_mla_decode`` is one routing surface for the *selection-only*
sparse-MLA decode op: attend exactly the KV tokens named by a precomputed
per-query top-k index list — ``softmax(scale * q @ K_sel^T) @ V_sel`` in the
latent space, with validity coming solely from the index metadata (invalid
slots / per-token active lengths). The kernel applies no positional masking
of its own: no causal structure, no sliding-window validity, no
kernel-derived windows. The front owns NO kernels: every backend is an
existing public FlashInfer entry, called through its own public API with its
own contract. The front routes, validates the routing-level arguments, and
refuses — loudly — whatever it cannot map without changing semantics.

Backends
--------
``"specialized"``
    :func:`flashinfer.mla.sparse_mla_decode_sm90` (SM90). BF16
    ``[B, q_len, 8, 576]`` query, uint8 ``fp8_ds_mla`` Page64
    ``[num_pages, 64, 656]`` paged cache, int32 ``[B, q_len, 2048]``
    flattened-storage-slot indices (``-1`` = invalid). Same math contract:
    softmax over exactly the valid indexed rows, latent (512-dim) value, no
    positional masking, no sink, no LSE output.
``"cuda"``
    :class:`flashinfer.mla.SparseMLASm120Wrapper` (SM120/SM121) pinned to
    its hand-written CUDA kernels, run as a single call on a front-cached
    runner. Same math contract for every storage family the wrapper serves
    (DSv3.2 / GLM-NSA at d_qk 576, GLM-5.3-NoPE / DSV4 / DSV4.1 at d_qk
    512): selection is purely caller-provided (``indices`` bounded by
    per-token ``backend_args['topk_length']``), there is no ``seq_lens``
    and no positional masking anywhere in its contract. The wrapper's
    optional extras are supersets of the same contract and are reachable
    through ``backend_args``: a second indexed pool (``extra_kv_cache`` /
    ``extra_indices`` / ``extra_topk_length`` — more rows in the SAME
    softmax), an ``attn_sink`` (extra denominator mass), and a base-2 LSE
    output (``return_lse`` / ``lse_scale``). Backend names reflect the
    implementation's nature, never the architecture.
``"cake"``
    The same wrapper surface pinned to its Cake SM120 kernels (the
    wrapper's ``backend="cake"`` routes: DSV4 NVFP4, and the DeepSeek-V4.1
    mixed cache), same selection-only math contract. The front owns the
    wrapper's backend knob, so ``backend_args['backend']`` is refused.

Deliberately NOT backends (different op, or different lifecycle)
----------------------------------------------------------------
:func:`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`
    The DeepSeek-V4 sparse-MLA decode contract is a DIFFERENT op, not just a
    different ABI: the kernel itself derives a per-token sliding-window
    validity mask from ``seq_lens`` (only the first
    ``clamp(seq_lens[b] - (q_len_b - 1 - q_off), 0, 128)`` SWA columns of a
    row are attended), the index table is semantically partitioned (a fixed
    128-entry SWA prefix addressing ``swa_kv_cache``, then compressed
    entries addressing ``compressed_kv_cache``), and the head geometry is
    the DSV4 512-wide QK with fused ``bmm1_scale``/``bmm2_scale``
    quantization semantics. Selection there is NOT purely caller-provided —
    the kernel imposes positional (causal/local) structure this op does not
    have — so routing it under this front would umbrella two different math
    contracts under one name. It remains directly reachable under its own
    name.
``flashinfer.attention.prims_ts`` ``batch_sparse_mla_decode_with_paged_kv_cache``
    Same op family, but an experimental namespace with a plan/run wrapper
    lifecycle; it remains directly reachable under its own name.

The two wired contracts share the math but NOT an ABI, and this front does
not pretend they do: the positional core (``query``, ``kv_cache``,
``indices``) is common in spirit only, each backend's tensor formats are its
own, and backend-specific arguments travel in ``backend_args`` under the
underlying entry's own parameter names. See :func:`sparse_mla_decode` for the
exact per-backend mapping and the documented refusals.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ..api_logging import flashinfer_api

# Backends behind the one public API (#5667 pattern; plain strings).
SPARSE_MLA_DECODE_BACKENDS = ("auto", "specialized", "cuda", "cake")

# ``backend="auto"`` device routing: compute capability -> backend.
# SM100/SM103/SM107 have no selection-only sparse-MLA decode entry in-tree
# (their trtllm_batch_decode_sparse_mla_dsv4 cubins implement the DSV4
# windowed dual-pool op — see the module docstring), so "auto" has no route
# there and raises.
_AUTO_ROUTE: Dict[Tuple[int, int], str] = {
    (9, 0): "specialized",
    # The wrapper's own default on SM120/SM121 is its hand-written CUDA
    # kernels; auto is purely a router over the named backends.
    (12, 0): "cuda",
    (12, 1): "cuda",
}

# Per-backend dispatch census (#5667-style introspection).
_stats: Dict[str, Any] = {
    "calls": 0,
    "backend_dispatches": {b: 0 for b in SPARSE_MLA_DECODE_BACKENDS[1:]},
    "auto_routes": {},
}


def _sparse_mla_decode_stats() -> dict:
    """Diagnostics: per-backend dispatch counts and auto-routing census."""
    return {
        "calls": _stats["calls"],
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "auto_routes": dict(_stats["auto_routes"]),
        "backends": SPARSE_MLA_DECODE_BACKENDS,
    }


def _reset_sparse_mla_decode_stats() -> None:
    _stats["calls"] = 0
    for k in _stats["backend_dispatches"]:
        _stats["backend_dispatches"][k] = 0
    _stats["auto_routes"].clear()


def is_sparse_mla_decode_supported(device: torch.device) -> bool:
    """Whether ``backend="auto"`` has a route for ``device``."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) in _AUTO_ROUTE


def _capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        raise ValueError(
            f"sparse_mla_decode requires CUDA tensors, got device {device}"
        )
    return torch.cuda.get_device_capability(device)


def _resolve_backend(backend: str, device: torch.device) -> str:
    if backend not in SPARSE_MLA_DECODE_BACKENDS:
        raise ValueError(
            f"sparse_mla_decode supports backend in "
            f"{SPARSE_MLA_DECODE_BACKENDS}, got {backend!r}; an explicitly "
            "selected backend is never silently swapped"
        )
    if backend != "auto":
        return backend
    cc = _capability(device)
    routed = _AUTO_ROUTE.get(cc)
    if routed is None:
        raise ValueError(
            f"sparse_mla_decode backend='auto' has no selection-only "
            f"sparse-MLA decode route for compute capability "
            f"{cc[0]}.{cc[1]}; supported: SM90 ('specialized') and "
            "SM120/SM121 ('cuda' or 'cake'). The DeepSeek-V4 windowed "
            "dual-pool op (trtllm_batch_decode_sparse_mla_dsv4) is a "
            "different op and keeps its own entry"
        )
    key = f"{cc[0]}.{cc[1]}->{routed}"
    _stats["auto_routes"][key] = _stats["auto_routes"].get(key, 0) + 1
    return routed


def _check_backend_args(
    backend: str, backend_args: Dict[str, Any], allowed: Tuple[str, ...]
) -> None:
    unknown = sorted(set(backend_args) - set(allowed))
    if unknown:
        raise ValueError(
            f"sparse_mla_decode backend={backend!r} does not accept "
            f"backend_args keys {unknown}; allowed: {sorted(allowed)}"
        )


# SparseMLASm120Wrapper constructor knobs (cached per device + knob set) and
# run() keywords, both under the wrapper's own public names. The wrapper's
# own ``backend`` ctor knob is NOT passed through: the front's backend name
# ("cuda" -> the wrapper's hand-written kernels, "cake" -> its Cake kernels)
# owns that choice, so the front never routes one named backend's call to a
# different implementation.
_SM120_CTOR_KEYS = (
    "max_num_tokens",
    "max_num_heads",
    "d_v",
    "kv_scale_format",
    "kv_cache_format",
    "extra_kv_fp4",
    "compute_precision",
)
_SM120_RUN_KEYS = (
    "topk_length",
    "attn_sink",
    "extra_kv_cache",
    "extra_indices",
    "extra_topk_length",
    "out_lse",
    "mid_out",
    "mid_lse",
    "prefill_impl",
    "return_lse",
    "lse_scale",
)

# Front-cached SM120 runners keyed on (device, frozen constructor knobs).
_sm120_runners: Dict[Tuple[Any, ...], Any] = {}


def _run_specialized(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    workspace_buffer: Optional[torch.Tensor],
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    requested: str,
    backend_args: Dict[str, Any],
) -> torch.Tensor:
    _check_backend_args("specialized", backend_args, ())
    if seq_lens is None:
        raise ValueError(
            "sparse_mla_decode backend='specialized' requires seq_lens "
            "(int32 [B] scheduling metadata; see sparse_mla_decode_sm90)"
        )
    if workspace_buffer is None:
        raise ValueError(
            "sparse_mla_decode backend='specialized' requires "
            "workspace_buffer; size it with "
            "get_sparse_mla_sm90_workspace_size_bytes"
        )
    if sm_scale is None:
        raise ValueError("sparse_mla_decode backend='specialized' requires sm_scale")
    from ._sparse_mla_sm90 import sparse_mla_decode_sm90

    # Explicit front selection pins the underlying explicit backend; the
    # front's "auto" defers to the SM90 entry's own dispatch allowlist.
    inner = "auto" if requested == "auto" else "specialized"
    return sparse_mla_decode_sm90(
        query,
        kv_cache,
        indices,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        out=out,
        backend=inner,
    )


def _run_sm120_wrapper(
    backend: str,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    workspace_buffer: Optional[torch.Tensor],
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    backend_args: Dict[str, Any],
) -> torch.Tensor:
    _check_backend_args(backend, backend_args, _SM120_CTOR_KEYS + _SM120_RUN_KEYS)
    if seq_lens is not None:
        # The wrapper consumes per-token active lengths (topk_length), not
        # per-request original KV lengths; mapping seq_lens onto it would
        # change semantics. Refuse instead of papering over the difference.
        raise ValueError(
            f"sparse_mla_decode backend={backend!r} does not read seq_lens: "
            "SparseMLASm120Wrapper takes per-token active lengths as "
            "backend_args['topk_length'] (and "
            "backend_args['extra_topk_length'] for the second indexed pool)"
        )
    if workspace_buffer is not None:
        raise ValueError(
            f"sparse_mla_decode backend={backend!r} does not take a "
            "workspace_buffer: the wrapper manages split-K scratch "
            "internally (or via backend_args['mid_out']/'mid_lse')"
        )
    if sm_scale is None:
        raise ValueError(f"sparse_mla_decode backend={backend!r} requires sm_scale")
    ctor = {k: backend_args[k] for k in _SM120_CTOR_KEYS if k in backend_args}
    # The front's backend name pins the wrapper's implementation: "cuda" is
    # the wrapper's hand-written SM120 kernels (its "sparse" route), "cake"
    # its Cake kernels. An explicitly selected backend is never swapped.
    ctor["backend"] = "sparse" if backend == "cuda" else "cake"
    run = {k: backend_args[k] for k in _SM120_RUN_KEYS if k in backend_args}
    device = query.device
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (device,) + tuple(sorted(ctor.items()))
    runner = _sm120_runners.get(key)
    if runner is None:
        from ._sparse_mla_sm120 import SparseMLASm120Wrapper

        runner = SparseMLASm120Wrapper(device=device, **ctor)
        _sm120_runners[key] = runner
    if out is None:
        d_v = int(ctor.get("d_v", 512))
        out = torch.empty(
            (*query.shape[:-1], d_v), dtype=torch.bfloat16, device=query.device
        )
    lse = runner.run(query, kv_cache, indices, out, sm_scale, **run)
    if run.get("return_lse", False):
        return out, lse  # type: ignore[return-value]
    return out


@flashinfer_api
def sparse_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    *,
    sm_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
    backend_args: Optional[Dict[str, Any]] = None,
):
    """Selection-only sparse-MLA decode over precomputed top-k indices.

    One generalized front for the selection-only sparse-MLA decode op:
    attend exactly the KV tokens named by the caller's index metadata, with
    no positional masking in the kernel (no causal structure, no
    kernel-derived sliding-window validity). It routes to the existing
    public same-op per-arch entries and adds no kernels of its own:

    - ``backend="specialized"`` calls
      :func:`sparse_mla_decode_sm90` (SM90 fused kernel, with that entry's
      own composable-reference fallback and kill switch).
    - ``backend="cuda"`` runs a front-cached
      :class:`SparseMLASm120Wrapper` (SM120/SM121) pinned to its
      hand-written CUDA kernels, as a single call; ``backend="cake"`` the
      same wrapper pinned to its Cake kernels. Backend names reflect the
      implementation's nature, never the architecture.
    - ``backend="auto"`` routes by the query device's compute capability:
      SM90 to ``"specialized"`` (that entry's own allowlist then decides
      fused vs composable) and SM120/SM121 to ``"cuda"`` (the wrapper's own
      default implementation; auto is purely a router). Other devices
      raise — in particular SM100/SM103/SM107, whose in-tree sparse-MLA
      decode entry (:func:`trtllm_batch_decode_sparse_mla_dsv4`) implements
      the DeepSeek-V4 op, not this one (see Notes). An explicitly selected
      backend is never silently swapped.

    The backends implement the same math contract but NOT the same ABI, and
    this front does not unify what the kernels do not: ``query`` /
    ``kv_cache`` / ``indices`` follow the selected backend's documented
    formats (``specialized``: bf16 ``[B, q_len, 8, 576]`` over a uint8
    ``fp8_ds_mla`` Page64 ``[num_pages, 64, 656]`` cache with flattened
    storage-slot indices; ``cuda``/``cake``: packed uint8 FP8/NVFP4 records
    with per-token ``topk_length`` metadata). Everything beyond the shared
    routing core travels in ``backend_args`` under the underlying entry's
    OWN parameter names and is forwarded verbatim; unknown keys raise.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor in the selected backend's documented format.
    kv_cache : torch.Tensor
        Main KV pool. For ``cuda``/``cake`` a second indexed pool may go in
        ``backend_args['extra_kv_cache']`` (more rows in the same softmax).
    indices : torch.Tensor
        Top-k index table in the backend's documented convention. Index
        ENCODING differs by backend (``-1``-padded fixed-width rows for
        ``specialized`` vs per-token active lengths for the wrapper
        backends) and is
        not translated here; the semantics — attend exactly the named rows —
        are the op.
    seq_lens : torch.Tensor, optional
        Per-request original KV lengths, int32 ``[B]``. Required by
        ``specialized`` (scheduling metadata only; it does not mask).
        REFUSED by ``cuda``/``cake`` (that wrapper takes per-token
        ``backend_args['topk_length']`` instead — a documented contract
        difference, not a silent remap).
    workspace_buffer : torch.Tensor, optional
        Byte workspace. Required by ``specialized`` (size with
        :func:`get_sparse_mla_sm90_workspace_size_bytes`). REFUSED by
        ``cuda``/``cake`` (the wrapper manages its own scratch).
    sm_scale : float, optional
        Softmax scale. Required by every backend.
    out : torch.Tensor, optional
        Preallocated output in the backend's documented format. For the
        wrapper backends the front allocates a bf16 latent-space output
        when omitted.
    backend : str
        One of ``SPARSE_MLA_DECODE_BACKENDS = ("auto", "specialized",
        "cuda", "cake")``. Unknown names raise.
    backend_args : dict, optional
        Backend-specific keyword arguments under the underlying entry's own
        names (for ``cuda``/``cake``: constructor knobs such as
        ``kv_cache_format`` / ``kv_scale_format`` and run keywords such as
        ``topk_length`` / ``attn_sink`` / ``return_lse``; the wrapper's own
        ``backend`` ctor knob is owned by the front's backend name and is
        refused). Keys the selected backend does not accept raise
        ``ValueError``.

    Returns
    -------
    torch.Tensor or tuple
        The selected backend's documented return value (``(out, lse)`` for
        the wrapper backends with ``backend_args['return_lse']=True``).

    Notes
    -----
    :func:`trtllm_batch_decode_sparse_mla_dsv4` is intentionally NOT a
    backend: the DeepSeek-V4 contract it implements derives a per-token
    sliding-window validity mask from ``seq_lens``, partitions its index
    table into a fixed 128-entry SWA prefix plus compressed entries over two
    mandatory pools, and uses the DSV4 512-wide QK geometry with fused
    ``bmm1_scale``/``bmm2_scale`` quantization semantics — a different math
    contract, not a different ABI for this one. The experimental
    ``flashinfer.attention.prims_ts`` sparse-MLA decode entry is same-op but
    not wired (experimental namespace, plan/run lifecycle); use it directly.
    Per-backend dispatch counts are visible through
    ``flashinfer.mla._sparse_decode._sparse_mla_decode_stats()``.
    """
    args = dict(backend_args) if backend_args else {}
    resolved = _resolve_backend(backend, query.device)
    _stats["calls"] += 1
    _stats["backend_dispatches"][resolved] += 1
    if resolved == "specialized":
        return _run_specialized(
            query,
            kv_cache,
            indices,
            seq_lens,
            workspace_buffer,
            sm_scale,
            out,
            backend,
            args,
        )
    return _run_sm120_wrapper(
        resolved,
        query,
        kv_cache,
        indices,
        seq_lens,
        workspace_buffer,
        sm_scale,
        out,
        args,
    )
