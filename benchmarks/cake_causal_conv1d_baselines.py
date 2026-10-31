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

Non-Cake baselines ("arms") of the Cake causal conv1d + SiLU training benchmarks
``bench_cake_causal_conv1d_train.py`` (single rank) and ``bench_cake_causal_conv1d_train_kcp.py`` (context parallel).

Every arm is the public training entry point of another library, called through autograd on the benchmark's own
tensors without a copy (``[S, C]`` maps to ``[1, S, C]`` or to its channel-last ``[1, C, S]`` view).  Index tensors
and contexts are built once per row, outside the timing.  The phases are the benchmark's: ``fwd`` (the training
forward, its graph recorded and dropped), ``bwd`` (the backward of a recorded forward, ``retain_graph=True``),
``step`` (forward + backward, single rank) and ``ckpt`` (``torch.utils.checkpoint`` around the entry, then its
backward); the leaves' gradients are reset before every backward.

Single-rank arms (the packed documents as one ``B = 1`` batch):

* ``fla``: flash-linear-attention ``fla.modules.causal_conv1d.causal_conv1d`` (``cu_seqlens``), its default Triton
  backend (``FLA_CONV_GLUON`` and ``FLA_GLUON`` unset while the arm runs);
* ``fla_gluon``: the same entry point with its opt-in Gluon backend (``FLA_CONV_GLUON=1``; the Gluon verifiers
  must accept the call, else the arm is unsupported);
* ``dao``: Dao-AILab ``causal_conv1d.causal_conv1d_fn`` on the channel-last view with a per-token ``seq_idx``,
  its default atomic weight gradient (``CAUSAL_CONV1D_DETERMINISTIC`` unset);
* ``dao_det``: the same with ``CAUSAL_CONV1D_DETERMINISTIC=1`` (an ordered weight-gradient reduction);
* ``cudnn_fe``: the cuDNN frontend's ``cudnn.ops.causal_conv1d`` on the channel-last view with ``cu_seqlens`` (its
  CuTe DSL route; with ``torch`` deterministic algorithms off, an atomic weight gradient).

Context-parallel arms (the benchmark's process group; the halo exchange is part of every phase):

* ``fla_cp_triton`` / ``fla_cp_cuda``: FLA's ``causal_conv1d(..., cp_context=build_cp_context(...),
  backend="triton" | "cuda")`` on FLA's own partition of the pack, padded with one zero document to a multiple of
  ``2 W`` tokens (``contiguous`` and ``zigzag``; ``cuda`` runs Dao's kernel on the ``contiguous`` layout only);
* ``attn_gym_cp``: Attention Gym's ``context_parallel_conv_history`` (the fragment tails all-gathered) feeding
  ``attn_gym.linear.short_conv.causal_conv1d`` (CuTe DSL), on the benchmark's own fragment table (every layout).
  Its taps must have the activations' dtype, so it runs BF16 taps against the FP32 taps of the Cake rows.

Each arm records the library version (``fla.__version__``, ``causal_conv1d.__version__``, ``cudnn.__version__``;
Attention Gym has none, so ``ATTN_GYM_COMMIT`` names its commit), its determinism, and its feature match against
the Cake route of the row: ``exact``, ``more_precise`` (SiLU' at the FP32 preactivation where the Cake route takes
the BF16-rounded one) or ``less_precise`` (the BF16-rounded preactivation against the Cake ``fp32`` policy, or BF16
taps against FP32 taps).  One untimed call per arm checks its numerics against the Cake non-deterministic result
(relative L2 error of ``y``, ``dx``, ``dweight``, ``dbias`` below ``NUMERICS_REL_L2``).  An arm is ``eligible`` for
the best baseline when it is an exact or more precise match and passes the check.  A row an arm cannot run is
reported with the concrete reason.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from torch.utils.checkpoint import checkpoint

WIDTH = 4
NUMERICS_REL_L2 = 1e-2
SINGLE_RANK_ARMS = ("fla", "fla_gluon", "dao", "dao_det", "cudnn_fe")
KCP_ARMS = ("fla_cp_triton", "fla_cp_cuda", "attn_gym_cp")
FLA_ENTRY = "fla.modules.causal_conv1d.causal_conv1d"
DAO_ENTRY = "causal_conv1d.causal_conv1d_fn"
CUDNN_ENTRY = "cudnn.ops.causal_conv1d"
ATTN_GYM_ENTRY = (
    "attn_gym.linear.context_parallel.context_parallel_conv_history + "
    "attn_gym.linear.short_conv.causal_conv1d"
)
# While an arm runs: its environment (None: unset).
_FLA_TRITON_ENV: dict[str, Optional[str]] = {"FLA_CONV_GLUON": None, "FLA_GLUON": None}
_FLA_GLUON_ENV: dict[str, Optional[str]] = {"FLA_CONV_GLUON": "1", "FLA_GLUON": None}
_DAO_ENV: dict[str, Optional[str]] = {"CAUSAL_CONV1D_DETERMINISTIC": None}
_DAO_DET_ENV: dict[str, Optional[str]] = {"CAUSAL_CONV1D_DETERMINISTIC": "1"}


class Unsupported(Exception):
    """The arm cannot run the row; the message is the concrete reason."""


@contextlib.contextmanager
def environment(values: dict[str, Optional[str]]) -> Iterator[None]:
    """Set (or unset, for ``None``) environment variables for the block, then restore them."""

    saved = {key: os.environ.get(key) for key in values}

    def apply(settings: dict[str, Optional[str]]) -> None:
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    apply(values)
    try:
        yield
    finally:
        apply(saved)


def dtype_name(dtype: Optional[torch.dtype]) -> Optional[str]:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _library(module: str, package: str) -> str:
    """The library's version; ``Unsupported`` when it is not importable."""

    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 -- any import failure means the arm is absent
        raise Unsupported(
            f"{package} is not importable: {type(exc).__name__}: {exc}"
        ) from exc
    version = getattr(mod, "__version__", None)
    if version is None and module == "attn_gym":
        version = os.environ.get("ATTN_GYM_COMMIT")
    return f"{package} {version or 'unknown'}"


def _distribution(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def feature(
    *,
    preactivation: str,
    taps: Optional[torch.dtype],
    bias: Optional[torch.dtype],
    cake_preactivation: str,
    cake_taps: torch.dtype,
) -> dict:
    """The arm's features on a row and their match against the Cake route of the row."""

    match, notes = "exact", []
    if preactivation != cake_preactivation:
        match = "more_precise" if preactivation == "fp32" else "less_precise"
        notes.append(
            f"SiLU' at the {preactivation} preactivation (Cake route: {cake_preactivation})"
        )
    if taps != cake_taps:
        match = "less_precise"
        notes.append(
            f"{dtype_name(taps)} taps (Cake route: {dtype_name(cake_taps)} taps)"
        )
    return {
        "taps": dtype_name(taps),
        "bias": dtype_name(bias),
        "preactivation": preactivation,
        "match": match,
        "notes": notes,
    }


def autograd_calls(
    layer: Callable[..., tuple],
    leaves: Sequence[torch.Tensor],
    dys: Sequence[torch.Tensor],
) -> dict[str, Callable[[], object]]:
    """``{name: fn}`` of an arm through autograd: the timed phases ``fwd`` / ``bwd`` / ``step`` / ``ckpt`` and
    ``record`` (the forward ``bwd`` replays the backward of)."""

    state: dict[str, tuple] = {}

    def reset() -> None:
        for leaf in leaves:
            leaf.grad = None

    def grads() -> list[Optional[torch.Tensor]]:
        return [leaf.grad for leaf in leaves]

    def fwd():
        return layer(*leaves)

    def record():
        state["ys"] = layer(*leaves)
        return state["ys"]

    def bwd():
        reset()
        torch.autograd.backward(list(state["ys"]), list(dys), retain_graph=True)
        return grads()

    def step():
        reset()
        torch.autograd.backward(list(layer(*leaves)), list(dys))
        return grads()

    def ckpt():
        reset()
        ys = checkpoint(layer, *leaves, use_reentrant=False)
        torch.autograd.backward(list(ys), list(dys))
        return grads()

    return {
        "fwd": fwd,
        "record": record,
        "bwd": bwd,
        "step": step,
        "ckpt": ckpt,
    }


def _sums(
    out: torch.Tensor, ref: torch.Tensor
) -> tuple[float, float]:  # (sum (out - ref)^2, sum ref^2)
    diff = out.detach().double() - ref.detach().double()
    return float((diff * diff).sum()), float(ref.detach().double().square().sum())


def _rel(sums: tuple[float, float]) -> float:
    err, norm = sums
    return (err / norm) ** 0.5 if norm > 0 else (0.0 if err == 0 else float("inf"))


def _numerics(rel: dict[str, float]) -> dict:
    return {
        "rel_l2": rel,
        "ok": all(value < NUMERICS_REL_L2 for value in rel.values()),
        "budget": NUMERICS_REL_L2,
    }


def _bias(leaves: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    return leaves[2] if len(leaves) > 2 else None


# ---------------------------------------------------------------------------------------------------------------------
# Single rank
# ---------------------------------------------------------------------------------------------------------------------


def _fla_layer(cu: torch.Tensor) -> Callable[..., tuple]:
    from fla.modules.causal_conv1d import causal_conv1d  # type: ignore

    cu_long = cu.long()
    cu_cpu = cu_long.cpu()

    def layer(*t: torch.Tensor) -> tuple:
        y, _ = causal_conv1d(
            t[0].unsqueeze(0),
            t[1],
            bias=_bias(t),
            activation="silu",
            cu_seqlens=cu_long,
            cu_seqlens_cpu=cu_cpu,
        )
        return (y.squeeze(0),)

    return layer


def _gluon_verdict(x: torch.Tensor, w: torch.Tensor, cu: torch.Tensor) -> None:
    """``Unsupported`` unless FLA's dispatch would route this call's forward and backward to Gluon."""

    try:
        from fla.backends import GluonBackend  # type: ignore
        from fla.modules.causal_conv1d.backends import gluon  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise Unsupported(
            f"FLA Gluon backend not importable: {type(exc).__name__}: {exc}"
        ) from exc
    if not GluonBackend.is_available():
        raise Unsupported("FLA GluonBackend.is_available() is False")
    with environment(_FLA_GLUON_ENV):
        if not GluonBackend(env_var="FLA_CONV_GLUON").is_enabled():
            raise Unsupported("FLA Gluon backend not enabled by FLA_CONV_GLUON=1")
    x3 = x.unsqueeze(0)
    for verifier, args in (
        (gluon.causal_conv1d_fwd_verifier, dict(x=x3, weight=w, cu_seqlens=cu)),
        (
            gluon.causal_conv1d_bwd_verifier,
            dict(x=x3, dy=x3, dht=None, weight=w, cu_seqlens=cu),
        ),
    ):
        accepted, reason = verifier(**args)
        if not accepted:
            raise Unsupported(
                f"FLA Gluon {verifier.__name__} rejects the call: {reason}"
            )


def _dao_layer(x: torch.Tensor, lens: Sequence[int]) -> Callable[..., tuple]:
    from causal_conv1d import causal_conv1d_fn  # type: ignore

    # One document id per token (int32 [1, S]), built once: the channel-last kernels take seq_idx, not cu_seqlens.
    seq_idx = (
        torch.repeat_interleave(
            torch.arange(len(lens), dtype=torch.int32, device=x.device),
            torch.tensor(list(lens), device=x.device),
        )
        .unsqueeze(0)
        .contiguous()
    )

    def layer(*t: torch.Tensor) -> tuple:
        y = causal_conv1d_fn(
            t[0].unsqueeze(0).transpose(1, 2),
            t[1],
            _bias(t),
            seq_idx=seq_idx,
            activation="silu",
        )
        return (y.transpose(1, 2).squeeze(0),)

    return layer


def _cudnn_layer(cu: torch.Tensor) -> tuple[Callable[..., tuple], Callable[[], Any]]:
    ops = importlib.import_module("cudnn.ops.causal_conv1d")

    def layer(*t: torch.Tensor) -> tuple:
        y = ops.causal_conv1d(
            t[0].unsqueeze(0).transpose(1, 2), t[1], _bias(t), "silu", cu_seqlens=cu
        )
        return (y.transpose(1, 2).squeeze(0),)

    return layer, ops._get_causal_conv1d_last_route


def single_rank_arm(
    name: str,
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    dy: torch.Tensor,
    cu: torch.Tensor,
    lens: Sequence[int],
    cake_preactivation: str,
    reference: Sequence[Optional[torch.Tensor]],
) -> tuple[dict, Optional[dict[str, Callable[[], object]]]]:
    """``(entry, calls)`` of one single-rank arm on a row: ``entry`` describes it (library, version, determinism,
    features, numerics and ``env``, the environment to hold around its calls; or ``unsupported`` with the reason);
    ``calls`` (``None`` when unsupported) are its :func:`autograd_calls`, recorded once.  ``reference`` is the Cake
    non-deterministic ``(y, dx, dweight, dbias)`` of the row."""

    entry: dict = {"name": name}
    try:
        env: dict[str, Optional[str]] = {}
        route_of: Optional[Callable[[], Any]] = None
        if name in ("fla", "fla_gluon"):
            entry.update(
                library=_library("fla", "flash-linear-attention"),
                entry=FLA_ENTRY + (" [Gluon]" if name == "fla_gluon" else " [Triton]"),
                determinism="deterministic (FP32 per-chunk partials, ordered reduce)",
                preactivation="bf16",
            )
            env = _FLA_GLUON_ENV if name == "fla_gluon" else _FLA_TRITON_ENV
            if name == "fla_gluon":
                _gluon_verdict(x, w, cu)
            layer = _fla_layer(cu)
        elif name in ("dao", "dao_det"):
            entry.update(
                library=_library("causal_conv1d", "causal-conv1d"),
                entry=DAO_ENTRY + " (channel-last view, seq_idx)",
                determinism="deterministic (CAUSAL_CONV1D_DETERMINISTIC=1)"
                if name == "dao_det"
                else "non-deterministic (FP32 atomic dweight / dbias)",
                preactivation="fp32",
            )
            if b is not None and b.dtype != w.dtype:
                raise Unsupported(
                    f"causal_conv1d_fn needs the bias dtype to equal the weight dtype "
                    f"({dtype_name(b.dtype)} bias, {dtype_name(w.dtype)} taps)"
                )
            env = _DAO_DET_ENV if name == "dao_det" else _DAO_ENV
            layer = _dao_layer(x, lens)
        elif name == "cudnn_fe":
            entry.update(
                library=_library("cudnn", "nvidia-cudnn-frontend"),
                cutlass_dsl=_distribution("nvidia-cutlass-dsl"),
                entry=CUDNN_ENTRY + " (channel-last view, cu_seqlens)",
                determinism="non-deterministic (atomic dweight; torch deterministic algorithms off)",
                preactivation="fp32",
            )
            arch = importlib.import_module("cudnn._causal_conv1d_arch")
            capability = tuple(torch.cuda.get_device_capability(x.device))
            if not arch.is_functional_arch(capability):
                raise Unsupported(
                    f"compute capability {capability[0]}.{capability[1]} is not in cuDNN-FE's causal_conv1d "
                    "FUNCTIONAL_COMPUTE_CAPABILITIES"
                )
            if b is not None and w.dtype != torch.bfloat16:
                raise Unsupported(
                    "cuDNN-FE's packed route takes a bias only with BF16 taps (FP32 taps: bias-free only)"
                )
            if b is not None and b.dtype != torch.bfloat16:
                raise Unsupported("cuDNN-FE's packed route takes a BF16 bias only")
            layer, route_of = _cudnn_layer(cu)
        else:
            raise ValueError(f"unknown single-rank arm {name!r}")
        entry["feature"] = feature(
            preactivation=entry["preactivation"],
            taps=w.dtype,
            bias=None if b is None else b.dtype,
            cake_preactivation=cake_preactivation,
            cake_taps=w.dtype,
        )
        leaves = [
            t.detach().clone().requires_grad_(True) for t in (x, w, b) if t is not None
        ]
        entry["env"] = env
        calls = autograd_calls(layer, leaves, [dy])
        with environment(env):
            (y,) = calls["record"]()
            if route_of is not None:
                entry["route"] = route_of()
            grads = calls["bwd"]()
            torch.cuda.synchronize(x.device)
        names = ("y", "dx", "dweight", "dbias")
        outs = [y, *grads]
        entry["numerics"] = _numerics(
            {
                key: _rel(_sums(out, ref))
                for key, out, ref in zip(names, outs, reference, strict=False)
                if out is not None and ref is not None
            }
        )
        entry["eligible"] = (
            entry["feature"]["match"] in ("exact", "more_precise")
            and entry["numerics"]["ok"]
        )
        return entry, calls
    except Unsupported as exc:
        entry["unsupported"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
        entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    entry["eligible"] = False
    return entry, None


# ---------------------------------------------------------------------------------------------------------------------
# Context parallel
# ---------------------------------------------------------------------------------------------------------------------


def merged_fragments(
    cu: Sequence[int], fragments: Sequence[Any]
) -> list[tuple[int, int]]:
    """A rank's fragments (``doc`` / ``off`` / ``length``) as half-open global token ranges, adjacent ones merged,
    in span order."""

    out: list[tuple[int, int]] = []
    for f in fragments:
        if f.length <= 0:
            continue
        start = cu[f.doc] + f.off
        stop = start + f.length
        if out and out[-1][1] == start:
            out[-1] = (out[-1][0], stop)
        else:
            out.append((start, stop))
    return out


def _take(t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``t[rows]`` with zero rows where ``rows`` is past the end of ``t`` (padding)."""

    rows = rows.to(t.device)
    valid = rows < t.shape[0]
    out = t.new_zeros((rows.numel(), *t.shape[1:]))
    out[valid] = t[rows[valid]]
    return out.contiguous()


def kcp_arm(
    name: str,
    *,
    xs: Sequence[torch.Tensor],
    ws: Sequence[torch.Tensor],
    dys: Sequence[torch.Tensor],
    cu: Sequence[int],
    fragments: Sequence[Sequence[tuple[int, int]]],
    layout: str,
    group: Any,
    rank: int,
    world: int,
) -> tuple[dict, Optional[tuple[dict[str, Callable[[], object]], torch.Tensor]]]:
    """``(entry, (calls, rows))`` of one context-parallel arm on a row: ``xs`` / ``dys`` are the whole pack's
    tensors (every rank holds them), ``ws`` the FP32 taps, ``fragments`` every rank's global token ranges in span
    order (the benchmark's partition).  ``rows`` maps the arm's local rows to global tokens (past the pack: FLA's
    padding).  ``entry`` describes the arm or carries ``unsupported`` with the reason; then the second item is
    ``None``."""

    entry: dict = {"name": name}
    device = xs[0].device
    total = int(cu[-1])
    try:
        if name in ("fla_cp_triton", "fla_cp_cuda"):
            backend = name.removeprefix("fla_cp_")
            entry.update(
                library=_library("fla", "flash-linear-attention"),
                entry=f"{FLA_ENTRY}(cp_context=build_cp_context(...), backend={backend!r})",
                determinism="deterministic (FP32 per-chunk partials, ordered reduce)"
                if backend == "triton"
                else "non-deterministic (Dao kernel, FP32 atomic dweight)",
                preactivation="bf16" if backend == "triton" else "fp32",
                taps_used="float32",
            )
            if layout not in ("contiguous", "zigzag"):
                raise Unsupported(
                    f"FLA CP partitions are contiguous or zigzag; no {layout} form"
                )
            if backend == "cuda":
                if layout != "contiguous":
                    raise Unsupported(
                        "FLA's CUDA CP backend takes the contiguous layout only"
                    )
                entry["dao"] = _library("causal_conv1d", "causal-conv1d")
            from fla.modules.causal_conv1d import causal_conv1d  # type: ignore
            from fla.ops.cp.context import build_cp_context  # type: ignore

            pad = (-total) % (2 * world)
            bounds = list(cu) + ([total + pad] if pad else [])
            cu_cpu = torch.tensor(bounds, dtype=torch.long)
            ctx = build_cp_context(
                cu_cpu.to(device),
                group,
                conv1d_kernel_size=WIDTH,
                cu_seqlens_cpu=cu_cpu,
                layout=layout,
            )
            padded = bounds[-1]
            if layout == "contiguous":
                share = padded // world
                rows = torch.arange(rank * share, (rank + 1) * share)
            else:
                piece = padded // (2 * world)
                rows = torch.cat(
                    [
                        torch.arange(rank * piece, (rank + 1) * piece),
                        torch.arange(
                            (2 * world - 1 - rank) * piece, (2 * world - rank) * piece
                        ),
                    ]
                )
            local_tokens = int(ctx.cu_seqlens_cpu[-1])
            if rows.numel() != local_tokens:
                raise Unsupported(
                    f"FLA's local token count {local_tokens} differs from the {layout} shard {rows.numel()}"
                )
            entry["padding_tokens"] = pad
            leaves = [_take(x, rows).requires_grad_(True) for x in xs] + [
                w.detach().clone().requires_grad_(True) for w in ws
            ]
            local_dys = [_take(dy, rows) for dy in dys]

            def layer(*t: torch.Tensor) -> tuple:
                n = len(t) // 2
                return tuple(
                    causal_conv1d(
                        x.unsqueeze(0),
                        w,
                        activation="silu",
                        cp_context=ctx,
                        backend=backend,
                    )[0].squeeze(0)
                    for x, w in zip(t[:n], t[n:], strict=True)
                )

        elif name == "attn_gym_cp":
            entry.update(
                library=_library("attn_gym", "attention-gym"),
                cutlass_dsl=_distribution("nvidia-cutlass-dsl"),
                entry=ATTN_GYM_ENTRY,
                determinism="deterministic (FP32 partials, torch sum)",
                preactivation="fp32",
                taps_used="bfloat16",
            )
            from attn_gym.linear.context_parallel import (  # type: ignore
                ContextParallelRouting,
                context_parallel_conv_history,
            )
            from attn_gym.linear.short_conv import causal_conv1d as short_conv  # type: ignore

            if any(not owned for owned in fragments):
                raise Unsupported(
                    "a rank owns no tokens (Attention Gym rejects empty ranks)"
                )
            routing = ContextParallelRouting.from_fragments(
                list(cu), fragments, rank, device, conv_history=WIDTH - 1
            )
            rows = torch.cat(
                [torch.arange(start, stop) for start, stop in fragments[rank]]
            )
            leaves = [_take(x, rows).requires_grad_(True) for x in xs] + [
                w.detach().to(torch.bfloat16).requires_grad_(True) for w in ws
            ]
            local_dys = [_take(dy, rows) for dy in dys]

            def layer(*t: torch.Tensor) -> tuple:
                n = len(t) // 2
                out = []
                for x, w in zip(t[:n], t[n:], strict=True):
                    x3 = x.unsqueeze(0)
                    history = context_parallel_conv_history(x3, routing, group)
                    out.append(
                        short_conv(
                            x3,
                            w,
                            activation="silu",
                            cu_seqlens=routing.cu_seqlens,
                            initial_state=history,
                        ).squeeze(0)
                    )
                return tuple(out)

        else:
            raise ValueError(f"unknown KCP arm {name!r}")
        taps = torch.bfloat16 if entry["taps_used"] == "bfloat16" else torch.float32
        entry["feature"] = feature(
            preactivation=entry["preactivation"],
            taps=taps,
            bias=None,
            cake_preactivation="bf16",
            cake_taps=torch.float32,
        )
        return entry, (autograd_calls(layer, leaves, local_dys), rows)
    except Unsupported as exc:
        entry["unsupported"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
        entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return entry, None


def kcp_numerics(
    calls: dict[str, Callable[[], object]],
    rows: torch.Tensor,
    references: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    all_reduce: Callable[[torch.Tensor], torch.Tensor],
) -> dict:
    """One recorded forward and backward of a KCP arm against the whole-pack Cake non-deterministic reference
    ``(y, dx, dweight)`` per convolution: ``y`` and ``dx`` on the arm's own rows (padding excluded), ``dweight`` as
    the sum of the ranks' partials; the squared sums are reduced over the ranks (``all_reduce``: a sum)."""

    ys = calls["record"]()
    grads = calls["bwd"]()
    n = len(references)
    dxs, dws = grads[:n], grads[n:]
    device = ys[0].device
    rows = rows.to(device)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    dw_rel = 0.0
    for (y_ref, dx_ref, dw_ref), y, dx, dw in zip(
        references, ys, dxs, dws, strict=True
    ):
        valid = rows < y_ref.shape[0]
        index = rows[valid]
        for slot, (out, ref) in enumerate(((y, y_ref), (dx, dx_ref))):
            err, norm = _sums(out[valid], ref[index])
            totals[2 * slot] += err
            totals[2 * slot + 1] += norm
        dw_sum = all_reduce(dw.detach().double().clone())
        dw_rel = max(dw_rel, _rel(_sums(dw_sum, dw_ref)))
    totals = all_reduce(totals)
    values = totals.tolist()
    return _numerics(
        {
            "y": _rel((values[0], values[1])),
            "dx": _rel((values[2], values[3])),
            "dweight": dw_rel,
        }
    )


def best_baseline(
    arms: dict[str, dict], nondet: dict, keys: Sequence[str]
) -> tuple[dict, dict]:
    """``(best, speedup)``: per ``{phase}_{kind}`` key the fastest eligible arm (``{"arm", "us"}``) and its time
    over the non-deterministic Cake time (above 1: Cake faster).  Every timed arm also gets its own ``speedup``."""

    best: dict = {}
    for key in keys:
        timed = [
            (entry[f"{key}_us"], name)
            for name, entry in arms.items()
            if entry.get("eligible") and f"{key}_us" in entry
        ]
        if timed:
            us, name = min(timed)
            best[key] = {"arm": name, "us": us}
    for entry in arms.values():
        entry["speedup"] = {
            key: entry[f"{key}_us"] / nondet[f"{key}_us"]
            for key in keys
            if f"{key}_us" in entry and f"{key}_us" in nondet
        }
    speedup = {
        key: value["us"] / nondet[f"{key}_us"]
        for key, value in best.items()
        if f"{key}_us" in nondet
    }
    return best, speedup


def describe(entry: dict) -> str:
    """One line per arm: library, determinism, match and numerics, or the unsupported reason."""

    if "unsupported" in entry:
        return f"{entry['name']}: unsupported ({entry['unsupported']})"
    numerics = entry.get("numerics", {})
    rel = " ".join(f"{k} {v:.1e}" for k, v in numerics.get("rel_l2", {}).items())
    route = f" route {entry['route']}" if entry.get("route") else ""
    notes = "; ".join(entry["feature"]["notes"])
    return (
        f"{entry['name']}: {entry['library']}, {entry['determinism']}, match {entry['feature']['match']}"
        + (f" ({notes})" if notes else "")
        + f", numerics {'ok' if numerics.get('ok') else 'FAIL'} [{rel}]{route}"
        + ("" if entry.get("eligible") else ", not eligible")
    )


def format_sota(result: dict, head: str, phases: Sequence[str], kind: str) -> list[str]:
    """Per phase every baseline's ``kind`` time (us) and the best eligible one over the non-deterministic time;
    then one line per baseline: library, determinism, match, numerics, or the reason it cannot run the row."""

    arms = result.get("sota")
    if not arms:
        return []
    lines = []
    for phase in phases:
        key = f"{phase}_{kind}"
        cells = " | ".join(
            f"{name} {entry[f'{key}_us']:9.2f}" if f"{key}_us" in entry else f"{name} -"
            for name, entry in arms.items()
        )
        best = result["best_sota"].get(key)
        tail = (
            "best -"
            if best is None
            else f"best {best['arm']} {best['us']:9.2f} -> sota/nondet {result['speedup_vs_sota'][key]:5.3f}x"
        )
        lines.append(f"{head} {phase:4s} sota {kind} | {cells} | {tail}")
    lines.extend(f"{head} sota {describe(entry)}" for entry in arms.values())
    return lines


def ineligible_reason(entry: dict) -> Optional[str]:
    """Why an arm is not a candidate for the best baseline of a row (``None``: it is one)."""

    if "unsupported" in entry:
        return entry["unsupported"]
    reasons = []
    if entry["feature"]["match"] not in ("exact", "more_precise"):
        reasons.append(
            f"feature mismatch ({entry['feature']['match']}: "
            + "; ".join(entry["feature"]["notes"])
            + ")"
        )
    numerics = entry.get("numerics", {})
    if not numerics.get("ok"):
        reasons.append(
            f"numerics above the {NUMERICS_REL_L2:g} relative L2 budget {numerics.get('rel_l2')}"
        )
    return "; ".join(reasons) or None


def comparison(result: dict, phases: Sequence[str], column: str, rule: str) -> dict:
    """The two-baseline comparison of a row on one timing ``column`` (``gpu``: the CUPTI cold-L2 span; ``ahead``:
    the GPU time with the host enqueue hidden): per phase the deterministic and non-deterministic Cake times, the
    best eligible baseline (or ``unsupported`` with every arm's reason) and the ratios ``det / nondet`` and
    ``sota / nondet`` (above 1: the non-deterministic Cake mode is faster); every arm is listed with its time and,
    when it cannot be the best, the reason."""

    det, nondet = result["modes"]["det"], result["modes"]["nondet"]
    arms = result.get("sota", {})
    reasons = {name: ineligible_reason(entry) for name, entry in arms.items()}
    excluded = "; ".join(
        f"{name}: {reason}" for name, reason in reasons.items() if reason
    )
    out: dict = {"column": column, "rule": rule, "phases": {}}
    for phase in phases:
        key = f"{phase}_{column}"
        row: dict = {
            "det_us": det.get(f"{key}_us"),
            "nondet_us": nondet.get(f"{key}_us"),
            "arms": {
                name: {
                    "us": entry.get(f"{key}_us"),
                    "eligible": reasons[name] is None,
                    **({"excluded": reasons[name]} if reasons[name] else {}),
                }
                for name, entry in arms.items()
            },
        }
        if row["det_us"] is not None and row["nondet_us"]:
            row["det_over_nondet"] = row["det_us"] / row["nondet_us"]
        best = result.get("best_sota", {}).get(key)
        if best is not None and row["nondet_us"]:
            row["sota"] = {
                "arm": best["arm"],
                "library": arms[best["arm"]].get("library"),
                "determinism": arms[best["arm"]].get("determinism"),
                "match": arms[best["arm"]]["feature"]["match"],
                "us": best["us"],
                "sota_over_nondet": best["us"] / row["nondet_us"],
            }
        else:
            row["sota"] = {"unsupported": excluded or "no non-Cake baseline requested"}
        out["phases"][phase] = row
    return out


def format_comparison(result: dict, head: str) -> list[str]:
    """One line per phase of the row's two-baseline comparison: the timing column, det, nondet, the best eligible
    baseline and the two ratios, or ``unsupported: <reason>``."""

    block = result.get("comparison")
    if not block:
        return []
    lines = []
    for phase, row in block["phases"].items():
        det, nondet = row["det_us"], row["nondet_us"]
        cake = (
            f"det {det:9.2f} nondet {nondet:9.2f} det/nondet {row['det_over_nondet']:5.3f}x"
            if det is not None and nondet
            else "det - nondet -"
        )
        sota = row["sota"]
        best = (
            f"unsupported: {sota['unsupported']}"
            if "unsupported" in sota
            else f"best {sota['arm']} ({sota['match']}) {sota['us']:9.2f} sota/nondet {sota['sota_over_nondet']:5.3f}x"
        )
        lines.append(f"{head} {phase:4s} compare [{block['column']}] {cake} | {best}")
    return lines


def parse_arms(values: Sequence[str], known: Sequence[str]) -> list[str]:
    """``--sota`` values: ``all``, ``none`` or arm names."""

    if not values or list(values) == ["all"]:
        return list(known)
    if list(values) == ["none"]:
        return []
    unknown = [v for v in values if v not in known]
    if unknown:
        raise SystemExit(f"unknown --sota arm(s) {unknown}; known: {list(known)}")
    return list(values)
