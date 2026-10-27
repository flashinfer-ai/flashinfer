#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import tracer for the gated attention block's SERVING forms (the closure lane of the FlashInfer Qwen 3.5 block draft).

Runs ONE process per invocation: installs (1) a sys.meta_path recorder that logs every module import with the module that
triggered it (the import graph), (2) a recording proxy in front of ``cudnn._pybind_module`` (the nanobind extension) so every
attribute the Python route touches on it is logged with its caller, (3) a ``builtins.open`` hook that logs every file under the
``cudnn`` package read by path (kernel templates loaded through ``template_loader`` / rendered by the GEMM compiler, JSON tables),
and (4) hooks on ``pygraph._lower_to_cpp`` / ``cudnn.create_handle`` so a C++ lowering or a cuDNN handle creation on the route is
a recorded FACT, not a guess.  Then it declares the requested serving form(s) the way ``test_block_serving_{decode,extend}.py``
do, and -- with ``--mode execute`` on a Rubin device -- runs check_support / compile / execute / synchronize, logging the stage
and the wall time of each (a compile budget is NOT a kernel timeout: the stage line says which one ran out).

    trace_closure.py --form {prefill,extend,decode,decode_qsa,all} --mode {declare,execute} --device {cpu,cuda} --out <json> [--frost-flag unset|keep]

``decode_qsa`` (the Qwen 3.8 Flash-Next decode route): the in-block indexer's DECODE step over the pools -- ``QsaSpec(index_source=
"indexer", index_band=True)`` with the two paged side caches (the compressed-key cache on the main block table, the raw-key ring) --
declared FOUR ways so every lazily imported module of the route is traced: the fused prepare launch with the position-table RoPE arm
and the ring rows from the stack's own slot mapping (the serving form), the same with the rows derived in-kernel, and the unfused
twin (per-token cos / sin) with and without the slot mapping; ``--mode execute`` runs all four on the device.

The JSON holds: the ordered module list with files + sizes, the import edges (importer -> imported) restricted to the
``cudnn`` package, every third-party top-level package touched, the data files opened under the package, the extension
attributes accessed (with the calling module and the stage), the lowering / handle facts, and the per-stage timings or errors.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.abc
import importlib.machinery
import json
import os
import sys
from typing import Any
import time
import traceback

T0 = time.time()
STAGE = {"name": "startup"}


def log(msg):
    print(f"[{time.time() - T0:8.2f}s] [{STAGE['name']}] {msg}", flush=True)


# --------------------------------------------------------------------------- (1) the import recorder
IMPORT_EVENTS = []  # (module name, origin, importer module name, stage)
MODULE_ORDER = []


def _importer_name():
    """The nearest calling module (by frame globals) outside importlib and this tracer."""
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        name = f.f_globals.get("__name__", "?")
        if (
            "importlib" not in fn
            and not fn.startswith("<frozen")
            and fn != __file__
            and name not in ("traceback", "linecache")
        ):
            return name
        f = f.f_back
    return "?"


class _Recorder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # ask the remaining finders; record the result; never claim the module
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                origin = getattr(spec, "origin", None)
                IMPORT_EVENTS.append(
                    (
                        fullname,
                        origin,
                        _importer_name() if fullname.startswith("cudnn") else "",
                        STAGE["name"],
                    )
                )
                MODULE_ORDER.append(fullname)
                return spec
        return None


sys.meta_path.insert(0, _Recorder())

# --------------------------------------------------------------------------- (3) the open() hook (data files under the package)
OPENED = []
_real_open = builtins.open


def _hook_open(file, *a, **k):
    try:
        p = os.fspath(file)
        if isinstance(p, bytes):
            p = p.decode()
        if "/python/cudnn/" in p.replace("\\", "/") and "__pycache__" not in p:
            OPENED.append((p, STAGE["name"], _importer_name()))
    except Exception:  # noqa: BLE001
        pass
    return _real_open(file, *a, **k)


builtins.open = _hook_open

# also catch pathlib reads (template_loader uses Path.read_bytes -> io.open; pathlib's open goes through io.open, not builtins.open)
import io  # noqa: E402

_real_io_open = io.open


def _hook_io_open(file, *a, **k):
    try:
        p = os.fspath(file)
        if isinstance(p, bytes):
            p = p.decode()
        if "/python/cudnn/" in p.replace("\\", "/") and "__pycache__" not in p:
            OPENED.append((p, STAGE["name"], _importer_name()))
    except Exception:  # noqa: BLE001
        pass
    return _real_io_open(file, *a, **k)


io.open = _hook_io_open
import pathlib  # noqa: E402

_real_path_open = pathlib.Path.open


def _hook_path_open(self, *a, **k):
    try:
        p = str(self)
        if "/python/cudnn/" in p and "__pycache__" not in p:
            OPENED.append((p, STAGE["name"], _importer_name()))
    except Exception:  # noqa: BLE001
        pass
    return _real_path_open(self, *a, **k)


pathlib.Path.open = _hook_path_open  # type: ignore[method-assign]

# --------------------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument(
    "--form",
    default="all",
    choices=["prefill", "extend", "decode", "decode_qsa", "all"],
)
ap.add_argument("--mode", default="declare", choices=["declare", "execute"])
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
ap.add_argument("--out", required=True)
ap.add_argument(
    "--frost-flag",
    default="unset",
    choices=["unset", "keep"],
    help="CUDNN_FRONTEND_ENABLE_FROST_ENGINES: unset it (a serving stack's process) or keep the environment's",
)
ap.add_argument("--fuse-norm-rope", default="1")
ap.add_argument("--fuse-gate", default="1")
args = ap.parse_args()
if args.frost_flag == "unset":
    os.environ.pop("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", None)
FROST_FLAG = os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES")
log(
    f"python {sys.version.split()[0]} CUDNN_FRONTEND_ENABLE_FROST_ENGINES={FROST_FLAG!r} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
)

STAGE["name"] = "import_torch"
import torch  # noqa: E402

log(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")

STAGE["name"] = "import_cudnn"
import cudnn  # noqa: E402  (the package __init__ loads the nanobind extension: unavoidable in the REAL package)

log(f"cudnn {cudnn.__version__} {cudnn.__file__}")

# --------------------------------------------------------------------------- (2) the extension proxy
NATIVE_ATTRS = []  # (attr, caller module, stage)
_real_pybind = cudnn._pybind_module


class _PybindProxy:
    """Records every attribute the Python route reads off the nanobind extension, then forwards."""

    def __getattr__(self, name):
        if name.startswith("__"):
            return getattr(_real_pybind, name)
        NATIVE_ATTRS.append((name, _importer_name(), STAGE["name"]))
        return getattr(_real_pybind, name)


_proxy = _PybindProxy()
cudnn._pybind_module = _proxy
# `from cudnn import _pybind_module` binds the package attribute; `from ._compiled_module import ...` the module
sys.modules["cudnn._compiled_module"] = _proxy  # type: ignore[assignment]
# modules already imported by cudnn/__init__ that bound the real module at import time: re-point the ones the route can reach
for _m in ("cudnn._device", "cudnn.datatypes"):
    mod = sys.modules.get(_m)
    if mod is not None and hasattr(mod, "_pybind_module"):
        mod._pybind_module = _proxy  # type: ignore[attr-defined]

# --------------------------------------------------------------------------- (4) lowering / handle facts
FACTS: dict[str, Any] = {
    "lower_to_cpp_calls": 0,
    "create_handle_calls": 0,
    "lower_to_cpp_stages": [],
    "create_handle_stages": [],
}
from cudnn import _pygraph as _pg  # noqa: E402

_real_lower = _pg.pygraph._lower_to_cpp


def _hook_lower(self):
    FACTS["lower_to_cpp_calls"] += 1
    FACTS["lower_to_cpp_stages"].append(STAGE["name"])
    log("FACT: pygraph._lower_to_cpp() CALLED (C++ lowering of the GEMM graph)")
    return _real_lower(self)


_pg.pygraph._lower_to_cpp = _hook_lower
_real_create_handle = cudnn.create_handle


def _hook_create_handle(*a, **k):
    FACTS["create_handle_calls"] += 1
    FACTS["create_handle_stages"].append(STAGE["name"])
    log("FACT: cudnn.create_handle() CALLED")
    return _real_create_handle(*a, **k)


cudnn.create_handle = _hook_create_handle

# --------------------------------------------------------------------------- the block
STAGE["name"] = "import_block"
from cudnn.gated_attention_block import (  # noqa: E402
    KV_SOURCE_PAGED,
    GatedAttentionBlockFwd,
    GatedAttentionBlockGeometry,
    QsaSpec,
    decode_route,
    qkvg_from_vllm,
)

log("block imported")
DEV = torch.device(args.device)
DT = torch.bfloat16
D, ROPE, D_MODEL = 256, 64, 512
H_Q, H_KV = (
    24,
    2,
)  # Flash-Next TP1 heads at a shrunk d_model (the serving tests' matrix)
PAGE = 16
FUSE_NR = args.fuse_norm_rope == "1"
FUSE_GATE = args.fuse_gate == "1"
RESULT = {
    "args": vars(args),
    "frost_flag": FROST_FLAG,
    "forms": {},
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cudnn_file": cudnn.__file__,
}


def geom():
    return GatedAttentionBlockGeometry(
        d_model=D_MODEL,
        h_q=H_Q,
        h_kv=H_KV,
        d_head=D,
        rope_dim=ROPE,
        causal_bottom_right=True,
    )


def rope_tables(positions):  # [B, S] int positions -> cos, sin [B, S, ROPE]
    inv = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, ROPE, 2, device=DEV, dtype=torch.float32) / ROPE)
    )
    f = positions.to(DEV, torch.float32)[..., None] * inv
    e = torch.cat((f, f), dim=-1)
    return e.cos().to(DT), e.sin().to(DT)


def weights(g, gen):
    rnd = lambda *s, std=0.02: (
        torch.randn(*s, generator=gen, device=DEV, dtype=torch.float32) * std
    ).to(DT)  # noqa: E731
    return dict(
        w_qkvg=rnd(g.n_qkvg, D_MODEL),
        w_q_norm=torch.ones(D, dtype=DT, device=DEV),
        w_k_norm=torch.ones(D, dtype=DT, device=DEV),
        w_o=rnd(D_MODEL, H_Q * D),
    )


def pools(n_pages):
    return tuple(
        torch.zeros(n_pages, H_KV, PAGE, D, dtype=DT, device=DEV) for _ in range(2)
    )


def run_stages(name, blk, exec_args, exec_kwargs, out):
    rec = RESULT["forms"][name]
    rec["stages"] = {}
    if args.mode != "execute":
        return
    for st in ("check_support", "compile", "workspace", "execute", "synchronize"):
        STAGE["name"] = f"{name}.{st}"
        t = time.time()
        try:
            if st == "check_support":
                blk.check_support()
            elif st == "compile":
                blk.compile()
            elif st == "workspace":
                n = blk.get_workspace_size()
                rec["workspace_bytes"] = int(n)
                exec_args.append(torch.zeros(n, dtype=torch.uint8, device=DEV))
            elif st == "execute":
                blk.execute(*exec_args, **exec_kwargs)
            else:
                torch.cuda.synchronize()
                rec["out_finite"] = bool(torch.isfinite(out.float()).all().item())
            rec["stages"][st] = {"ok": True, "s": round(time.time() - t, 3)}
            log(f"{st}: ok in {time.time() - t:.2f}s")
        except Exception as exc:  # noqa: BLE001
            rec["stages"][st] = {
                "ok": False,
                "s": round(time.time() - t, 3),
                "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            }
            log(f"{st}: FAILED {type(exc).__name__}: {str(exc)[:600]}")
            traceback.print_exc()
            return
    # a second execute (the cached path) -- the per-launch imports, if any
    STAGE["name"] = f"{name}.execute2"
    try:
        blk.execute(*exec_args, **exec_kwargs)
        torch.cuda.synchronize()
        rec["stages"]["execute2"] = {"ok": True}
        log("execute2: ok")
    except Exception as exc:  # noqa: BLE001
        rec["stages"]["execute2"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:600]}",
        }
        log(f"execute2: FAILED {type(exc).__name__}: {str(exc)[:600]}")


def declare(name, fn):
    STAGE["name"] = f"{name}.declare"
    t = time.time()
    RESULT["forms"][name] = {}
    try:
        blk, exec_args, exec_kwargs, out = fn()
        RESULT["forms"][name]["declared"] = True
        RESULT["forms"][name]["declare_s"] = round(time.time() - t, 3)
        RESULT["forms"][name]["stage_list"] = [
            type(s).__name__ for s in getattr(blk, "_stages", ())
        ]
        RESULT["forms"][name]["facts"] = {
            k: str(getattr(blk, k, None))
            for k in (
                "kv_source",
                "thd",
                "token_capacity",
                "fuse_gate",
                "fuse_norm_rope",
                "cache_write_fused",
                "gate_fused",
                "paged_kv_page_size",
                "sparse_tile_form",
                "sparse_tile",
                "decode_split_kv",
            )
        }
        log(
            f"declared in {time.time() - t:.2f}s; stages {RESULT['forms'][name]['stage_list']}"
        )
    except Exception as exc:  # noqa: BLE001
        RESULT["forms"][name]["declared"] = False
        RESULT["forms"][name]["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
        log(f"DECLARATION FAILED {type(exc).__name__}: {str(exc)[:800]}")
        traceback.print_exc()
        return
    run_stages(name, blk, exec_args, exec_kwargs, out)


def form_prefill():
    """The dense prefill with write-through: [B, S, d_model], paged_kv_page_size=P, kv_source='self'."""
    g, gen = geom(), torch.Generator(device=DEV).manual_seed(1)
    B, S = 2, 64
    w = weights(g, gen)
    h = (torch.randn(B, S, D_MODEL, generator=gen, device=DEV, dtype=torch.float32)).to(
        DT
    )
    cos, sin = rope_tables(torch.arange(S, device=DEV).expand(B, S))
    out = torch.empty(B, S, D_MODEL, dtype=DT, device=DEV)
    n_pages = (B * S + PAGE - 1) // PAGE + 1
    kc, vc = pools(n_pages)
    slot = torch.arange(B * S, dtype=torch.int32, device=DEV)
    blk = GatedAttentionBlockFwd(
        h,
        w["w_qkvg"],
        w["w_q_norm"],
        w["w_k_norm"],
        cos,
        sin,
        w["w_o"],
        out,
        g,
        paged_kv_page_size=PAGE,
        fuse_norm_rope=FUSE_NR,
        fuse_gate=FUSE_GATE,
    )
    return (
        blk,
        [h, w["w_qkvg"], w["w_q_norm"], w["w_k_norm"], cos, sin, w["w_o"], out],
        dict(k_cache=kc, v_cache=vc, slot_mapping=slot),
        out,
    )


def form_extend():
    """The EXTEND form: packed, bounded, kv_source='paged' (cu_seqlens prefix sums, block_table + kv_lens)."""
    g, gen = geom(), torch.Generator(device=DEV).manual_seed(2)
    B_MAX, T_MAX, MAX_KV = 2, 64, 64
    max_pages = (MAX_KV + PAGE - 1) // PAGE
    n_pages = B_MAX * max_pages + 1
    w = weights(g, gen)
    h = torch.randn(T_MAX, D_MODEL, generator=gen, device=DEV, dtype=torch.float32).to(
        DT
    )
    # step: seq 0 = 24 new tokens after a 16-token prefix; seq 1 = 20 new tokens, no prefix; 20 pad rows
    q_lens, prefix = [24, 20], [16, 0]
    pos = torch.zeros(1, T_MAX, dtype=torch.int64, device=DEV)
    row = 0
    for q, p in zip(q_lens, prefix, strict=False):
        pos[0, row : row + q] = torch.arange(p, p + q, device=DEV)
        row += q
    cos, sin = rope_tables(pos)
    cos, sin = cos[0].contiguous(), sin[0].contiguous()
    out = torch.empty(T_MAX, D_MODEL, dtype=DT, device=DEV)
    kc, vc = pools(n_pages)
    table = torch.zeros(B_MAX, max_pages, dtype=torch.int32, device=DEV)
    kv_lens = torch.tensor(
        [p + q for q, p in zip(q_lens, prefix, strict=False)],
        dtype=torch.int32,
        device=DEV,
    )
    # pages: seq b takes pages [1 + b*max_pages, ...]
    slots = torch.full((T_MAX,), -1, dtype=torch.int32, device=DEV)
    row = 0
    for b, (q, p) in enumerate(zip(q_lens, prefix, strict=False)):
        pages = [1 + b * max_pages + i for i in range(max_pages)]
        n = (p + q + PAGE - 1) // PAGE
        table[b, :n] = torch.tensor(pages[:n], dtype=torch.int32, device=DEV)
        for j in range(p, p + q):
            slots[row + j - p] = pages[j // PAGE] * PAGE + j % PAGE
        # a random finite prefix in the pools
        for j in range(p):
            kc[pages[j // PAGE], :, j % PAGE] = torch.randn(
                H_KV, D, generator=gen, device=DEV, dtype=torch.float32
            ).to(DT)
            vc[pages[j // PAGE], :, j % PAGE] = torch.randn(
                H_KV, D, generator=gen, device=DEV, dtype=torch.float32
            ).to(DT)
        row += q
    seq_lens = torch.tensor(
        [0, q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int32, device=DEV
    )
    blk = GatedAttentionBlockFwd(
        h,
        w["w_qkvg"],
        w["w_q_norm"],
        w["w_k_norm"],
        cos,
        sin,
        w["w_o"],
        out,
        g,
        thd=True,
        num_sequences=B_MAX,
        max_seq_len=T_MAX,
        cu_seqlens=True,
        token_capacity=T_MAX,
        paged_kv_page_size=PAGE,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=table,
        fuse_norm_rope=FUSE_NR,
        fuse_gate=FUSE_GATE,
    )
    return (
        blk,
        [h, w["w_qkvg"], w["w_q_norm"], w["w_k_norm"], cos, sin, w["w_o"], out],
        dict(
            seq_lens=seq_lens,
            k_cache=kc,
            v_cache=vc,
            slot_mapping=slots,
            block_table=table,
            kv_lens=kv_lens,
        ),
        out,
    )


def form_decode():
    """The DECODE form: dense [B, S_q] over the pools (block_table + kv_lens), S_q = 2 at group 12."""
    g, gen = geom(), torch.Generator(device=DEV).manual_seed(3)
    B, S_Q = 4, 2
    kv_lens_l = [40, 0, 17, 64]
    L_MAX = 64
    max_pages = (L_MAX + PAGE - 1) // PAGE
    n_pages = sum((L + PAGE - 1) // PAGE for L in kv_lens_l) + 3
    w = weights(g, gen)
    h = torch.randn(B, S_Q, D_MODEL, generator=gen, device=DEV, dtype=torch.float32).to(
        DT
    )
    pos = torch.tensor(
        [list(range(L - S_Q, L)) if L else list(range(S_Q)) for L in kv_lens_l],
        device=DEV,
    )
    cos, sin = rope_tables(pos)
    out = torch.empty(B, S_Q, D_MODEL, dtype=DT, device=DEV)
    kc, vc = pools(n_pages)
    table = torch.zeros(B, max_pages, dtype=torch.int32, device=DEV)
    slots = torch.full((B * S_Q,), -1, dtype=torch.int32, device=DEV)
    nxt = 1
    for b, L in enumerate(kv_lens_l):
        if L == 0:
            continue
        n = (L + PAGE - 1) // PAGE
        pages = list(range(nxt, nxt + n))
        nxt += n
        table[b, :n] = torch.tensor(pages, dtype=torch.int32, device=DEV)
        for p in range(L - S_Q):
            kc[pages[p // PAGE], :, p % PAGE] = torch.randn(
                H_KV, D, generator=gen, device=DEV, dtype=torch.float32
            ).to(DT)
            vc[pages[p // PAGE], :, p % PAGE] = torch.randn(
                H_KV, D, generator=gen, device=DEV, dtype=torch.float32
            ).to(DT)
        for i in range(S_Q):
            p = L - S_Q + i
            slots[b * S_Q + i] = pages[p // PAGE] * PAGE + p % PAGE
    kv_lens = torch.tensor(kv_lens_l, dtype=torch.int32, device=DEV)
    blk = GatedAttentionBlockFwd(
        h,
        w["w_qkvg"],
        w["w_q_norm"],
        w["w_k_norm"],
        cos,
        sin,
        w["w_o"],
        out,
        g,
        paged_kv_page_size=PAGE,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=table,
        fuse_norm_rope=FUSE_NR,
        fuse_gate=FUSE_GATE,
    )
    RESULT["forms"]["decode"]["route"] = str(decode_route(g, S_Q))
    return (
        blk,
        [h, w["w_qkvg"], w["w_q_norm"], w["w_k_norm"], cos, sin, w["w_o"], out],
        dict(
            k_cache=kc,
            v_cache=vc,
            block_table=table,
            kv_lens=kv_lens,
            slot_mapping=slots,
        ),
        out,
    )


# --------------------------------------------------------------------------- the host-side API surface a FlashInfer wrapper reads
STAGE["name"] = "host_api"
try:
    g0 = geom()
    env = GatedAttentionBlockFwd.serving_envelope(g0, DT, DEV, device_cc=(10, 7))
    RESULT["serving_envelope_rubin"] = {
        "forms": list(env.forms),
        "extend_served": bool(env.extend.served),
        "decode_served": bool(env.decode.served),
        "cutedsl_ok": bool(env.cutedsl_ok),
    }
    env2 = GatedAttentionBlockFwd.serving_envelope(g0, DT, DEV)
    RESULT["serving_envelope_here"] = {
        "forms": list(env2.forms),
        "device_cc": str(env2.device_cc),
    }
    RESULT["decode_routes"] = {s: str(decode_route(g0, s)) for s in (1, 2, 4, 5)}
    # the vLLM loader twin on host tensors
    q_proj = torch.zeros(H_Q * 2 * D, D_MODEL, dtype=DT, device=DEV)
    k_proj = torch.zeros(H_KV * D, D_MODEL, dtype=DT, device=DEV)
    v_proj = torch.zeros(H_KV * D, D_MODEL, dtype=DT, device=DEV)
    try:
        import inspect

        RESULT["qkvg_from_vllm_signature"] = str(inspect.signature(qkvg_from_vllm))
    except Exception as exc:  # noqa: BLE001
        RESULT["qkvg_from_vllm_signature"] = f"? {exc}"
    log(
        f"serving_envelope(rubin): {RESULT['serving_envelope_rubin']}; here: {RESULT['serving_envelope_here']}"
    )
except Exception as exc:  # noqa: BLE001
    RESULT["host_api_error"] = f"{type(exc).__name__}: {exc}"
    log(f"host api FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def form_decode_qsa(
    *,
    fused: bool,
    tables_only: bool,
    ring_slots: bool,
    batch: int = 4,
    s_q: int = 4,
    sparse_tile_form: str = "auto",
):
    """The in-block indexer's DECODE step over the pools (the Qwen 3.8 Flash-Next decode route): a dense [B, S_q] declaration on a
    ``QsaSpec(index_source="indexer", index_band=True)`` geometry with ``kv_source="paged"`` and the two paged side caches -- the
    compressed-key cache ``[num_pages, P / 4, 1, 128]`` on the MAIN block table and the raw-key ring ``[num_ring_blocks, cap, 1, 128]``
    behind its own ``[B, 1]`` table -- at B 4 x S_q 4 (the MTP verify width; the tcgen05 core) by default, the four input-contract arms
    as the keyword arguments say.  ``batch`` / ``s_q`` (appended) shrink the step to the serving stacks' B 1 cells, where the sparse
    core's body knob ``sparse_tile_form="auto"`` (the FlashInfer default) resolves the LEAN decode tile on cc 10.7 (the placement bound
    ``B x H_kv <= 4`` CTAs per chunk since cuDNN Frontend 977757763 -- ``B x H_kv x S_q <= 4`` units before it -- where the lean body admits
    the step; under the in-block indexer the list is per row, which the lean body serves at one row only): the lean kernel module is
    loaded by path only on that route, so the closure needs a trace of it."""

    def make():
        g0 = GatedAttentionBlockGeometry(
            d_model=D_MODEL,
            h_q=H_Q,
            h_kv=H_KV,
            d_head=D,
            rope_dim=ROPE,
            causal_bottom_right=True,
            qsa=QsaSpec(top_k=512, index_source="indexer", index_band=True),
        )
        gen = torch.Generator(device=DEV).manual_seed(4)
        B, S_Q, CAP, D_I = int(batch), int(s_q), 8, 128
        kv_lens_l = [40, 0, 17, 64][:B] if B <= 4 else [40, 0, 17, 64] + [64] * (B - 4)
        L_MAX = 64
        max_pages = (L_MAX + PAGE - 1) // PAGE
        n_pages = sum((L + PAGE - 1) // PAGE for L in kv_lens_l) + 3
        n_pos = max_pages * PAGE
        w = weights(
            g0, gen
        )  # w_qkvg carries the fifth (indexer) band: g0.n_qkvg includes it
        w_iq = torch.ones(D_I, dtype=DT, device=DEV)
        w_ik = torch.ones(D_I, dtype=DT, device=DEV)
        h = torch.randn(
            B, S_Q, D_MODEL, generator=gen, device=DEV, dtype=torch.float32
        ).to(DT)
        pos_tab = torch.arange(n_pos, device=DEV)[None]
        cos_pos, sin_pos = rope_tables(
            pos_tab
        )  # [1, n_pos, ROPE]: the position-indexed tables (the block starts; every rotation under the table arm)
        pos = torch.tensor(
            [
                list(range(L - S_Q, L)) if L >= S_Q else list(range(S_Q))
                for L in kv_lens_l
            ],
            device=DEV,
        )
        cos, sin = rope_tables(pos)
        out = torch.empty(B, S_Q, D_MODEL, dtype=DT, device=DEV)
        kc, vc = pools(n_pages)
        table = torch.zeros(B, max_pages, dtype=torch.int32, device=DEV)
        slots = torch.full((B * S_Q,), -1, dtype=torch.int64, device=DEV)
        ring_slots_t = torch.full((B * S_Q,), -1, dtype=torch.int64, device=DEV)
        icache = torch.zeros(n_pages, PAGE // 4, 1, D_I, dtype=DT, device=DEV)
        ring = torch.zeros(B + 1, CAP, 1, D_I, dtype=DT, device=DEV)
        ring_table = torch.zeros(B, 1, dtype=torch.int32, device=DEV)
        nxt = 1
        for b, L in enumerate(kv_lens_l):
            if L == 0:
                continue
            n = (L + PAGE - 1) // PAGE
            pages = list(range(nxt, nxt + n))
            nxt += n
            table[b, :n] = torch.tensor(pages, dtype=torch.int32, device=DEV)
            ring_table[b, 0] = b + 1
            for p in range(L - S_Q):
                kc[pages[p // PAGE], :, p % PAGE] = torch.randn(
                    H_KV, D, generator=gen, device=DEV, dtype=torch.float32
                ).to(DT)
                vc[pages[p // PAGE], :, p % PAGE] = torch.randn(
                    H_KV, D, generator=gen, device=DEV, dtype=torch.float32
                ).to(DT)
            for i in range(S_Q):
                p = L - S_Q + i
                if p >= 0:
                    slots[b * S_Q + i] = pages[p // PAGE] * PAGE + p % PAGE
                    ring_slots_t[b * S_Q + i] = (b + 1) * CAP + p % CAP
        kv_lens = torch.tensor(kv_lens_l, dtype=torch.int32, device=DEV)
        tables = (None, None) if tables_only else (cos, sin)
        blk = GatedAttentionBlockFwd(
            h,
            w["w_qkvg"],
            w["w_q_norm"],
            w["w_k_norm"],
            *tables,
            w["w_o"],
            out,
            g0,
            paged_kv_page_size=PAGE,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=kc,
            sample_v_cache=vc,
            sample_block_table=table,
            fuse_norm_rope=False,
            fuse_gate=FUSE_GATE,
            fuse_prepare=fused,
            sample_index_kv_cache=icache,
            sample_index_state_cache=ring,
            sample_index_state_block_table=ring_table,
            index_state_slot_mapping_present=ring_slots,
            rope_position_tables_only=tables_only,
            sparse_tile_form=sparse_tile_form,
        )
        ids_out = torch.full((B, S_Q, 512), -7, dtype=torch.int32, device=DEV)
        lens_out = torch.full((B, S_Q), -7, dtype=torch.int32, device=DEV)
        return (
            blk,
            [h, w["w_qkvg"], w["w_q_norm"], w["w_k_norm"], *tables, w["w_o"], out],
            dict(
                k_cache=kc,
                v_cache=vc,
                block_table=table,
                kv_lens=kv_lens,
                slot_mapping=slots,
                w_iq_norm=w_iq,
                w_ik_norm=w_ik,
                block_ids_out=ids_out,
                block_lens_out=lens_out,
                index_kv_cache=icache,
                index_state_cache=ring,
                index_state_block_table=ring_table,
                index_state_slot_mapping=ring_slots_t if ring_slots else None,
                cos_pos=cos_pos,
                sin_pos=sin_pos,
            ),
            out,
        )

    return make


forms = {"prefill": form_prefill, "extend": form_extend, "decode": form_decode}
# the Qwen 3.8 decode route, four declarations (one per input-contract arm the serving stacks take; all four executed on a device)
QSA_FORMS = {
    "decode_qsa_fused_tables_slots": form_decode_qsa(
        fused=True, tables_only=True, ring_slots=True
    ),
    "decode_qsa_fused_tables": form_decode_qsa(
        fused=True, tables_only=True, ring_slots=False
    ),
    "decode_qsa_unfused_pertoken_slots": form_decode_qsa(
        fused=False, tables_only=False, ring_slots=True
    ),
    "decode_qsa_unfused_pertoken": form_decode_qsa(
        fused=False, tables_only=False, ring_slots=False
    ),
    # the serving stacks' B 1 cells (appended at the lean-tile re-vendor): ``sparse_tile_form="auto"`` resolves the LEAN decode tile on cc 10.7
    # at B 1 x S_q 1 (2 units: the vLLM-shaped step, the fused prepare on the position tables with the ring slots) and at B 1 x S_q 2
    # (4 units, the bound; the per-token unfused arm) -- the lean kernel module and its shared helper enter the closure through these two
    "decode_qsa_lean_b1_sq1_fused_tables_slots": form_decode_qsa(
        fused=True, tables_only=True, ring_slots=True, batch=1, s_q=1
    ),
    "decode_qsa_lean_b1_sq2_unfused_pertoken": form_decode_qsa(
        fused=False, tables_only=False, ring_slots=False, batch=1, s_q=2
    ),
    # the placement bound's cells (appended at the 977757763 re-vendor: the bound counts the CTAs per chunk, ``B x H_kv``, not
    # ``B x H_kv x S_q``): B 2 x S_q 1 = 4 CTAs per chunk, the bound itself -> the lean body; B 4 x S_q 1 = 8 -> the tcgen05 core; B 1 x S_q 4
    # = 2 CTAs per chunk, inside the bound, but under the in-block indexer the list is PER ROW and the lean body serves per-row lists at one
    # row only (the list-form decline, independent of the bound) -> the tcgen05 core.  The three resolutions are recorded as declared facts.
    "decode_qsa_bound_b2_sq1_fused_tables_slots": form_decode_qsa(
        fused=True, tables_only=True, ring_slots=True, batch=2, s_q=1
    ),
    "decode_qsa_bound_b4_sq1_fused_tables_slots": form_decode_qsa(
        fused=True, tables_only=True, ring_slots=True, batch=4, s_q=1
    ),
    "decode_qsa_bound_b1_sq4_fused_tables_slots": form_decode_qsa(
        fused=True, tables_only=True, ring_slots=True, batch=1, s_q=4
    ),
}
if args.form == "all":
    selected = {**forms, **QSA_FORMS}
elif args.form == "decode_qsa":
    selected = QSA_FORMS
else:
    selected = {args.form: forms[args.form]}
for nm, fn in selected.items():
    declare(nm, fn)

# --------------------------------------------------------------------------- the report
STAGE["name"] = "report"
mods: dict[str, Any] = {}
for name, origin, importer, stage in IMPORT_EVENTS:
    m = sys.modules.get(name)
    f = getattr(m, "__file__", None) or origin
    size = None
    try:
        if f and os.path.isfile(f):
            size = os.path.getsize(f)
    except Exception:  # noqa: BLE001
        pass
    mods.setdefault(
        name, {"file": f, "size": size, "first_importer": importer, "stage": stage}
    )
# modules loaded by template_loader (spec_from_file_location) are named cudnn.frost._templates.<tag>_<n>: record their files
templates = {}
for name, m in list(sys.modules.items()):
    if name.startswith("cudnn.frost._templates."):
        templates[name] = getattr(m, "__file__", None)
cudnn_mods = {k: v for k, v in mods.items() if k == "cudnn" or k.startswith("cudnn.")}
third = sorted({k.split(".")[0] for k in mods if not k.startswith("cudnn")})
RESULT.update(
    {
        "n_modules_total": len(mods),
        "cudnn_modules": cudnn_mods,
        "template_modules": templates,
        "third_party_top_level": third,
        "opened_files": sorted({(p, st) for p, st, _ in OPENED}),
        "native_attrs": NATIVE_ATTRS,
        "facts": FACTS,
        "import_edges": sorted(
            {
                (imp, name)
                for name, _o, imp, _s in IMPORT_EVENTS
                if name.startswith("cudnn") and imp
            }
        ),
    }
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with _real_open(args.out, "w") as fh:
    json.dump(RESULT, fh, indent=1, default=str)
log(
    f"wrote {args.out}: {len(cudnn_mods)} cudnn modules, {len(templates)} template modules, {len(third)} third-party packages, {len(NATIVE_ATTRS)} native attribute reads, facts {FACTS}"
)
ok = all(
    f.get("declared") and all(s.get("ok") for s in f.get("stages", {}).values())
    for f in RESULT["forms"].values()
)
print("TRACE-DONE rc=" + ("0" if ok else "1"), flush=True)
sys.exit(0 if ok else 1)
