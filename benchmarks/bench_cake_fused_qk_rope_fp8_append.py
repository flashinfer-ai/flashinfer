# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark: Cake fused QK RMSNorm + NeoX RoPE + FP8 quantize + paged KV append.

Arms (identical inputs, caller-owned buffers re-created per arm):

- ``cake``: :func:`flashinfer.cake_fused_qk_rope_fp8_append.cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache`
  (one launch).
- ``baseline``: the previous Cake route, timed in the same process on the same inputs.
  ``--baseline-ref <git revision>`` stages the route of that FlashInfer revision (its
  public module, JIT loader and generated sources, copied out of the git history into a
  private package; see :func:`stage_previous_cake`), the paired arm of the speedup
  table; ``--baseline-module`` names an already importable module instead (the same
  keyword signature and entry-point name).
- ``stable``: the step-wise pipeline from stable FlashInfer / torch ops
  (``flashinfer.norm.rmsnorm`` -> ``apply_rope_pos_ids_inplace`` -> torch FP8 quantize
  -> ``flashinfer.page.append_paged_kv_cache``), report only: its BF16 intermediates
  cannot meet the FP8 contract tolerance, it does not clear the last-page tail, and it
  does not express ragged requests, caller-owned K/V or a ``max_seqlen`` beyond the
  request length; the rows it covers are declared in the JSON ``arms`` metadata before
  the run.

Timing: ``flashinfer.testing.bench_gpu_time`` (CUPTI, cold L2) over the whole call (the
union of every kernel one public call issues), eager and CUDA-graph replay
(``--graph-instances`` independent captures replayed in rotation), paired A/B/B/A groups;
the median over ``--iters`` of each group, the median of the groups reported.  The SM
clock (MHz) is sampled per row before and after its groups when the driver exposes it.

    python benchmarks/bench_cake_fused_qk_rope_fp8_append.py --rows P_decode_8x1_q1_n2 --json out.json
    python benchmarks/bench_cake_fused_qk_rope_fp8_append.py --baseline-ref <revision> \\
        --groups 3 --iters 300 --warmup 100 --graph-instances 8      # previous Cake route
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

import flashinfer
from flashinfer.cake_fused_qk_rope_fp8_append import (
    cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache,
)
from flashinfer.cake_fused_qk_rope_fp8_append_backend import cake_jit as loader
from flashinfer.testing.utils import bench_gpu_time

HEAD_DIM = 128
FP8_MAX = 448.0
MODES = ("graph", "eager")


@dataclass(frozen=True)
class Row:
    name: str
    hq: int
    hkv: int
    q_lens: tuple[int, ...]
    ctx_lens: tuple[int, ...]
    seed: int
    quant: int = 1
    norm: int = 2
    page_size: int = 64
    prefill: bool = True
    max_seqlen: int = 0
    out_kv: bool = False

    @property
    def batch(self) -> int:
        return len(self.q_lens)

    @property
    def num_rows(self) -> int:
        return sum(self.q_lens)

    @property
    def ragged(self) -> bool:
        return len(set(self.q_lens)) > 1 or len(set(self.ctx_lens)) > 1

    @property
    def padded_max_seqlen(self) -> bool:
        return self.max_seqlen > max(self.q_lens)

    def geometry(self) -> str:
        if self.ragged:
            shape = f"B{self.batch} ragged T{self.num_rows}"
        else:
            shape = f"B{self.batch} Q{self.q_lens[0]} ctx{self.ctx_lens[0]}"
        kind = "prefill" if self.prefill else "decode"
        extra = (f" maxseq{self.max_seqlen}" if self.max_seqlen else "") + (
            " outkv" if self.out_kv else ""
        )
        return f"({self.hq},{self.hkv}) {kind} {shape} q{self.quant} n{self.norm} p{self.page_size}{extra}"


def _row(name, hq, hkv, batch, q, ctx, seed, **extra) -> Row:
    return Row(name, hq, hkv, (q,) * batch, (ctx,) * batch, seed, **extra)


def _gate_rows() -> list[Row]:
    """Group P: decode (B32 Q1) and prefill (B8 Q16, ``max_seqlen`` 16) at both head
    configurations over the quantization policies 1, 2 and the norm policies 0, 1, 2."""
    rows: list[Row] = []
    seed = 893001
    for prefill, batch, q, max_seqlen in ((False, 32, 1, 0), (True, 8, 16, 16)):
        kind = "prefill" if prefill else "decode"
        for hq, hkv in ((8, 1), (64, 8)):
            for quant in (1, 2):
                for norm in (0, 1, 2):
                    rows.append(
                        _row(
                            f"P_{kind}_{hq}x{hkv}_q{quant}_n{norm}",
                            hq,
                            hkv,
                            batch,
                            q,
                            2048,
                            seed,
                            quant=quant,
                            norm=norm,
                            prefill=prefill,
                            max_seqlen=max_seqlen,
                        )
                    )
                    seed += 1
    return rows


def _boundary_rows() -> list[Row]:
    """Group E: prefill B4 at the 128-row alignment boundary of the dynamic Q scale."""
    rows: list[Row] = []
    seed = 893025
    for hq, hkv in ((8, 1), (64, 8)):
        for q in (127, 128, 129):
            rows.append(
                _row(
                    f"E_prefill_{hq}x{hkv}_q{q}",
                    hq,
                    hkv,
                    4,
                    q,
                    2048,
                    seed,
                    max_seqlen=q,
                )
            )
            seed += 1
    return rows


_RAGGED_Q_LENS = (7, 32, 0, 15, 1, 24, 9, 30)
_REPORT_ROWS = (
    _row("X_decode_8x1_B1_q1_n2", 8, 1, 1, 1, 2048, 893031, prefill=False),
    _row("X_decode_8x1_B128_q1_n2", 8, 1, 128, 1, 2048, 893032, prefill=False),
    _row("X_decode_8x1_B256_q1_n2", 8, 1, 256, 1, 2048, 893033, prefill=False),
    _row("X_prefill_8x1_B1_Q1024_q1_n2", 8, 1, 1, 1024, 2048, 893034, max_seqlen=1024),
    _row("X_decode_64x8_B1_q1_n2", 64, 8, 1, 1, 2048, 893035, prefill=False),
    _row("X_decode_64x8_B128_q1_n2", 64, 8, 128, 1, 2048, 893036, prefill=False),
    _row("X_decode_64x8_B256_q1_n2", 64, 8, 256, 1, 2048, 893037, prefill=False),
    _row("X_prefill_64x8_B1_Q1024_q1_n2", 64, 8, 1, 1024, 2048, 893038, max_seqlen=1024),
    _row("X_decode_8x1_q1_n2_page16", 8, 1, 32, 1, 2048, 893039, prefill=False, page_size=16),
    _row("X_decode_8x1_q1_n2_page128", 8, 1, 32, 1, 2048, 893040, prefill=False, page_size=128),
    _row("X_decode_8x1_q1_n2_ctx32768", 8, 1, 32, 1, 32768, 893041, prefill=False),
    Row("X_prefill_8x1_ragged_q1_n2", 8, 1, _RAGGED_Q_LENS, (2048,) * 8, 893042, max_seqlen=32),
    _row("X_decode_8x1_q1_n2_outkv", 8, 1, 32, 1, 2048, 893043, prefill=False, out_kv=True),
    _row("X_prefill_8x1_q1_n2_maxseq1024", 8, 1, 8, 16, 2048, 893044, max_seqlen=1024),
)  # fmt: skip

# The forty-four rows of the kernel's acceptance table: P (24), E (6), X (14); every row is
# timed by every arm that covers it.
ROWS = {r.name: r for r in (*_gate_rows(), *_boundary_rows(), *_REPORT_ROWS)}
assert len(ROWS) == 44


def rotary_cos_sin_table(
    max_positions: int, device, base: float = 10000.0
) -> torch.Tensor:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    pos = torch.arange(max_positions, dtype=torch.float64, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)


def build_inputs(row: Row, device):
    g = torch.Generator(device="cpu").manual_seed(row.seed)
    B = row.batch
    q_lens = list(row.q_lens)
    seq_lens = [c + q for c, q in zip(row.ctx_lens, q_lens, strict=True)]
    pages_per_req = [max(1, (s + row.page_size - 1) // row.page_size) for s in seq_lens]
    max_pages = max(pages_per_req)
    total_pages = sum(pages_per_req) + 4
    perm = torch.randperm(total_pages, generator=g)
    page_indices = torch.zeros((B, max_pages), dtype=torch.int32)
    cursor = 0
    for b in range(B):
        n = pages_per_req[b]
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    T = int(q_indptr[-1])
    width = (row.hq + 2 * row.hkv) * HEAD_DIM
    qkv = (
        torch.randn((T, width), generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    q_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16)
    k_w = (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16)
    positions = torch.cat(
        [
            torch.arange(ctx, ctx + q, dtype=torch.int32)
            for ctx, q in zip(row.ctx_lens, q_lens, strict=True)
        ]
    )
    batch_indices = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32), torch.tensor(q_lens, dtype=torch.int64)
    )
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor(pages_per_req, dtype=torch.int32), 0)
    kv_indices = torch.cat([page_indices[b, : pages_per_req[b]] for b in range(B)])
    kv_last_page_len = torch.tensor(
        [(s - 1) % row.page_size + 1 for s in seq_lens], dtype=torch.int32
    )
    return {
        "row": row,
        "T": T,
        "max_pages": max_pages,
        "is_prefill": row.prefill,
        "max_seqlen": row.max_seqlen,
        "qkv": qkv,
        "cos_sin": rotary_cos_sin_table(max(seq_lens) + 1, device),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "q_indptr": q_indptr.to(device),
        "page_indices": page_indices.to(device),
        "q_norm_weight": q_w.float().to(device),
        "k_norm_weight": k_w.float().to(device),
        "q_norm_weight_bf16": q_w.to(device),
        "k_norm_weight_bf16": k_w.to(device),
        "k_scale": torch.tensor([0.02], dtype=torch.float32, device=device),
        "v_scale": torch.tensor([0.03], dtype=torch.float32, device=device),
        "q_scale_inv": torch.tensor([1.0 / 0.02], dtype=torch.float32, device=device),
        "positions": positions.to(device),
        "batch_indices": batch_indices.to(device),
        "kv_indptr": kv_indptr.to(device),
        "kv_indices": kv_indices.to(device),
        "kv_last_page_len": kv_last_page_len.to(device),
        "total_pages": total_pages,
    }


def fresh_buffers(inp, device):
    row = inp["row"]
    shape = (inp["total_pages"], row.page_size, row.hkv, HEAD_DIM)
    kc = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    vc = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
    oq = torch.empty(
        (inp["T"], row.hq, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    if row.quant == 2:
        qs = torch.empty(0, dtype=torch.float32, device=device)
    elif inp["is_prefill"]:
        aligned = (inp["max_seqlen"] + 127) // 128 * 128
        qs = torch.empty(
            (row.batch, row.hq, aligned), dtype=torch.float32, device=device
        )
    else:
        qs = torch.empty((inp["T"], row.hq), dtype=torch.float32, device=device)
    flags = torch.zeros((row.batch, row.hkv), dtype=torch.int32, device=device)
    ok = ov = None
    if row.out_kv:
        ok = torch.zeros(
            (inp["T"], row.hkv, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
        )
        ov = torch.zeros_like(ok)
    return {"kc": kc, "vc": vc, "oq": oq, "qs": qs, "flags": flags, "ok": ok, "ov": ov}


def _fused_kwargs(inp, bufs):
    row = inp["row"]
    return dict(
        args=(
            inp["qkv"],
            inp["cos_sin"],
            inp["seq_lens"],
            inp["q_indptr"],
            inp["page_indices"],
            (bufs["kc"], bufs["vc"]),
            inp["is_prefill"],
            inp["k_scale"],
            inp["v_scale"],
            row.quant,
        ),
        kwargs=dict(
            max_seqlen=inp["max_seqlen"],
            upper_max=FP8_MAX,
            q_scale_inv=inp["q_scale_inv"] if row.quant == 2 else None,
            q_norm_weight=inp["q_norm_weight"] if row.norm else None,
            k_norm_weight=inp["k_norm_weight"] if row.norm else None,
            qk_norm_policy=row.norm,
            out_q=bufs["oq"],
            out_k=bufs["ok"],
            out_v=bufs["ov"],
            q_scale=bufs["qs"],
            split_k_flag=bufs["flags"],
        ),
    )


def make_cake_call(
    inp, device, fn=cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache
):
    """The Cake keyword signature (the package route, or a staged previous Cake route)."""
    bufs = fresh_buffers(inp, device)
    bound = _fused_kwargs(inp, bufs)

    def call():
        fn(*bound["args"], **bound["kwargs"])

    return call, bufs


def stable_exclusion(row: Row) -> str | None:
    """Why the stable pipeline does not cover ``row`` (``None`` when it does)."""
    if row.out_kv:
        return "caller-owned out_k / out_v: the stable pipeline appends to the caches"
    if row.ragged:
        return "ragged requests: the stable q_scale copy assumes one request length"
    if row.quant == 1 and row.prefill and row.padded_max_seqlen:
        return "max_seqlen beyond the request length: a kernel q_scale layout contract"
    return None


def make_stable_call(inp, device):
    """rmsnorm + apply_rope_pos_ids_inplace + torch FP8 quantize + append_paged_kv_cache."""
    row = inp["row"]
    reason = stable_exclusion(row)
    if reason is not None:
        raise ValueError(reason)
    bufs = fresh_buffers(inp, device)
    kc, vc, oq, qs = bufs["kc"], bufs["vc"], bufs["oq"], bufs["qs"]
    T = inp["T"]
    q_len = row.q_lens[0]
    q = inp["qkv"][:, : row.hq * HEAD_DIM].view(T, row.hq, HEAD_DIM)
    k = inp["qkv"][:, row.hq * HEAD_DIM : (row.hq + row.hkv) * HEAD_DIM].view(
        T, row.hkv, HEAD_DIM
    )
    v = inp["qkv"][:, (row.hq + row.hkv) * HEAD_DIM :].view(T, row.hkv, HEAD_DIM)
    tmp_q, tmp_k = torch.empty_like(q), torch.empty_like(k)
    norm_q, norm_k = torch.empty_like(q), torch.empty_like(k)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    qw, kw = inp["q_norm_weight_bf16"], inp["k_norm_weight_bf16"]
    paged = (kc, vc)
    k_inv, v_inv = 1.0 / inp["k_scale"], 1.0 / inp["v_scale"]

    def call():
        if row.norm == 2:
            flashinfer.norm.rmsnorm(
                q.reshape(-1, HEAD_DIM), qw, out=norm_q.view(-1, HEAD_DIM)
            )
            flashinfer.norm.rmsnorm(
                k.reshape(-1, HEAD_DIM), kw, out=norm_k.view(-1, HEAD_DIM)
            )
            flashinfer.rope.apply_rope_pos_ids_inplace(
                norm_q, norm_k, inp["positions"], interleave=False
            )
            q_src, k_src = norm_q, norm_k
        elif row.norm == 1:
            tmp_q.copy_(q)
            tmp_k.copy_(k)
            flashinfer.rope.apply_rope_pos_ids_inplace(
                tmp_q, tmp_k, inp["positions"], interleave=False
            )
            flashinfer.norm.rmsnorm(
                tmp_q.view(-1, HEAD_DIM), qw, out=norm_q.view(-1, HEAD_DIM)
            )
            flashinfer.norm.rmsnorm(
                tmp_k.view(-1, HEAD_DIM), kw, out=norm_k.view(-1, HEAD_DIM)
            )
            q_src, k_src = norm_q, norm_k
        else:
            tmp_q.copy_(q)
            tmp_k.copy_(k)
            flashinfer.rope.apply_rope_pos_ids_inplace(
                tmp_q, tmp_k, inp["positions"], interleave=False
            )
            q_src, k_src = tmp_q, tmp_k
        qf = q_src.float()
        if row.quant == 1:
            scale = qf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / FP8_MAX
            oq.copy_((qf / scale).clamp(-FP8_MAX, FP8_MAX))
            if inp["is_prefill"]:
                qs.view(row.batch, row.hq, -1)[:, :, :q_len].copy_(
                    scale.view(row.batch, q_len, row.hq).permute(0, 2, 1)
                )
            else:
                qs.copy_(scale.view(T, row.hq))
        else:
            oq.copy_((qf * inp["q_scale_inv"]).clamp(-FP8_MAX, FP8_MAX))
        k_fp8.copy_((k_src.float() * k_inv).clamp(-FP8_MAX, FP8_MAX))
        v_fp8.copy_((v.float() * v_inv).clamp(-FP8_MAX, FP8_MAX))
        flashinfer.page.append_paged_kv_cache(
            k_fp8,
            v_fp8,
            inp["batch_indices"],
            inp["positions"],
            paged,
            inp["kv_indices"],
            inp["kv_indptr"],
            inp["kv_last_page_len"],
            kv_layout="NHD",
        )

    return call, bufs


# ----------------------------------------------------------------------------------------------
# The previous Cake route as the paired baseline arm: the facade, its JIT loader (and the
# shared program runtime when the loader uses one) and the operator's generated sources of
# one FlashInfer git revision, copied out of the history into a private package that this
# process imports beside the package under test.  Only the relative imports of the copied
# Python modules are rewritten (to ``flashinfer.*``); the sources are copied byte for byte.

PREVIOUS_FACADE = "cake_fused_qk_rope_fp8_append"
PREVIOUS_OPERATOR_DIR = "cake_fused_qk_rope_fp8_append"
PREVIOUS_ENTRY_POINT = "cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache"


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def _git_blob(repo: Path, ref: str, path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True
    )
    return proc.stdout if proc.returncode == 0 else None


def _rewrite_relative_imports(
    text: str, *, under_jit: bool, copied_jit: set[str]
) -> tuple[str, list[str]]:
    """Point the relative imports of a copied module at ``flashinfer`` (``.utils`` ->
    ``flashinfer.utils``, ``.core`` -> ``flashinfer.jit.core``), keeping those that name
    another copied module (the facade's ``.jit.<loader>``, the loader's ``.cake_program``)."""
    notes: list[str] = []

    def keep_or(target: str, name: str, repl: str) -> str:
        if name in copied_jit:
            return target
        notes.append(f"{target.strip()!r} -> {repl.strip()!r}")
        return repl

    if under_jit:
        text = re.sub(
            r"^from \. import (\w+)",
            lambda m: keep_or(
                m.group(0), m.group(1), f"from flashinfer.jit import {m.group(1)}"
            ),
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^from \.(\w+) import",
            lambda m: keep_or(
                m.group(0), m.group(1), f"from flashinfer.jit.{m.group(1)} import"
            ),
            text,
            flags=re.MULTILINE,
        )
    else:
        text = re.sub(
            r"^from \.jit\.(\w+) import",
            lambda m: keep_or(
                m.group(0), m.group(1), f"from flashinfer.jit.{m.group(1)} import"
            ),
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^from \.jit import",
            lambda m: keep_or(m.group(0), "", "from flashinfer.jit import"),
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^from \.(\w+) import",
            lambda m: keep_or(m.group(0), "", f"from flashinfer.{m.group(1)} import"),
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(
            r"^from \. import (\w+)",
            lambda m: keep_or(m.group(0), "", f"from flashinfer import {m.group(1)}"),
            text,
            flags=re.MULTILINE,
        )
    return text, notes


def stage_previous_cake(ref: str, repo: Path, stage_dir: Path) -> dict:
    """Stage the Cake route of FlashInfer revision ``ref`` (from the git history of ``repo``)
    under ``stage_dir`` as the importable package ``cake_prev_<facade>_<sha7>`` and return its
    manifest (``package``, ``module``, ``ref``, per-file sha256, the import rewrites).

    Layout: ``<stage_dir>/<package>/<facade>.py``, ``<package>/jit/<loader>.py`` (+
    ``jit/cake_program.py`` when the loader imports it) and ``<stage_dir>/csrc/<operator>/``;
    the loaders of every shipped form find their sources two directories above the loader
    (``csrc/<operator>``), so the copied sources are the ones they compile, and a binding
    that includes ``tvm_ffi_utils.h`` gets that revision's header beside them.
    """
    sha = _git(repo, "rev-parse", f"{ref}^{{commit}}").decode().strip()
    package = f"cake_prev_{PREVIOUS_FACADE}_{sha[:7]}"
    root = stage_dir / package
    manifest_path = stage_dir / f"{package}.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("ref") == sha and all(
            (stage_dir / rel).is_file()
            and hashlib.sha256((stage_dir / rel).read_bytes()).hexdigest() == digest
            for rel, digest in manifest["files"].items()
        ):
            return manifest
    facade_src = f"flashinfer/{PREVIOUS_FACADE}.py"
    loader_src = f"flashinfer/jit/{PREVIOUS_FACADE}.py"
    facade = _git_blob(repo, sha, facade_src)
    if facade is None:
        raise ValueError(
            f"{sha[:12]} has no {facade_src}: no previous Cake route to stage"
        )
    loader = _git_blob(repo, sha, loader_src)
    copied_jit = {PREVIOUS_FACADE}
    runtime = None
    if loader is not None and re.search(
        r"^from \.(?: import cake_program|cake_program import)", loader.decode(), re.M
    ):
        runtime = _git_blob(repo, sha, "flashinfer/jit/cake_program.py")
        copied_jit.add("cake_program")
    sources = [
        line
        for line in _git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            sha,
            "--",
            f"csrc/{PREVIOUS_OPERATOR_DIR}",
        )
        .decode()
        .splitlines()
        if line
    ]
    if not sources:
        raise ValueError(f"{sha[:12]} has no csrc/{PREVIOUS_OPERATOR_DIR}/ sources")
    import shutil

    if root.exists():
        shutil.rmtree(root)
    if (stage_dir / "csrc" / PREVIOUS_OPERATOR_DIR).exists():
        shutil.rmtree(stage_dir / "csrc" / PREVIOUS_OPERATOR_DIR)
    (root / "jit").mkdir(parents=True)
    files: dict[str, str] = {}
    rewrites: dict[str, list[str]] = {}

    def put(rel: str, data: bytes) -> None:
        path = stage_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()

    put(f"{package}/__init__.py", b"")
    put(f"{package}/jit/__init__.py", b"")
    text, notes = _rewrite_relative_imports(
        facade.decode(), under_jit=False, copied_jit=copied_jit
    )
    put(f"{package}/{PREVIOUS_FACADE}.py", text.encode())
    rewrites[facade_src] = notes
    if loader is not None:
        text, notes = _rewrite_relative_imports(
            loader.decode(), under_jit=True, copied_jit=copied_jit
        )
        put(f"{package}/jit/{PREVIOUS_FACADE}.py", text.encode())
        rewrites[loader_src] = notes
    if runtime is not None:
        text, notes = _rewrite_relative_imports(
            runtime.decode(), under_jit=True, copied_jit=copied_jit
        )
        put(f"{package}/jit/cake_program.py", text.encode())
        rewrites["flashinfer/jit/cake_program.py"] = notes
    needs_ffi_utils = False
    for src in sources:
        blob = _git_blob(repo, sha, src)
        assert blob is not None, src
        needs_ffi_utils = needs_ffi_utils or b'"tvm_ffi_utils.h"' in blob
        put(src, blob)
    if needs_ffi_utils:
        header = _git_blob(repo, sha, "csrc/tvm_ffi_utils.h")
        if header is None:
            raise ValueError(
                f"{sha[:12]}: the bindings include tvm_ffi_utils.h but the revision has none"
            )
        put("csrc/tvm_ffi_utils.h", header)
    manifest = {
        "ref": sha,
        "repo": str(repo),
        "package": package,
        "module": f"{package}.{PREVIOUS_FACADE}",
        "files": files,
        "import_rewrites": rewrites,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def import_staged_baseline(manifest: dict, stage_dir: Path):
    """Import the staged route and check that its loader compiles the staged sources
    (never the sources of the checkout or of an installed wheel)."""
    if str(stage_dir) not in sys.path:
        sys.path.insert(0, str(stage_dir))
    module = importlib.import_module(manifest["module"])
    loader = sys.modules.get(f"{manifest['package']}.jit.{PREVIOUS_FACADE}")
    runtime = sys.modules.get(f"{manifest['package']}.jit.cake_program")
    resolved: list[Path] = []
    if loader is not None and hasattr(loader, "_get_csrc_dir"):
        resolved.append(Path(loader._get_csrc_dir()))
    if (
        loader is not None
        and runtime is not None
        and hasattr(runtime, "operator_csrc_dir")
    ):
        for value in vars(loader).values():
            operator_dir = getattr(value, "operator_dir", None)
            if isinstance(operator_dir, str):
                resolved.append(Path(runtime.operator_csrc_dir(operator_dir)))
    for path in resolved:
        if stage_dir.resolve() not in path.resolve().parents:
            raise RuntimeError(
                f"the staged baseline would compile sources outside {stage_dir}: {path}"
            )
    manifest["resolved_sources"] = [str(p) for p in resolved]
    return module


def resolve_baseline(module_name: str, kind: str):
    """The fused baseline entry point in ``module_name`` (``cake``: a previous Cake route
    exposing ``cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache``)."""
    mod = importlib.import_module(module_name)
    fn = getattr(mod, PREVIOUS_ENTRY_POINT, None)
    if fn is None:
        raise ValueError(f"{module_name} has no {PREVIOUS_ENTRY_POINT}")
    return fn, "cake"


def time_arm(call, *, graph: bool, warmup: int, iters: int, graph_instances: int):
    if graph:
        s = torch.cuda.Stream()
        graphs = []
        with torch.cuda.stream(s):
            for _ in range(3):
                call()
            for _ in range(graph_instances):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    call()
                graphs.append(g)
        torch.cuda.synchronize()
        rotation = itertools.cycle(graphs)

        def fn():
            next(rotation).replay()

    else:
        fn = call
    times = bench_gpu_time(
        fn,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        enable_cupti=True,
    )
    return float(statistics.median(times)) * 1e3  # ms -> us


def sm_clock_mhz(device) -> int | None:
    try:
        return int(torch.cuda.clock_rate(device))
    except Exception:  # the driver / NVML binding may be unavailable
        return None


def selected_stage(inp, device) -> str:
    row = inp["row"]
    index = device.index if device.index is not None else torch.cuda.current_device()
    return loader.stage_for(
        row.hq,
        row.hkv,
        arch=loader.device_arch(index),
        num_rows=inp["T"],
        num_requests=row.batch,
        page_size=row.page_size,
        num_sms=loader.sm_count(index),
        is_prefill=row.prefill,
        max_pages_per_request=inp["max_pages"],
        has_out_kv=row.out_kv,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", default=",".join(ROWS), help="comma list of row names")
    ap.add_argument("--groups", type=int, default=3, help="paired A/B/B/A groups")
    ap.add_argument("--iters", type=int, default=30, help="timed samples per group")
    ap.add_argument("--warmup", type=int, default=10, help="warm-up launches per group")
    ap.add_argument(
        "--graph-instances",
        type=int,
        default=8,
        help="independent CUDA-graph captures replayed in rotation (graph mode)",
    )
    ap.add_argument(
        "--no-stable", action="store_true", help="skip the stable pipeline arm"
    )
    ap.add_argument(
        "--baseline-ref",
        default=None,
        help=(
            "FlashInfer git revision whose Cake route is staged out of the history and timed "
            "as the baseline arm (the previous Cake version of this kernel)"
        ),
    )
    ap.add_argument(
        "--baseline-repo",
        default=str(Path(__file__).resolve().parents[1]),
        help="git checkout holding --baseline-ref (default: this checkout)",
    )
    ap.add_argument(
        "--baseline-stage-dir",
        default=str(Path(tempfile.gettempdir()) / f"cake_prev_{PREVIOUS_FACADE}"),
        help="where the staged baseline package is written",
    )
    ap.add_argument(
        "--baseline-module",
        default=None,
        help="importable previous Cake route (same entry-point name, keyword signature)",
    )
    ap.add_argument(
        "--baseline-kind",
        choices=("auto", "cake"),
        default="auto",
        help="signature of the baseline module's entry point",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    names = args.rows.split(",")
    arms = ["cake"]
    if not args.no_stable:
        arms.append("stable")
    baseline_fn = baseline_kind = None
    staged = None
    if args.baseline_ref:
        if args.baseline_module:
            raise SystemExit("--baseline-ref and --baseline-module are exclusive")
        stage_dir = Path(args.baseline_stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = stage_previous_cake(
            args.baseline_ref, Path(args.baseline_repo), stage_dir
        )
        import_staged_baseline(staged, stage_dir)
        args.baseline_module, args.baseline_kind = staged["module"], "cake"
        print(
            f"baseline: Cake route of {staged['ref'][:12]} staged as {staged['package']} "
            f"({len(staged['files'])} files under {stage_dir})"
        )
    if args.baseline_module:
        baseline_fn, baseline_kind = resolve_baseline(
            args.baseline_module, args.baseline_kind
        )
        arms.append("baseline")
    # The rows every arm covers, declared before the first measurement.
    arms_meta = {"cake": {"role": "candidate", "rows": list(names)}}
    if "stable" in arms:
        excluded = {
            n: stable_exclusion(ROWS[n])
            for n in names
            if stable_exclusion(ROWS[n]) is not None
        }
        arms_meta["stable"] = {
            "role": "report-only (BF16 intermediates, no last-page clear)",
            "rows": [n for n in names if n not in excluded],
            "excluded": excluded,
        }
    if "baseline" in arms:
        arms_meta["baseline"] = {
            "role": "baseline",
            "kind": baseline_kind,
            "module": args.baseline_module,
            "staged": staged,
            "rows": list(names),
        }
    capability = torch.cuda.get_device_capability(device)
    print(
        f"device {torch.cuda.get_device_name(device)} sm_{capability[0]}{capability[1]} "
        f"| {torch.cuda.get_device_properties(device).multi_processor_count} SMs "
        f"| CUDA {torch.version.cuda} | torch {torch.__version__}"
    )
    print(
        f"arms {arms} | groups {args.groups} x iters {args.iters} (warmup {args.warmup}) "
        f"| graph instances {args.graph_instances}"
    )
    for arm, meta in arms_meta.items():
        if meta.get("excluded"):
            print(f"{arm}: skips {sorted(meta['excluded'])} ({meta['role']})")
    print(
        f"{'row':34s} {'cfg':46s} "
        + " ".join(f"{a + '-' + m:>15s}" for a in arms for m in MODES)
    )
    results = []
    for name in names:
        row = ROWS[name]
        inp = build_inputs(row, device)
        calls = {}
        for arm in arms:
            if arm == "cake":
                calls[arm] = make_cake_call(inp, device)[0]
            elif arm == "stable":
                if name in arms_meta["stable"]["excluded"]:
                    continue
                calls[arm] = make_stable_call(inp, device)[0]
            else:
                calls[arm] = make_cake_call(inp, device, baseline_fn)[0]
        timed = [a for a in arms if a in calls]
        samples = {(a, m): [] for a in timed for m in MODES}
        clock_before = sm_clock_mhz(device)
        for g in range(args.groups):
            order = timed if g % 2 == 0 else list(reversed(timed))
            for arm in order:
                for mode in MODES:
                    samples[(arm, mode)].append(
                        time_arm(
                            calls[arm],
                            graph=(mode == "graph"),
                            warmup=args.warmup,
                            iters=args.iters,
                            graph_instances=args.graph_instances,
                        )
                    )
        clock_after = sm_clock_mhz(device)
        med = {k: statistics.median(v) for k, v in samples.items()}
        cfg = row.geometry()
        cells = []
        for a in arms:
            for m in MODES:
                cells.append(
                    f"{med[(a, m)]:>12.2f} us" if (a, m) in med else f"{'-':>15s}"
                )
        print(f"{name:34s} {cfg:46s} " + " ".join(cells))
        speedup = {
            f"{a}/cake-{m}": med[(a, m)] / med[("cake", m)]
            for a in timed
            if a != "cake"
            for m in MODES
        }
        results.append(
            {
                "row": name,
                "config": cfg,
                "seed": row.seed,
                "stage": selected_stage(inp, device),
                "medians_us": {f"{a}-{m}": med[(a, m)] for a, m in med},
                "samples_us": {f"{a}-{m}": v for (a, m), v in samples.items()},
                "speedup": speedup,
                "sm_clock_mhz": {"before": clock_before, "after": clock_after},
            }
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "capability": list(capability),
                    "sm_count": torch.cuda.get_device_properties(
                        device
                    ).multi_processor_count,
                    "cuda": torch.version.cuda,
                    "torch": torch.__version__,
                    "protocol": {
                        "groups": args.groups,
                        "iters": args.iters,
                        "warmup": args.warmup,
                        "graph_instances": args.graph_instances,
                        "timer": "flashinfer.testing.bench_gpu_time (CUPTI, cold L2)",
                        "statistic": "median over iters per group; median of groups",
                    },
                    "arms": arms_meta,
                    "rows": results,
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
