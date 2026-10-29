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

Backend suite of the Cake packed-varlen GQA training attention package (head dimension 256,
``flashinfer.experimental.cake_gqa_train``).

Without a device: the registry (one record per stage and architecture, the argument plans the host passes), the
build manifest (every record's library, the recorded source digests), the standalone loader and its pinned bytes,
no environment read, the launch grids against the exact tile counts of generated layouts and the scratch layout.

On SM100 / SM103 / SM107: the output, the LSE and every gradient against an FP64 reference of the same BF16 inputs on packed
layouts (unequal lengths, tails that are no multiple of a tile, keys fewer than queries, empty segments) in the three
dK / dV forms, under the accuracy budget declared below; rows that see no key exactly (``out = 0``, ``lse = -inf``,
zero gradients); document isolation; repeated destination rows accumulated into an existing FP32 destination;
dynamic shapes through one reused workspace; launches on the operands' device and stream while another device is
current; non-reentrant checkpointing (nothing retained, gradients equal to the
eager pair); no device synchronization in steady state; and the speed of the forward and the backward beside a
kernel resident on another stream.

The suite names its package once, through the shared alias helper, and imports nothing else from flashinfer.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_gqa_train")
cake_backend = cake_package("cake_gqa_train", "cake_backend")
cake_build = cake_package("cake_gqa_train", "cake_build")
cake_jit = cake_package("cake_gqa_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = tuple(cake_jit.ARCHES)
D = 256
SCALE = 0.0625
MIB = 2**20

# Accuracy budget (declared before the first run): per layout and tensor, the relative L2 error against the FP64
# reference of the same BF16 inputs is at most 1.10 times FlashAttention CuTe's (98c1cee0, its dedicated head-dim-256
# backward) on B300 against the same reference: out 2.3e-3, lse 1.5e-7, dq 2.6e-3, dk 2.6e-3, dv 2.4e-3 (the largest
# of its calibration layouts).  FP32 window gradients and destinations are compared in FP32.
FA4_REL_L2 = {"out": 2.3e-3, "lse": 1.5e-7, "dq": 2.6e-3, "dk": 2.6e-3, "dv": 2.4e-3}
BUDGET = 1.10
# dQ is reduced in FP32 with atomic adds, one partial per 128-key block, in a scheduling order, then rounded to BF16
# once: two runs of one input differ by the FP32 reordering error before that rounding.  The comparison is one BF16
# rounding step (DQ_RTOL) plus an absolute floor for the reordering error, which is absolute (FP32 ulps of the partials;
# it dominates on elements whose partials cancel to near zero): DQ_ATOL_FRACTION of the tensor's largest magnitude,
# 2^9 times below one BF16 step there.  O, LSE, dK and dV are compared bitwise.
DQ_RTOL = 2**-7
DQ_ATOL_FRACTION = 2**-16
# Checkpointing: memory kept after a checkpointed forward, less the output storage.
CKPT_RETAINED_LIMIT_MIB = 2.0
# Beside a resident side-stream kernel: busy over alone within MAX_RATIO or at most MAX_EXTRA_MS above it.
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020


# ----------------------------------------------------------------------------------------------------------------
# Layouts: (name, q_lens, k_lens, H_q, H_kv)

LAYOUTS = (
    ("tails", (77, 300, 1, 129), (77, 700, 1000, 129), 24, 4),
    ("keys_fewer_than_queries", (200, 150, 64), (77, 0, 63), 24, 4),
    ("empty_segments", (0, 100, 0, 37, 0), (0, 300, 50, 37, 0), 24, 4),
    ("zigzag_cp2", (250, 250, 129, 130), (250, 1000, 129, 517), 24, 4),
    ("long_keys", (256, 255), (4099, 2049), 24, 4),
    ("group_2", (300, 77), (513, 77), 8, 4),
)
LAYOUT_IDS = [layout[0] for layout in LAYOUTS]
DKV_FORMS = ("bf16", "fp32", "rowmap")


def _cu(lens) -> list[int]:
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + int(n))
    return cu


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    arch = cake_backend.arch_for_capability(torch.cuda.get_device_capability(0))
    if arch is None:
        pytest.skip(
            f"the package ships {ARCHES}; this device is {torch.cuda.get_device_capability(0)}"
        )
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first timed or sync-checked call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == (
        "fwd",
        "bwd_pre",
        "bwd_fp32",
        "bwd_rowmap",
        "bwd_bf16",
        "bwd_post",
    )
    for arch in ARCHES:
        for stage in cake_jit.STAGES:
            serving = [
                n
                for n, rec in cake_jit.MODULES.items()
                if rec["stage"] == stage and arch in rec["arches"]
            ]
            assert len(serving) == 1, (stage, arch, serving)
    for name, rec in cake_jit.MODULES.items():
        assert set(rec["arches"]) <= set(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert len(rec["library"]) == len("cake_gqa_train_") + 20 and rec[
            "library"
        ].startswith("cake_gqa_train_")
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        pair = rec["stage"] in ("fwd", "bwd_fp32", "bwd_rowmap", "bwd_bf16")
        assert tuple(launch["cluster"]) == ((2, 1, 1) if pair else (1, 1, 1)), name
        assert launch["block"][0] % 32 == 0, name
        # Every parameter the host passes is in the plan, in the order the kernel declares.
        assert cake_backend.parameters(name)[0] in ("Q", "O", "dQ_accum"), name


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {
        f"{rec['library']}_{arch}"
        for rec in cake_jit.MODULES.values()
        for arch in rec["arches"]
    }
    assert set(entries) == wanted
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert (
                hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest
            ), (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_the_loader_imports_standalone_without_flashinfer() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a process where ``flashinfer`` cannot be imported
    reads the manifest (the prebuilt route imports nothing from FlashInfer)."""
    _manifest_or_skip()
    code = (
        "import importlib.util, sys\n"
        "sys.modules['flashinfer'] = None\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {str(PACKAGE_DIR / 'cake_build.py')!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    records = sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


def test_the_loader_and_header_are_the_pinned_bytes_of_every_cake_package() -> None:
    _manifest_or_skip()
    # Another package's copy, read from the repository (not imported: the vendored gate blocks flashinfer).
    other = (
        Path(__file__).resolve().parents[2] / "flashinfer/experimental/cake_kda_train"
    )
    for relative in ("cake_build.py", "csrc/cake_ffi_checks.h"):
        assert (PACKAGE_DIR / relative).read_bytes() == (
            other / relative
        ).read_bytes(), relative


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path; without one the JIT spec names the entry."""
    _manifest_or_skip()
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_backend.library_name(name, arch)
            assert (
                cake_build.prebuilt_path(library)
                == PACKAGE_DIR / "aot" / f"{library}.so"
            )
            assert (
                cake_backend.is_prebuilt(name, arch)
                == cake_build.prebuilt_path(library).is_file()
            )


def test_the_package_reads_no_environment_variable() -> None:
    offenders = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def _exact_tiles(q_lens, k_lens) -> tuple[int, int]:
    """Forward tiles per query head and backward pair tiles per key head of a layout."""
    return (
        sum(-(-q // cake_jit.FWD_TILE_ROWS) for q in q_lens),
        sum(-(-k // cake_jit.BWD_TILE_KEYS) for k in k_lens),
    )


def test_grids_cover_every_tile() -> None:
    rng = random.Random(7073)
    for _ in range(500):
        n = rng.randint(1, 40)
        q_lens = [
            rng.choice((0, 1, rng.randint(1, 600), rng.randint(1, 9000)))
            for _ in range(n)
        ]
        k_lens = [rng.choice((0, q, q + rng.randint(0, 20000))) for q in q_lens]
        dims = dict(
            total_q=sum(q_lens),
            total_k=sum(k_lens),
            num_heads_q=24,
            num_heads_kv=4,
            num_segments=n,
            sm_count=148,
        )
        fwd_tiles, bwd_tiles = _exact_tiles(q_lens, k_lens)
        # The forward is persistent: any pair count >= 1 covers every tile; the bound caps it at one pair per two
        # SMs and at the tile count bound.
        bound = dims["total_q"] // cake_jit.FWD_TILE_ROWS + n
        assert bound >= fwd_tiles
        assert cake_jit.grid("fwd", **dims) == (
            2 * max(1, min(148 // 2, 24 * bound)),
            1,
            1,
        )
        # The backward main kernel: a pair per (128-key block of the longest segment, key head, segment), sized by
        # max_seqlen_k when given (exactly) and by T_k otherwise (an upper bound).
        longest = max(-(-k // cake_jit.BWD_TILE_KEYS) for k in k_lens)
        assert bwd_tiles <= n * max(1, longest)
        for stage in ("bwd_fp32", "bwd_rowmap", "bwd_bf16"):
            gx, gy, gz = cake_jit.grid(stage, **dims)
            assert gx % 2 == 0 and gx // 2 >= longest and (gy, gz) == (4, n)
            exact = cake_jit.grid(stage, **dims, max_seqlen_k=max(k_lens))
            assert exact == (2 * max(1, longest), 4, n)
        rows = cake_jit.rows_pad(dims["total_q"], n)
        assert rows % 128 == 0 and rows >= sum(-(-2 * q // 128) * 128 for q in q_lens)
        assert cake_jit.grid("bwd_pre", **dims) == (rows // 32, 12, 1)
        assert cake_jit.grid("bwd_post", **dims) == (rows // 64, 12, 1)


def test_scratch_layout_is_aligned_and_sized() -> None:
    for total_q, heads, segs in ((1, 24, 1), (30193, 24, 12), (77, 8, 3)):
        rows = cake_jit.rows_pad(total_q, segs)
        layout = cake_jit.scratch_layout(total_q, heads, segs)
        assert layout["dq_accum"][1] == (heads // 2, rows, D)
        assert layout["lse_log2"][1] == layout["dpsum"][1] == (heads // 2, rows)
        for offset, _shape in layout.values():
            assert offset % cake_jit.SCRATCH_ALIGN == 0
        floats = (heads // 2) * rows * (D + 2)
        assert (
            4 * floats
            <= cake_backend.workspace_bytes(total_q, heads, segs)
            <= 4 * floats + 3 * 256
        )


# ----------------------------------------------------------------------------------------------------------------
# The FP64 reference


def _inputs(q_lens, k_lens, h_q, h_kv, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    t_q, t_k = sum(q_lens), sum(k_lens)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(
            torch.bfloat16
        )

    return {
        "q": randn(t_q, h_q, D),
        "k": randn(t_k, h_kv, D),
        "v": randn(t_k, h_kv, D),
        "dout": randn(t_q, h_q, D),
        "cu_q": torch.tensor(_cu(q_lens), dtype=torch.int32, device=device),
        "cu_k": torch.tensor(_cu(k_lens), dtype=torch.int32, device=device),
        "cq": _cu(q_lens),
        "ck": _cu(k_lens),
    }


def _reference(x, scale=SCALE):
    """FP64 out, lse and gradients of the same BF16 inputs (bottom-right causal per segment)."""
    q = x["q"].double().requires_grad_()
    k = x["k"].double().requires_grad_()
    v = x["v"].double().requires_grad_()
    group = q.shape[1] // k.shape[1]
    outs, lses = [], []
    for b in range(len(x["cq"]) - 1):
        q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
        ql, kl = q1 - q0, k1 - k0
        qs = q[q0:q1]
        ks = k[k0:k1].repeat_interleave(group, dim=1)
        vs = v[k0:k1].repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        seen = torch.arange(kl, device=q.device)[None, :] <= torch.arange(
            ql, device=q.device
        )[:, None] + (kl - ql)
        has_key = seen.any(dim=-1)
        # Rows without a key take finite dummy scores (no NaN reaches the gradients) and a zero P.
        s = s.masked_fill(~seen, -math.inf).masked_fill(~has_key[None, :, None], 0.0)
        lse = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse[..., None]) * seen
        outs.append(torch.einsum("hqk,khd->qhd", p, vs))
        lses.append(lse.detach().masked_fill(~has_key[None, :], -math.inf))
    out = torch.cat(outs)
    lse = torch.cat(lses, dim=1)
    out.backward(x["dout"].double())
    return {
        "out": out.detach(),
        "lse": lse.detach(),
        "dq": q.grad,
        "dk": k.grad,
        "dv": v.grad,
    }


def _assert_dq_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Two dQ of one input (see DQ_RTOL / DQ_ATOL_FRACTION)."""
    floor = DQ_ATOL_FRACTION * float(ref.float().abs().max()) if ref.numel() else 0.0
    torch.testing.assert_close(got, ref, rtol=DQ_RTOL, atol=floor)


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    finite = torch.isfinite(ref)
    num = torch.linalg.vector_norm((got - ref)[finite])
    den = torch.linalg.vector_norm(ref[finite])
    return float(num / den) if den > 0 else float(num)


def _no_key_rows(x, h_q):
    """Mask ``[T_q]`` of the query rows that see no key (``i + k_len - q_len < 0``)."""
    rows = []
    for b in range(len(x["cq"]) - 1):
        ql, kl = x["cq"][b + 1] - x["cq"][b], x["ck"][b + 1] - x["ck"][b]
        rows += [i + kl - ql < 0 for i in range(ql)]
    return torch.tensor(rows, dtype=torch.bool, device=x["q"].device)


def _run(x, form, *, workspace=None, dst_init=None, row_map=None):
    """Forward and backward of the package: ``{"out", "lse", "dq", "dk", "dv"}`` (dk / dv as the form returns them)."""
    out, lse = cake_backend.gqa_attention_fwd(
        x["q"],
        x["k"],
        x["v"],
        x["cu_q"],
        x["cu_k"],
        softmax_scale=SCALE,
        workspace=workspace,
    )
    # The BF16 form sizes the backward grid by T_k (no hint), the FP32 forms by the exact longest key segment.
    ck = x["ck"]
    longest = max(ck[i + 1] - ck[i] for i in range(len(ck) - 1))
    max_k = None if form == "bf16" else longest
    kwargs = {"softmax_scale": SCALE, "workspace": workspace, "max_seqlen_k": max_k}
    if form == "rowmap":
        kwargs.update(
            dk_acc=dst_init[0].clone(), dv_acc=dst_init[1].clone(), row_map=row_map
        )
    elif form == "fp32":
        kwargs["dkv_dtype"] = torch.float32
    dq, dk, dv = cake_backend.gqa_attention_bwd(
        x["q"], x["k"], x["v"], out, lse, x["dout"], x["cu_q"], x["cu_k"], **kwargs
    )
    return {"out": out, "lse": lse, "dq": dq, "dk": dk, "dv": dv}


def _window_row_map(x, seed=0):
    """A row map sending every window row to a destination row, the windows of a segment pair sharing rows
    (repeated destination rows across windows), plus untouched destination rows."""
    ck = x["ck"]
    rows = []
    for b in range(len(ck) - 1):
        start = (
            0 if b % 2 else 3
        )  # odd windows restart at row 0: they overlap the even windows' rows
        rows += [start + j for j in range(ck[b + 1] - ck[b])]
    n_dst = (max(rows) + 1 if rows else 0) + 5
    return torch.tensor(rows, dtype=torch.int32, device=x["q"].device), n_dst


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107


@pytest.mark.parametrize("form", DKV_FORMS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_matches_the_fp64_reference(layout, form, device) -> None:
    name, q_lens, k_lens, h_q, h_kv = layout
    x = _inputs(q_lens, k_lens, h_q, h_kv, device)
    ref = _reference(x)
    dst_init = row_map = None
    if form == "rowmap":
        row_map, n_dst = _window_row_map(x)
        g = torch.Generator(device=device).manual_seed(1)
        dst_init = tuple(
            torch.randn(n_dst, h_kv, D, generator=g, device=device) for _ in range(2)
        )
    got = _run(x, form, dst_init=dst_init, row_map=row_map)
    torch.cuda.synchronize()
    if form == "rowmap":
        rm = row_map.long()
        for t, init in (("dk", dst_init[0]), ("dv", dst_init[1])):
            expected = init.double().index_add(0, rm, ref[t])
            touched = torch.zeros(init.shape[0], dtype=torch.bool, device=device)
            touched[rm] = True
            assert torch.equal(got[t][~touched], init[~touched]), (
                f"{name}: untouched {t} rows changed"
            )
            ref[t] = expected - init.double()
            got[t] = got[t].double() - init.double()
    no_key = _no_key_rows(x, h_q)
    errors = {}
    for t in ("out", "lse", "dq", "dk", "dv"):
        assert torch.isfinite(got[t][torch.isfinite(ref[t])]).all(), (
            f"{name}: non-finite {t}"
        )
        errors[t] = _rel_l2(got[t], ref[t])
    print(
        f"REL_L2 {name} {form} " + " ".join(f"{t}={e:.3e}" for t, e in errors.items())
    )
    for t, e in errors.items():
        assert e <= BUDGET * FA4_REL_L2[t], (
            f"{name} {form}: {t} rel-L2 {e:.3e} > {BUDGET} x {FA4_REL_L2[t]:.1e}"
        )
    # Rows that see no key: out 0, lse -inf, dq 0, exactly.
    assert torch.equal(got["out"][no_key], torch.zeros_like(got["out"][no_key]))
    assert bool((got["lse"][:, no_key] == -math.inf).all())
    assert bool((got["lse"][:, ~no_key] > -math.inf).all())
    assert torch.equal(got["dq"][no_key], torch.zeros_like(got["dq"][no_key]))
    if form != "rowmap":
        # Keys no query sees (a segment without queries) get exactly zero gradients.
        unseen = ref["dk"].abs().amax(dim=(1, 2)) == 0
        unseen &= ref["dv"].abs().amax(dim=(1, 2)) == 0
        assert torch.equal(got["dk"][unseen], torch.zeros_like(got["dk"][unseen]))
        assert torch.equal(got["dv"][unseen], torch.zeros_like(got["dv"][unseen]))


def test_documents_are_isolated(device) -> None:
    q_lens, k_lens = (130, 77, 300, 41), (500, 77, 301, 1000)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    base = _run(x, "fp32")
    again = _run(x, "fp32")
    y = dict(x)
    b = 2
    q0, q1, k0, k1 = x["cq"][b], x["cq"][b + 1], x["ck"][b], x["ck"][b + 1]
    for key, lo, hi in (("q", q0, q1), ("dout", q0, q1), ("k", k0, k1), ("v", k0, k1)):
        y[key] = x[key].clone()
        y[key][lo:hi] = -y[key][lo:hi] * 1.5
    pert = _run(y, "fp32")
    torch.cuda.synchronize()
    q_other = torch.ones(sum(q_lens), dtype=torch.bool, device=device)
    q_other[q0:q1] = False
    k_other = torch.ones(sum(k_lens), dtype=torch.bool, device=device)
    k_other[k0:k1] = False
    assert not torch.equal(pert["out"][~q_other], base["out"][~q_other])
    assert torch.equal(pert["out"][q_other], base["out"][q_other])
    assert torch.equal(pert["lse"][:, q_other], base["lse"][:, q_other])
    assert torch.equal(pert["dk"][k_other], base["dk"][k_other])
    assert torch.equal(pert["dv"][k_other], base["dv"][k_other])
    # dQ is reduced with atomic adds: the other segments agree to the reduction-order rounding of two plain runs.
    _assert_dq_close(again["dq"], base["dq"])
    _assert_dq_close(pert["dq"][q_other], base["dq"][q_other])


def test_dynamic_shapes_reuse_one_workspace(device) -> None:
    ws = cake_backend.Workspace(device)
    sizes = []
    for i, (q_lens, k_lens) in enumerate(
        (
            ((3000, 77), (9000, 77)),
            ((1,), (1,)),
            ((500, 0, 260), (500, 40, 4000)),
            ((2900, 300), (3000, 8000)),
        )
    ):
        x = _inputs(q_lens, k_lens, 24, 4, device, seed=i)
        shared = _run(x, "fp32", workspace=ws)
        fresh = _run(x, "fp32", workspace=cake_backend.Workspace(device))
        torch.cuda.synchronize()
        for t in ("out", "lse", "dk", "dv"):
            assert torch.equal(shared[t], fresh[t]), (q_lens, t)
        _assert_dq_close(shared["dq"], fresh["dq"])
        sizes.append(ws.nbytes)
    assert sizes == sorted(sizes), "the workspace shrank"
    need = max(
        cake_backend.workspace_bytes(sum(q), 24, len(q))
        for q in ((3000, 77), (1,), (500, 0, 260), (2900, 300))
    )
    assert ws.nbytes - ws.tile_counter().untyped_storage().nbytes() == need


def test_launches_on_the_operands_device_and_stream(device) -> None:
    """Operands on ``device`` with another device current: every launch runs on ``device``'s current stream."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device to make current")
    other = 1 if device.index == 0 else 0
    x = _inputs((700, 300, 41), (2000, 300, 900), 24, 4, device, seed=3)
    with torch.cuda.device(device):
        ref = _run(x, "fp32")
        torch.cuda.synchronize(device)
    prev_device = torch.cuda.current_device()
    prev_stream = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    try:
        torch.cuda.set_stream(side)  # also makes ``device`` current
        torch.cuda.set_device(other)  # ``device`` keeps ``side`` as its current stream
        assert torch.cuda.current_device() == other
        assert torch.cuda.current_stream(device) == side
        side.wait_stream(prev_stream)
        got = _run(x, "fp32")
        side.synchronize()
    finally:
        torch.cuda.set_stream(prev_stream)
        torch.cuda.set_device(prev_device)
    for t in ("out", "lse", "dk", "dv"):
        assert got[t].device == device, t
        assert torch.equal(got[t], ref[t]), t
    _assert_dq_close(got["dq"], ref["dq"])


def _ckpt_retained_mib(fn, *inputs) -> float:
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = checkpoint(fn, *inputs, use_reentrant=False)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
    del out
    return kept / MIB


@pytest.mark.parametrize("form", ("bf16", "rowmap"))
def test_checkpoint_keeps_nothing_alive_and_matches_eager(form, device) -> None:
    q_lens, k_lens = (1000, 3000), (1000, 24000)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    ws = cake_backend.Workspace(device)
    row_map, n_dst = _window_row_map(x)
    dst = {}

    def dkv_acc():
        return dst["k"], dst["v"], row_map

    def fn(q, k, v):
        return pkg.varlen_gqa_attention(
            q,
            k,
            v,
            x["cu_q"],
            x["cu_k"],
            softmax_scale=SCALE,
            workspace=ws,
            dkv_acc=dkv_acc if form == "rowmap" else None,
        )

    def leaves():
        return [x[t].clone().requires_grad_() for t in ("q", "k", "v")]

    results = []
    for use_checkpoint in (False, True, False):
        dst["k"] = torch.zeros(n_dst, 4, D, device=device)
        dst["v"] = torch.zeros(n_dst, 4, D, device=device)
        q, k, v = leaves()
        if use_checkpoint:
            assert _ckpt_retained_mib(fn, q, k, v) <= CKPT_RETAINED_LIMIT_MIB
            out = checkpoint(fn, q, k, v, use_reentrant=False)
        else:
            out = fn(q, k, v)
        out.backward(x["dout"])
        torch.cuda.synchronize()
        grads = (
            (q.grad, dst["k"].clone(), dst["v"].clone())
            if form == "rowmap"
            else (q.grad, k.grad, v.grad)
        )
        results.append((out.detach(), *grads))
    eager, ckpt, eager2 = results
    assert torch.equal(ckpt[0], eager[0])
    _assert_dq_close(ckpt[1], eager[1])
    if form == "bf16":
        assert torch.equal(ckpt[2], eager[2]) and torch.equal(ckpt[3], eager[3])
    else:
        # Repeated destination rows are added in a scheduling order: eager-to-eager is the spread to compare with.
        for i in (2, 3):
            spread = (eager2[i] - eager[i]).abs().max()
            assert (ckpt[i] - eager[i]).abs().max() <= max(
                spread * 4, 1e-5 * eager[i].abs().max()
            )


def test_steady_state_does_not_synchronize(device) -> None:
    xs = [
        _inputs(q, k, 24, 4, device, seed=i)
        for i, (q, k) in enumerate((((700, 77), (3000, 77)), ((300,), (300,))))
    ]
    ws = cake_backend.Workspace(device)

    def step(x):
        q, k, v = (x[t].clone().requires_grad_() for t in ("q", "k", "v"))
        out = pkg.varlen_gqa_attention(
            q, k, v, x["cu_q"], x["cu_k"], softmax_scale=SCALE, workspace=ws
        )
        out.backward(x["dout"])
        return q.grad, k.grad, v.grad

    for x in xs:
        step(x)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for x in xs + xs:
            step(x)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


def _timed(call, *, side):
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


@pytest.mark.parametrize("stage", ("fwd", "bwd"))
def test_keeps_its_speed_beside_a_resident_kernel(stage, device) -> None:
    q_lens, k_lens = (3000, 1100), (3000, 17000)
    x = _inputs(q_lens, k_lens, 24, 4, device)
    ws = cake_backend.Workspace(device)
    out, lse = cake_backend.gqa_attention_fwd(
        x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], softmax_scale=SCALE
    )

    def call():
        if stage == "fwd":
            o, l = cake_backend.gqa_attention_fwd(
                x["q"],
                x["k"],
                x["v"],
                x["cu_q"],
                x["cu_k"],
                softmax_scale=SCALE,
                workspace=ws,
            )
            return (o, l)
        _dq, dk, dv = cake_backend.gqa_attention_bwd(
            x["q"],
            x["k"],
            x["v"],
            out,
            lse,
            x["dout"],
            x["cu_q"],
            x["cu_k"],
            softmax_scale=SCALE,
            workspace=ws,
        )
        return (dk, dv)

    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL {stage} alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={busy_ms / alone_ms:.3f}"
    )
    reference = alone[0][1]
    for _t, result in alone[1:] + busy:
        for got, want in zip(result, reference, strict=True):
            assert torch.equal(got, want), (
                f"{stage}: the result depends on the scheduling"
            )
    assert busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS, (
        f"{stage}: {busy_ms:.3f} ms beside a resident kernel vs {alone_ms:.3f} ms alone"
    )


def test_rejects_unsupported_inputs(device) -> None:
    x = _inputs((10,), (10,), 24, 4, device)
    q, k, v, cq, ck = x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"]
    with pytest.raises(ValueError, match="even multiple"):
        cake_backend.gqa_attention_fwd(q[:, :12], k[:, :4], v[:, :4], cq, ck)  # group 3
    with pytest.raises(ValueError, match="bfloat16"):
        cake_backend.gqa_attention_fwd(q.half(), k, v, cq, ck)
    with pytest.raises(ValueError, match="int32"):
        cake_backend.gqa_attention_fwd(q, k, v, cq.long(), ck)
    with pytest.raises(ValueError, match="num_segments"):
        cake_backend.gqa_attention_fwd(q, k, v, cq, ck[:1])
    with pytest.raises(NotImplementedError):
        pkg.varlen_gqa_attention(q, k, v, cq, ck, causal=False)
