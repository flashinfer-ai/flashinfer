"""Session-scoped JIT pre-warm for the checkpointing-SSU suite.

Each test lazily JIT-compiles its module URI (~2 min, mostly ONE cicc at a
time), so after any kernel-header edit a full run spends 30-40 min in
SEQUENTIAL rebuilds.  This fixture prebuilds the suite's whole URI matrix in
parallel — on a 28-core box the rebuild takes ~10 min, and cold CI gets the
same win.

Each spec is built via its own ``JitSpec.build()`` (ninja rooted at the URI
dir) — the SAME invocation ``build_and_load()`` uses at import time — NOT via
``build_jit_specs``'s single batch graph.  The batch grapher keeps a separate
ninja db at the JIT root; two dbs over the same outputs invalidate each other
("stored deps info out of date"), so batch-prebuilt modules were fully
recompiled AGAIN at load time, once per test, every session.  With per-spec
builds there is one authoritative db per URI and the load-time ninja is a
~100ms no-op.  Parallelism comes from a pool of ninjas (each URI is only ~3
TUs, so a pool of ~nproc/3 keeps ~nproc nvcc's in flight, matching what the
batch graph achieved).

The matrix below is DATA, regenerable from a warm cache:
    ls $FLASHINFER_JIT_DIR | grep checkpointing_ssu_ | <parse the URI fields>
Rows missing from it simply build lazily as before; stale rows cost one no-op
ninja edge.  Disable with FLASHINFER_TEST_WARM_JIT=0.
"""

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

# (state, dt, weight, npredicted, max_window, hpg, ng, state_scale, philox, pdl);
# input=bf16, matrixA=f32, stateIndex=i32, dim=64, dstate=128 throughout.
_WARM_MATRIX = [
    ("bf16", "bf16", "bf16", 4, 8, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 4, 16, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 6, 6, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 6, 6, 16, 1, "-", 0, 1),
    ("bf16", "bf16", "bf16", 6, 8, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 6, 8, 16, 1, "-", 0, 1),
    ("bf16", "bf16", "bf16", 6, 16, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 8, 16, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 10, 16, 16, 1, "-", 0, 0),
    ("bf16", "bf16", "bf16", 10, 16, 16, 1, "-", 0, 1),
    ("bf16", "bf16", "bf16", 16, 16, 8, 2, "-", 0, 0),
    ("bf16", "bf16", "bf16", 16, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "bf16", 6, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 4, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 6, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 6, 16, 16, 1, "-", 0, 1),
    ("bf16", "f32", "f32", 8, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 10, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 12, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 14, 16, 16, 1, "-", 0, 0),
    ("bf16", "f32", "f32", 16, 16, 16, 1, "-", 0, 0),
    ("e4m3", "bf16", "bf16", 4, 16, 16, 1, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 6, 6, 16, 1, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 6, 6, 16, 1, "f32", 5, 0),
    ("e4m3", "bf16", "bf16", 6, 6, 16, 1, "f32", 5, 1),
    ("e4m3", "bf16", "bf16", 6, 6, 16, 1, "f32", 10, 0),
    ("e4m3", "bf16", "bf16", 6, 6, 16, 2, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 6, 6, 16, 2, "f32", 10, 0),
    ("e4m3", "bf16", "bf16", 8, 16, 16, 1, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 8, 16, 64, 1, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 16, 16, 16, 1, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 16, 16, 16, 1, "f32", 10, 0),
    ("e4m3", "bf16", "bf16", 16, 16, 16, 2, "f32", 0, 0),
    ("e4m3", "bf16", "bf16", 16, 16, 16, 2, "f32", 10, 0),
    ("f16", "bf16", "bf16", 3, 16, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 4, 8, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 4, 8, 16, 1, "-", 10, 0),
    ("f16", "bf16", "bf16", 4, 8, 16, 2, "-", 10, 0),
    ("f16", "bf16", "bf16", 4, 16, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 6, 6, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 6, 6, 16, 1, "-", 10, 0),
    ("f16", "bf16", "bf16", 6, 6, 16, 2, "-", 0, 0),
    ("f16", "bf16", "bf16", 6, 6, 16, 2, "-", 10, 0),
    ("f16", "bf16", "bf16", 6, 16, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 6, 16, 16, 1, "-", 5, 0),
    ("f16", "bf16", "bf16", 8, 16, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 8, 16, 64, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 10, 16, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 10, 16, 16, 1, "-", 10, 0),
    ("f16", "bf16", "bf16", 10, 16, 16, 2, "-", 0, 0),
    ("f16", "bf16", "bf16", 10, 16, 16, 2, "-", 10, 0),
    ("f16", "bf16", "bf16", 14, 16, 16, 1, "-", 10, 0),
    ("f16", "bf16", "bf16", 14, 16, 16, 2, "-", 10, 0),
    ("f16", "bf16", "bf16", 16, 16, 16, 1, "-", 0, 0),
    ("f16", "bf16", "bf16", 16, 16, 16, 1, "-", 10, 0),
    ("f16", "bf16", "bf16", 16, 16, 16, 2, "-", 0, 0),
    ("f16", "bf16", "bf16", 16, 16, 16, 2, "-", 10, 0),
    ("f16", "f16", "f16", 6, 16, 16, 1, "-", 0, 0),
    ("f16", "f32", "f32", 6, 16, 16, 1, "-", 0, 1),
    ("f16", "f32", "f32", 6, 16, 16, 1, "-", 5, 1),
    ("f32", "bf16", "bf16", 4, 8, 16, 1, "-", 0, 0),
    ("f32", "bf16", "bf16", 4, 16, 16, 1, "-", 0, 0),
    ("f32", "bf16", "bf16", 6, 6, 16, 1, "-", 0, 0),
    ("f32", "bf16", "bf16", 6, 16, 16, 1, "-", 0, 0),
    ("f32", "bf16", "bf16", 8, 16, 16, 1, "-", 0, 0),
    ("f32", "bf16", "bf16", 10, 16, 16, 1, "-", 0, 0),
    ("f32", "bf16", "bf16", 16, 16, 16, 1, "-", 0, 0),
    ("f32", "f32", "f32", 6, 16, 16, 1, "-", 0, 0),
    ("f32", "f32", "f32", 6, 16, 16, 1, "-", 0, 1),
    ("f32", "f32", "f32", 8, 16, 16, 1, "-", 0, 0),
    ("f32", "f32", "f32", 10, 16, 16, 1, "-", 0, 0),
    ("i8", "bf16", "bf16", 1, 1, 16, 1, "f32", 0, 0),
    ("i8", "bf16", "bf16", 1, 1, 16, 1, "f32", 5, 0),
    ("i8", "bf16", "bf16", 4, 16, 16, 1, "f32", 0, 0),
    ("i8", "bf16", "bf16", 6, 6, 16, 1, "f32", 0, 0),
    ("i8", "bf16", "bf16", 6, 6, 16, 1, "f32", 10, 0),
    ("i8", "bf16", "bf16", 6, 6, 16, 2, "f32", 0, 0),
    ("i8", "bf16", "bf16", 6, 6, 16, 2, "f32", 10, 0),
    ("i8", "bf16", "bf16", 8, 16, 16, 1, "f32", 0, 0),
    ("i8", "bf16", "bf16", 16, 16, 16, 1, "f32", 0, 0),
    ("i8", "bf16", "bf16", 16, 16, 16, 1, "f32", 10, 0),
    ("i8", "bf16", "bf16", 16, 16, 16, 2, "f32", 0, 0),
    ("i8", "bf16", "bf16", 16, 16, 16, 2, "f32", 10, 0),
]


@pytest.fixture(scope="session", autouse=True)
def warm_checkpointing_jit():
    if os.environ.get("FLASHINFER_TEST_WARM_JIT", "1") == "0":
        yield
        return
    import torch

    from flashinfer.jit.mamba.checkpointing_ssu import gen_checkpointing_ssu_module

    dt = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
        "i8": torch.int8,
        "i16": torch.int16,
        "e4m3": torch.float8_e4m3fn,
    }
    specs = []
    for s, d, w, np_, mw, hpg, ng, sc, pr, pdl in _WARM_MATRIX:
        with contextlib.suppress(Exception):  # a bad row must not block the suite
            specs.append(
                gen_checkpointing_ssu_module(
                    state_dtype=dt[s],
                    input_dtype=torch.bfloat16,
                    dt_dtype=dt[d],
                    weight_dtype=dt[w],
                    matrixA_dtype=torch.float32,
                    stateIndex_dtype=torch.int32,
                    state_scale_dtype=None if sc == "-" else dt[sc],
                    dim=64,
                    dstate=128,
                    npredicted=np_,
                    max_window=mw,
                    heads_per_group=hpg,
                    num_groups=ng,
                    philox_rounds=pr,
                    enable_pdl=bool(pdl),
                )
            )

    def _build(spec):
        if spec.is_aot:
            return None
        try:
            spec.build(verbose=False)
        except Exception as exc:  # non-fatal: the lazy JIT path still works
            return f"{spec.name}: {exc}"
        return None

    workers = max(2, (os.cpu_count() or 8) // 3)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for err in pool.map(_build, specs):
            if err is not None:
                print(f"[warm-jit] prebuild failed (will retry lazily): {err}")
    yield
