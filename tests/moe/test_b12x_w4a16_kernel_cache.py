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

Cache-contract tests for the SM12x W4A16 CuTe-DSL disk-cache adopters.

The kernel-name string is the sole per-kernel disk-cache key (the module
``meta.json`` guards only arch / DSL version / source hashes), so a name that
ignores a codegen parameter makes two different kernels collide on one
artifact. Unlike the direct-micro adopter's explicit-facts names
(``test_b12x_moe_kernel_cache.py``), the W4A16 names carry readable facts for
humans plus a sha256 digest of the entry's full ``cache_key`` -- the same key
that already gates the in-process caches -- so injectivity over the codegen
parameters is inherited from that digest rather than from the readable text.
These tests pin the properties that inheritance relies on:

1. Digest sensitivity: any perturbation of the cache key changes the name,
   including the float-sign and None-vs-0 cases that sanitized readable text
   alone would collide on.
2. Determinism: equal keys always produce the identical name.
3. Symbol safety: produced names are valid filename/symbol components.
4. On GPU, the disk layer round-trips: a compiled artifact is persisted,
   survives an in-process cache clear, is re-served from disk unchanged, and
   the warm-loaded kernel is CUDA-graph capturable.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("cutlass")

import torch  # noqa: E402

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (  # noqa: E402
    _CUTE_DSL_MODULE,
    _w4a16_disk_kernel_name,
    _w4a16_topk_sum_launch_flat,
    clear_w4a16_kernel_cache,
    compile_w4a16_activation,
    compile_w4a16_topk_sum,
    is_gated_moe_activation,
)

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9_]+$")

# A representative cache key mixing every value kind the W4A16 entries put in
# theirs: tags, dtype strings, ints, bools, floats and None.
_BASE_KEY = (
    "w4a16_activation",
    ("bf16", 768, "silu", True, True, 1.702, 1.0, None, 2),
)


def _perturbations(key):
    """Yield keys differing from ``key`` in exactly one nested element."""
    inner = key[1]
    yield ("w4a16_gemm", inner)  # tag change
    for idx, replacement in (
        (0, "fp16"),
        (1, 1536),
        (2, "relu2"),
        (3, False),
        (4, False),
        (5, -1.702),  # sign flip: sanitized text alone would collide
        (6, 2.0),
        (7, 7.0),  # None -> float
        (8, 1),
    ):
        mutated = list(inner)
        mutated[idx] = replacement
        yield (key[0], tuple(mutated))


def test_disk_name_digest_sensitivity():
    base = _w4a16_disk_kernel_name("activation", "bf16_i768_silu", _BASE_KEY)
    seen = {base}
    for perturbed_key in _perturbations(_BASE_KEY):
        name = _w4a16_disk_kernel_name("activation", "bf16_i768_silu", perturbed_key)
        assert name != base, (
            f"key perturbation did not change the name: {perturbed_key}"
        )
        assert name not in seen, f"two distinct keys collided on {name}"
        seen.add(name)


def test_disk_name_determinism():
    a = _w4a16_disk_kernel_name("fused", "bf16_e64_t2_h512_i256_b8", _BASE_KEY)
    b = _w4a16_disk_kernel_name(
        "fused", "bf16_e64_t2_h512_i256_b8", ("w4a16_activation", _BASE_KEY[1])
    )
    assert a == b, "equal keys must produce the identical artifact name"


def test_disk_name_prefix_separates_entries():
    facts = "bf16_t8_h2048"
    assert _w4a16_disk_kernel_name(
        "topk_sum", facts, _BASE_KEY
    ) != _w4a16_disk_kernel_name("activation", facts, _BASE_KEY)


def test_disk_name_symbol_safety():
    for prefix, facts in (
        ("topk_sum", "bf16_t8_h2048"),
        ("activation", "bf16_i768_silu"),
        ("fused", "bf16_e64_t2_h512_i256_b8"),
        ("gemm", "bf16_e64_n512_k512_t2_b8"),
    ):
        name = _w4a16_disk_kernel_name(prefix, facts, _BASE_KEY)
        assert _SYMBOL_RE.match(name), f"unsafe artifact name: {name!r}"


def _w4a16_module_artifacts():
    from flashinfer.jit import env as jit_env

    artifacts = {}
    root = jit_env.FLASHINFER_JIT_DIR
    if not root.is_dir():
        return artifacts
    for module_dir in root.glob(f"{_CUTE_DSL_MODULE}*"):
        for entry in module_dir.glob("*.o"):
            artifacts[entry.name] = entry.stat().st_mtime_ns
    return artifacts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_topk_sum_disk_round_trip():
    m, topk, hidden = 64, 8, 2048
    clear_w4a16_kernel_cache()
    compile_w4a16_topk_sum(m=m, topk=topk, hidden_size=hidden, element_dtype="bf16")
    after_cold = _w4a16_module_artifacts()
    assert any(name.startswith("topk_sum_") for name in after_cold), (
        "cold compile did not persist a topk_sum artifact"
    )

    # A cleared in-process cache forces the next compile through the disk
    # layer; the artifact set (names and mtimes) must be served unchanged.
    clear_w4a16_kernel_cache()
    compile_w4a16_topk_sum(m=m, topk=topk, hidden_size=hidden, element_dtype="bf16")
    after_warm = _w4a16_module_artifacts()
    assert after_warm == after_cold, "warm load rewrote or dropped artifacts"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_topk_sum_warm_artifact_numerics_and_graph_capture():
    m, topk, hidden = 64, 8, 2048
    tol = dict(rtol=1.6e-2, atol=1e-3)

    # Ensure the artifact exists on disk, then force the disk-load path.
    clear_w4a16_kernel_cache()
    compile_w4a16_topk_sum(m=m, topk=topk, hidden_size=hidden, element_dtype="bf16")
    clear_w4a16_kernel_cache()

    torch.manual_seed(0)
    fc2 = torch.randn(m * topk, hidden, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
    stream_int = int(torch.cuda.current_stream().cuda_stream)
    _w4a16_topk_sum_launch_flat(fc2, out, m, topk, hidden, "bf16", stream_int)
    torch.cuda.synchronize()
    ref = fc2.view(m, topk, hidden).float().sum(dim=1).to(torch.bfloat16)
    assert torch.allclose(out.float(), ref.float(), **tol)

    fc2_g = fc2.clone()
    out_g = torch.empty_like(out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _w4a16_topk_sum_launch_flat(
            fc2_g,
            out_g,
            m,
            topk,
            hidden,
            "bf16",
            int(torch.cuda.current_stream().cuda_stream),
        )
    fc2_g.copy_(torch.randn(m * topk, hidden, device="cuda", dtype=torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    ref_g = fc2_g.view(m, topk, hidden).float().sum(dim=1).to(torch.bfloat16)
    assert torch.allclose(out_g.float(), ref_g.float(), **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_activation_row_buckets_share_one_artifact(tmp_path, monkeypatch):
    """Both row-specialization buckets must resolve to one disk artifact.

    The activation cache key deliberately excludes the row bucket -- the
    in-process cache reuses one compiled kernel across rows via
    ``replace(cached, rows=rows)`` -- so under the extent policy the
    m-varying fake extents bake (1,) and a single artifact serves every
    rows value, warm-launchable at any row count.
    """
    from flashinfer.jit import env as jit_env

    # Isolate this test's artifacts from whatever the persistent JIT
    # directory has accumulated: the builder and the artifact scanner both
    # read jit_env.FLASHINFER_JIT_DIR dynamically, so patching it redirects
    # real .o writes, not just the scan.
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)

    inter = 768
    clear_w4a16_kernel_cache()
    compile_w4a16_activation(rows=1, intermediate_size=inter, activation="silu")
    after_bucket1 = _w4a16_module_artifacts()
    clear_w4a16_kernel_cache()
    compile_w4a16_activation(rows=64, intermediate_size=inter, activation="silu")
    after_bucket2 = _w4a16_module_artifacts()
    act_names = [n for n in after_bucket2 if n.startswith("activation_")]
    assert len(act_names) == 1, f"row buckets split into artifacts: {act_names}"
    assert after_bucket2 == after_bucket1, "second bucket rewrote the artifact"

    shards = 2 if is_gated_moe_activation("silu") else 1
    for rows in (1, 64):
        result = compile_w4a16_activation(
            rows=rows, intermediate_size=inter, activation="silu"
        )
        fc1 = torch.randn(rows * shards * inter, device="cuda", dtype=torch.bfloat16)
        activated = torch.zeros(rows * inter, device="cuda", dtype=torch.bfloat16)
        result.compiled(fc1[:1], activated[:1], rows)
        torch.cuda.synchronize()
        assert activated.abs().sum().item() > 0, f"no output written at rows={rows}"
