"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Tests for flashinfer.fi_trace: definition JSON generation."""

import json
import torch

from flashinfer.fi_trace import fi_trace


# ---------------------------------------------------------------------------
# Helper: validate common fields of a definition dict
# ---------------------------------------------------------------------------


def _check_defn(defn, op_type, fi_api_substr):
    assert isinstance(defn, dict), "fi_trace must return a dict"
    assert defn["op_type"] == op_type, f"op_type mismatch: {defn['op_type']!r}"
    assert "name" in defn and isinstance(defn["name"], str) and defn["name"]
    assert "axes" in defn and isinstance(defn["axes"], dict)
    assert "inputs" in defn and isinstance(defn["inputs"], dict)
    assert "outputs" in defn and isinstance(defn["outputs"], dict)
    assert any(fi_api_substr in t for t in defn["tags"]), (
        f"Expected fi_api tag containing {fi_api_substr!r}, got {defn['tags']}"
    )
    # Must be round-trippable through JSON
    json.dumps(defn)


# ---------------------------------------------------------------------------
# rmsnorm
# ---------------------------------------------------------------------------


def test_rmsnorm_fi_trace():
    import flashinfer.norm

    hidden = torch.randn(32, 4096, dtype=torch.bfloat16)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    # Access via the function attribute
    defn = flashinfer.norm.rmsnorm.fi_trace(input=hidden, weight=weight)
    _check_defn(defn, "rmsnorm", "flashinfer.norm.rmsnorm")

    axes = defn["axes"]
    assert axes["batch_size"]["type"] == "var"
    assert axes["hidden_size"]["type"] == "const"
    assert axes["hidden_size"]["value"] == 4096

    assert defn["inputs"]["hidden_states"]["shape"] == ["batch_size", "hidden_size"]
    assert defn["inputs"]["weight"]["shape"] == ["hidden_size"]
    assert defn["outputs"]["output"]["shape"] == ["batch_size", "hidden_size"]
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"


def test_rmsnorm_fi_trace_via_helper():
    import flashinfer.norm

    hidden = torch.randn(16, 7168, dtype=torch.bfloat16)
    weight = torch.ones(7168, dtype=torch.bfloat16)

    defn = fi_trace(flashinfer.norm.rmsnorm, input=hidden, weight=weight)
    _check_defn(defn, "rmsnorm", "flashinfer.norm.rmsnorm")
    assert defn["axes"]["hidden_size"]["value"] == 7168


def test_fused_add_rmsnorm_fi_trace():
    import flashinfer.norm

    x = torch.randn(8, 5120, dtype=torch.bfloat16)
    res = torch.randn(8, 5120, dtype=torch.bfloat16)
    weight = torch.ones(5120, dtype=torch.bfloat16)

    defn = flashinfer.norm.fused_add_rmsnorm.fi_trace(
        input=x, residual=res, weight=weight
    )
    _check_defn(defn, "rmsnorm", "flashinfer.norm.fused_add_rmsnorm")
    assert defn["axes"]["hidden_size"]["value"] == 5120
    assert "residual" in defn["inputs"]
    assert "residual" in defn["outputs"]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_top_k_sampling_fi_trace():
    import flashinfer.sampling

    probs = torch.rand(64, 128256, dtype=torch.float32)
    top_k = torch.full((64,), 50, dtype=torch.int32)

    defn = flashinfer.sampling.top_k_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k
    )
    _check_defn(defn, "sampling", "top_k_sampling_from_probs")
    assert defn["axes"]["vocab_size"]["value"] == 128256
    assert defn["inputs"]["probs"]["shape"] == ["batch_size", "vocab_size"]
    assert defn["outputs"]["samples"]["dtype"] == "int64"


def test_top_p_sampling_fi_trace():
    import flashinfer.sampling

    probs = torch.rand(32, 151936, dtype=torch.float32)
    top_p = torch.full((32,), 0.9, dtype=torch.float32)

    defn = flashinfer.sampling.top_p_sampling_from_probs.fi_trace(
        probs=probs, top_p=top_p
    )
    _check_defn(defn, "sampling", "top_p_sampling_from_probs")
    assert defn["axes"]["vocab_size"]["value"] == 151936


def test_top_k_top_p_sampling_fi_trace():
    import flashinfer.sampling

    probs = torch.rand(16, 129280, dtype=torch.float32)
    top_k = torch.full((16,), 100, dtype=torch.int32)
    top_p = torch.full((16,), 0.9, dtype=torch.float32)

    defn = flashinfer.sampling.top_k_top_p_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k, top_p=top_p
    )
    _check_defn(defn, "sampling", "top_k_top_p_sampling_from_probs")
    assert defn["axes"]["vocab_size"]["value"] == 129280
    assert "top_k" in defn["inputs"]
    assert "top_p" in defn["inputs"]


# ---------------------------------------------------------------------------
# gemm
# ---------------------------------------------------------------------------


def test_mm_bf16_fi_trace():
    import flashinfer.gemm

    a = torch.randn(128, 4096, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16)

    defn = flashinfer.gemm.mm_bf16.fi_trace(a=a, b=b)
    _check_defn(defn, "gemm_bf16", "mm_bf16")
    assert defn["axes"]["N"]["value"] == 4096
    assert defn["axes"]["K"]["value"] == 4096
    assert defn["axes"]["M"]["type"] == "var"
    assert defn["inputs"]["A"]["shape"] == ["M", "K"]
    assert defn["inputs"]["B"]["shape"] == ["K", "N"]
    assert defn["outputs"]["C"]["shape"] == ["M", "N"]


# ---------------------------------------------------------------------------
# GQA paged decode
# ---------------------------------------------------------------------------


def test_gqa_paged_decode_fi_trace():
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    batch_size = 32
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_pages = 512
    page_size = 16

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )

    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        q=q, paged_kv_cache=(k_cache, v_cache)
    )
    _check_defn(defn, "gqa_paged", "BatchDecodeWithPagedKVCacheWrapper")
    axes = defn["axes"]
    assert axes["num_qo_heads"]["value"] == num_qo_heads
    assert axes["num_kv_heads"]["value"] == num_kv_heads
    assert axes["head_dim"]["value"] == head_dim
    assert axes["page_size"]["value"] == page_size
    assert axes["batch_size"]["type"] == "var"
    assert axes["num_pages"]["type"] == "var"

    assert "k_cache" in defn["inputs"]
    assert "v_cache" in defn["inputs"]
    assert defn["inputs"]["k_cache"]["shape"] == [
        "num_pages",
        "page_size",
        "num_kv_heads",
        "head_dim",
    ]


# ---------------------------------------------------------------------------
# GQA ragged prefill
# ---------------------------------------------------------------------------


def test_gqa_ragged_prefill_fi_trace():
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    total_q = 256
    total_kv = 512
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16)

    defn = BatchPrefillWithRaggedKVCacheWrapper.run.fi_trace(q=q, k=k, v=v)
    _check_defn(defn, "gqa_ragged", "BatchPrefillWithRaggedKVCacheWrapper")
    axes = defn["axes"]
    assert axes["num_qo_heads"]["value"] == num_qo_heads
    assert axes["num_kv_heads"]["value"] == num_kv_heads
    assert axes["head_dim"]["value"] == head_dim
    assert axes["total_q"]["type"] == "var"
    assert axes["total_kv"]["type"] == "var"

    assert "constraints" in defn


# ---------------------------------------------------------------------------
# MLA paged
# ---------------------------------------------------------------------------


def test_mla_paged_fi_trace():
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    batch_size = 16
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 256
    page_size = 64

    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16)

    defn = BatchMLAPagedAttentionWrapper.run.fi_trace(
        q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv_cache, kpe_cache=kpe_cache
    )
    _check_defn(defn, "mla_paged", "BatchMLAPagedAttentionWrapper")
    axes = defn["axes"]
    assert axes["num_qo_heads"]["value"] == num_qo_heads
    assert axes["head_dim_ckv"]["value"] == head_dim_ckv
    assert axes["head_dim_kpe"]["value"] == head_dim_kpe
    assert axes["page_size"]["value"] == page_size


# ---------------------------------------------------------------------------
# GDN decode
# ---------------------------------------------------------------------------


def test_gdn_decode_fi_trace():
    import flashinfer.gdn_decode

    B, H, HV, K = 4, 8, 16, 128

    q = torch.randn(B, 1, H, K, dtype=torch.bfloat16)
    k = torch.randn(B, 1, H, K, dtype=torch.bfloat16)
    v = torch.randn(B, 1, HV, K, dtype=torch.bfloat16)
    state = torch.zeros(B, HV, K, K, dtype=torch.float32)
    A_log = torch.zeros(HV, dtype=torch.float32)
    a = torch.zeros(B, 1, HV, dtype=torch.bfloat16)
    dt_bias = torch.zeros(HV, dtype=torch.float32)
    b = torch.zeros(B, 1, HV, dtype=torch.bfloat16)

    defn = flashinfer.gdn_decode.gated_delta_rule_decode.fi_trace(
        q=q, k=k, v=v, state=state, A_log=A_log, a=a, dt_bias=dt_bias, b=b
    )
    _check_defn(defn, "gdn", "gated_delta_rule_decode")
    axes = defn["axes"]
    assert axes["seq_len"]["value"] == 1
    assert axes["num_q_heads"]["value"] == H
    assert axes["num_v_heads"]["value"] == HV
    assert axes["head_size"]["value"] == K
    assert axes["batch_size"]["type"] == "var"


# ---------------------------------------------------------------------------
# Named tensor layer: verify refine_names is applied
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level fi_trace helper: bound method support
# ---------------------------------------------------------------------------


def test_fi_trace_helper_bound_method():
    """fi_trace() helper must work with a bound method via __func__ unwrapping."""
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    q = torch.randn(64, 32, 128, dtype=torch.bfloat16)
    k = torch.randn(128, 8, 128, dtype=torch.bfloat16)
    v = torch.randn(128, 8, 128, dtype=torch.bfloat16)

    # Create a dummy instance — we don't call run(), only fi_trace()
    class _FakeWrapper:
        run = BatchPrefillWithRaggedKVCacheWrapper.run

    instance = _FakeWrapper()
    # Accessing instance.run gives a bound method; fi_trace() must handle it
    defn = fi_trace(instance.run, q=q, k=k, v=v)
    _check_defn(defn, "gqa_ragged", "BatchPrefillWithRaggedKVCacheWrapper")


# ---------------------------------------------------------------------------
# End-to-end use case: simulate a Llama-3.1-8B decode step and produce a
# complete flashinfer-bench definition file ready to save to disk.
# ---------------------------------------------------------------------------


def test_usecase_llama31_decode_step(tmp_path):
    """
    Use case: profiling a Llama-3.1-8B decode step.

    A developer wants to benchmark their model's attention kernel. They run a
    forward pass with representative tensors, call fi_trace on the wrapper's
    .run method, and get back a JSON definition they can pass directly to
    flashinfer-bench -- without manually figuring out axis names or shapes.

    Model config (TP=1):
      num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=16
    Runtime:
      batch_size=64, num_pages=8192 (across all sequences in the batch)
    """
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    # ── Shapes matching a Llama-3.1-8B decode at batch_size=64 ──────────────
    batch_size = 64
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_pages = 8192
    page_size = 16

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16
    )

    # ── Generate the definition and write it to disk in one call ─────────────
    traces_dir = tmp_path / "benchmark_traces"
    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        save_dir=traces_dir,
        q=q,
        paged_kv_cache=(k_cache, v_cache),
    )

    # ── Validate the definition matches the flashinfer-bench schema ──────────
    _check_defn(defn, "gqa_paged", "BatchDecodeWithPagedKVCacheWrapper")

    # Variable axes have no "value"; const axes carry the model config.
    assert defn["axes"]["batch_size"]["type"] == "var"
    assert defn["axes"]["num_pages"]["type"] == "var"
    assert defn["axes"]["num_qo_heads"] == {"type": "const", "value": num_qo_heads}
    assert defn["axes"]["num_kv_heads"] == {"type": "const", "value": num_kv_heads}
    assert defn["axes"]["head_dim"] == {"type": "const", "value": head_dim}
    assert defn["axes"]["page_size"] == {"type": "const", "value": page_size}

    # Input shapes use axis names, not raw integers.
    assert defn["inputs"]["q"]["shape"] == ["batch_size", "num_qo_heads", "head_dim"]
    assert defn["inputs"]["k_cache"]["shape"] == [
        "num_pages",
        "page_size",
        "num_kv_heads",
        "head_dim",
    ]
    assert defn["inputs"]["k_cache"]["dtype"] == "bfloat16"

    # Output mirrors the query shape.
    assert defn["outputs"]["output"]["shape"] == [
        "batch_size",
        "num_qo_heads",
        "head_dim",
    ]
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"
    assert defn["outputs"]["lse"]["shape"] == ["batch_size", "num_qo_heads"]
    assert defn["outputs"]["lse"]["dtype"] == "float32"

    # ── The JSON file was written to disk ────────────────────────────────────
    json_file = traces_dir / f"{defn['name']}.json"
    assert json_file.exists(), f"Expected definition file at {json_file}"
    on_disk = json.loads(json_file.read_text())
    assert on_disk["axes"]["num_qo_heads"]["value"] == num_qo_heads

    assert json.loads(json_file.read_text())["axes"]["num_qo_heads"]["value"] == 32


def test_usecase_deepseek_mla_decode():
    """
    Use case: profiling a DeepSeek-V3 MLA decode step (TP=8).

    Model config (TP=8):
      num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64, page_size=64
    """
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    batch_size = 128  # tokens in the decode batch
    num_qo_heads = 16  # after TP=8 split
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 4096
    page_size = 64

    q_nope = torch.randn(batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16)
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16)
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16)

    defn = BatchMLAPagedAttentionWrapper.run.fi_trace(
        q_nope=q_nope,
        q_pe=q_pe,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
    )

    _check_defn(defn, "mla_paged", "BatchMLAPagedAttentionWrapper")

    assert defn["axes"]["num_qo_heads"]["value"] == num_qo_heads
    assert defn["axes"]["head_dim_ckv"]["value"] == head_dim_ckv
    assert defn["axes"]["head_dim_kpe"]["value"] == head_dim_kpe
    assert defn["axes"]["page_size"]["value"] == page_size
    assert defn["axes"]["batch_size"]["type"] == "var"

    # The output uses the CKV head dimension (not KPE).
    assert defn["outputs"]["output"]["shape"] == [
        "batch_size",
        "num_qo_heads",
        "head_dim_ckv",
    ]

    # Enrich with model metadata, then round-trip through JSON.
    defn["tags"] += ["model:deepseek-v3", "model:deepseek-r1", "tp:8", "stage:decode"]
    assert json.loads(json.dumps(defn))["axes"]["head_dim_ckv"]["value"] == 512


def test_usecase_sampling_vocab_discovery():
    """
    Use case: automatically discover the vocabulary size from live tensors.
    """
    import flashinfer.sampling

    # Qwen3 vocabulary size
    vocab_size = 151936
    batch_size = 32

    probs = torch.rand(batch_size, vocab_size, dtype=torch.float32)
    top_k = torch.full((batch_size,), 40, dtype=torch.int32)
    top_p = torch.full((batch_size,), 0.95, dtype=torch.float32)

    defn = flashinfer.sampling.top_k_top_p_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k, top_p=top_p
    )

    # vocab_size is automatically discovered from the probs tensor shape.
    assert defn["axes"]["vocab_size"]["type"] == "const"
    assert defn["axes"]["vocab_size"]["value"] == vocab_size

    # The definition name embeds the const axes values.
    assert str(vocab_size) in defn["name"]

    # Confirm the JSON is ready for flashinfer-bench.
    parsed = json.loads(json.dumps(defn))
    assert parsed["inputs"]["probs"]["dtype"] == "float32"
    assert parsed["outputs"]["samples"]["dtype"] == "int64"


# ---------------------------------------------------------------------------
# JSON file output
# ---------------------------------------------------------------------------


def test_fi_trace_writes_json_file(tmp_path):
    """fi_trace writes a <name>.json file when save_dir is given."""
    import flashinfer.norm

    hidden = torch.randn(16, 4096, dtype=torch.bfloat16)
    weight = torch.ones(4096, dtype=torch.bfloat16)

    defn = flashinfer.norm.rmsnorm.fi_trace(
        save_dir=tmp_path, input=hidden, weight=weight
    )

    expected_file = tmp_path / f"{defn['name']}.json"
    assert expected_file.exists(), f"Expected JSON file at {expected_file}"

    on_disk = json.loads(expected_file.read_text())
    assert on_disk == defn


def test_fi_trace_helper_writes_json_file(tmp_path):
    """The module-level fi_trace() helper threads save_dir through correctly."""
    import flashinfer.norm

    hidden = torch.randn(8, 7168, dtype=torch.bfloat16)
    weight = torch.ones(7168, dtype=torch.bfloat16)

    defn = fi_trace(
        flashinfer.norm.rmsnorm,
        save_dir=tmp_path,
        input=hidden,
        weight=weight,
    )

    expected_file = tmp_path / f"{defn['name']}.json"
    assert expected_file.exists()
    on_disk = json.loads(expected_file.read_text())
    assert on_disk["axes"]["hidden_size"]["value"] == 7168


def test_fi_trace_env_var_writes_json_file(tmp_path, monkeypatch):
    """FLASHINFER_TRACE_DUMP_DIR env-var (shared with logging) triggers file writing without save_dir."""
    import flashinfer.sampling

    # Use the real env-var; the template reads os.environ at call time.
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))

    probs = torch.rand(4, 128256, dtype=torch.float32)
    top_k = torch.full((4,), 50, dtype=torch.int32)

    defn = flashinfer.sampling.top_k_sampling_from_probs.fi_trace(
        probs=probs, top_k=top_k
    )

    expected_file = tmp_path / f"{defn['name']}.json"
    assert expected_file.exists(), f"Expected file {expected_file}"
    assert json.loads(expected_file.read_text())["op_type"] == "sampling"


def test_fi_trace_creates_nested_save_dir(tmp_path):
    """save_dir is created automatically even if it doesn't exist yet."""
    import flashinfer.norm

    nested = tmp_path / "traces" / "rmsnorm"
    assert not nested.exists()

    hidden = torch.randn(4, 2048, dtype=torch.bfloat16)
    weight = torch.ones(2048, dtype=torch.bfloat16)

    defn = flashinfer.norm.rmsnorm.fi_trace(
        save_dir=nested, input=hidden, weight=weight
    )

    assert nested.exists()
    files = list(nested.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["name"] == defn["name"]


def test_fi_trace_filename_matches_definition_name(tmp_path):
    """The written filename is exactly '<definition_name>.json'."""
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    q = torch.randn(4, 32, 128, dtype=torch.bfloat16)
    k_cache = torch.randn(64, 16, 8, 128, dtype=torch.bfloat16)
    v_cache = torch.randn(64, 16, 8, 128, dtype=torch.bfloat16)

    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        save_dir=tmp_path,
        q=q,
        paged_kv_cache=(k_cache, v_cache),
    )

    expected_name = defn["name"]
    expected_file = tmp_path / f"{expected_name}.json"
    assert expected_file.exists()
    assert json.loads(expected_file.read_text())["name"] == expected_name
