from __future__ import annotations


import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from b12x._lib.intrinsics import as_grouped_scale_view, quantize_grouped_nvfp4_torch
from b12x.quantization.nvfp4._impl import (
    allocate_bf16_to_fp4_tma_outputs,
    compile_bf16_to_fp4_tma,
)
from b12x.quantization.nvfp4._tuning import TUNING, Nvfp4QuantizationQuery

from b12x.testing.reference.helpers import dequantize_grouped_nvfp4, require_b12x


def _reference(
    source: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = source.shape[0]
    packed, scale_view = quantize_grouped_nvfp4_torch(
        source.unsqueeze(0),
        torch.tensor([rows], dtype=torch.int32, device=source.device),
        global_scale,
    )
    # Invert as_grouped_scale_view's logical permutation to recover the
    # physical contiguous scale-storage contract consumed by SM120 GEMMs.
    scale_storage = (
        scale_view.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8).reshape(-1)
    )
    return packed, scale_storage


def _assert_exact_outputs(
    source: torch.Tensor,
    global_scale: torch.Tensor,
    packed_storage: torch.Tensor,
    scale_storage: torch.Tensor,
) -> None:
    packed_ref, scale_ref = _reference(source, global_scale)
    packed_actual = packed_storage.permute(1, 2, 0)
    assert torch.count_nonzero(packed_actual).item() > 0
    assert torch.count_nonzero(scale_storage).item() > 0
    torch.testing.assert_close(packed_actual, packed_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scale_storage, scale_ref, rtol=0.0, atol=0.0)

    low_nibble = packed_actual & 0x0F
    high_nibble = packed_actual >> 4
    low_negative_zero = ((low_nibble & 0x07) == 0) & ((low_nibble & 0x08) != 0)
    high_negative_zero = ((high_nibble & 0x07) == 0) & ((high_nibble & 0x08) != 0)
    assert torch.count_nonzero(low_negative_zero).item() == 0
    assert torch.count_nonzero(high_negative_zero).item() == 0
    assert torch.all(scale_storage <= 0x7E).item()

    scale_view = as_grouped_scale_view(
        scale_storage.view(1, -1),
        source.shape[0],
        source.shape[1],
    )
    dequant = dequantize_grouped_nvfp4(
        packed_storage,
        scale_view,
        source.shape[1],
        global_scale,
    )
    assert torch.isfinite(dequant).all().item()


_RESOURCE_SHAPES = [
    pytest.param(128, 128, 0.125, id="minimum-tile"),
    pytest.param(128, 256, 3.25, id="multi-k"),
    pytest.param(128, 4096, 0.5, id="prefill-m128-k4096"),
    pytest.param(512, 4096, 2500.0, id="prefill-m512-k4096"),
    pytest.param(2048, 4096, 31.75, id="prefill-m2048-k4096"),
    pytest.param(128, 7168, 0.03125, id="prefill-m128-k7168-subnormal"),
    pytest.param(512, 7168, 448.0, id="prefill-m512-k7168"),
]


def test_dsl_compile_option_provenance_is_fresh_process_stable(
    tmp_path: Path,
) -> None:
    require_b12x()
    script = r"""
import json
import hashlib
import os
from pathlib import Path
import sys

os.environ["B12X_COMPILE_CACHE_DIR"] = sys.argv[1]

import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel

from b12x._lib.compiler import (
    _compile_kwargs_json_key,
    _dsl_compile_options_kwargs_key,
)
from validation.cutlass_migration.core.comparison_identity import (
    comparison_semantic_key_from_manifest,
    normalize_comparison_compile_options,
    normalize_comparison_compile_environment,
)
from b12x.quantization.nvfp4._impl import (
    allocate_bf16_to_fp4_tma_outputs,
    compile_bf16_to_fp4_tma,
)
from b12x.quantization.nvfp4._tuning import TUNING, Nvfp4QuantizationQuery

result = {}
for level in (1, 2):
    compile_callable = cute.compile[OptLevel(level)]
    options_key = _dsl_compile_options_kwargs_key(compile_callable)
    kwargs_json, kwargs_hash = _compile_kwargs_json_key(
        {"__dsl_compile_options_key": options_key}
    )
    result[str(level)] = {
        "options": options_key,
        "kwargs_json": kwargs_json,
        "kwargs_hash": kwargs_hash,
    }

# Exercise the compile provenance in the same fresh process that compiles and
# launches a real migrated CuTe DSL kernel.  A CUDA availability check alone
# would leave this as a CPU-only test.
device = torch.device("cuda")
source = torch.ones((128, 128), dtype=torch.bfloat16, device=device)
global_scale = torch.ones((1,), dtype=torch.float32, device=device)
outputs = allocate_bf16_to_fp4_tma_outputs(128, 128, device=device)
launch = compile_bf16_to_fp4_tma(128, 128, liveness_strategy=TUNING.default_config(
    Nvfp4QuantizationQuery(dtype="bfloat16", rows=128, columns=128), None,
).liveness_strategy)
launch(source, global_scale, outputs.packed_a_flat, outputs.scale_flat)
torch.cuda.synchronize(device)
result["gpu_execution"] = {
    "capability": list(torch.cuda.get_device_capability(device)),
    "name": torch.cuda.get_device_name(device),
    "packed_nonzero": int(torch.count_nonzero(outputs.packed_a_flat).item()),
    "scale_nonzero": int(torch.count_nonzero(outputs.scale_flat).item()),
}
assert result["gpu_execution"]["packed_nonzero"] > 0
assert result["gpu_execution"]["scale_nonzero"] > 0

manifests = [
    json.loads(path.read_text())
    for path in sorted(Path(sys.argv[1]).glob("*/*.json"))
    if json.loads(path.read_text()).get("kernel_id")
    == "quantization.bf16_to_fp4_tma"
]
assert len(manifests) == 1
manifest = manifests[0]
raw_environment = dict(manifest["semantic_payload"]["compile_environment"])
assert raw_environment == dict(manifest["compile_environment"])
package_runtime_components = []
for component in raw_environment.get("CUTE_DSL_LIBS", "").split(os.pathsep):
    candidate = Path(component)
    if (
        candidate.name == "libcute_dsl_runtime.so"
        and "nvidia_cutlass_dsl" in candidate.parts
    ):
        package_runtime_components.append(component)
assert package_runtime_components
assert "rdc=false" in manifest["compile_options"]
assert manifest["semantic_payload"]["compile_options"] == manifest["compile_options"]
raw_semantic_json = json.dumps(
    manifest["semantic_payload"],
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
assert hashlib.sha256(raw_semantic_json.encode()).hexdigest() == manifest["semantic_key"]
comparison_key = comparison_semantic_key_from_manifest(manifest)
assert comparison_key != manifest["semantic_key"]

package_runtime = (
    "/tmp/site-packages/nvidia_cutlass_dsl/cu13/lib/"
    "libcute_dsl_runtime.so"
)
custom_runtime = "/opt/b12x-custom/libserving_runtime.so"
synthetic_raw_environment = [
    ["CUDA_PATH", "/opt/cuda"],
    ["CUTE_DSL_LIBS", package_runtime + os.pathsep + custom_runtime],
]
synthetic_environment = dict(normalize_comparison_compile_environment(
    synthetic_raw_environment
))
assert synthetic_environment["CUTE_DSL_LIBS"] == custom_runtime
assert dict(synthetic_raw_environment)["CUTE_DSL_LIBS"] == (
    package_runtime + os.pathsep + custom_runtime
)
synthetic_raw_options = [
    "opt-level=3",
    "rdc=false",
    "dump-ptx-path='/tmp/diagnostic-a.ptx'",
]
assert normalize_comparison_compile_options(synthetic_raw_options) == [
    "opt-level=3"
]
assert synthetic_raw_options[-1] == "dump-ptx-path='/tmp/diagnostic-a.ptx'"
result["semantic_environment"] = {
    "actual": raw_environment,
    "custom_runtime": synthetic_environment["CUTE_DSL_LIBS"],
    "comparison_key": comparison_key,
    "raw_semantic_key": manifest["semantic_key"],
}
print(json.dumps(result, sort_keys=True))
"""
    repo_root = Path(__file__).resolve().parents[4]
    cache_dir = tmp_path / "provenance-compile-cache"

    def run_fresh_process() -> str:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(cache_dir)],
            check=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip().splitlines()[-1]

    first = run_fresh_process()
    second = run_fresh_process()
    assert first == second
    assert "object at" not in first

    provenance = json.loads(first)
    assert "opt-level=1" in provenance["1"]["options"]
    assert "opt-level=2" in provenance["2"]["options"]
    for level in ("1", "2"):
        assert "rdc=false" in provenance[level]["options"]
        kwargs = json.loads(provenance[level]["kwargs_json"])
        assert kwargs["__dsl_compile_options_key"] == provenance[level]["options"]
        assert (
            provenance[level]["kwargs_hash"]
            == hashlib.sha256(provenance[level]["kwargs_json"].encode()).hexdigest()
        )
    assert provenance["gpu_execution"]["capability"] == [12, 0]
    assert provenance["gpu_execution"]["packed_nonzero"] > 0
    assert provenance["gpu_execution"]["scale_nonzero"] > 0


def test_bf16_to_fp4_tma_compile_identity_separates_strategy_and_mac(
    tmp_path: Path,
) -> None:
    require_b12x()
    cache_dir = tmp_path / "compile-cache"
    script = r"""
import json
import os
from pathlib import Path
import sys

os.environ["B12X_COMPILE_CACHE_DIR"] = sys.argv[1]

import torch

import b12x.quantization.nvfp4._impl as quantization
from b12x._lib.compiler import clear_compile_cache, compile_cache_info
from b12x.quantization.nvfp4._tuning import TUNING, Nvfp4QuantizationQuery

clear_compile_cache()
quantization._KERNEL_CACHE.clear()
records = []
for M, K, mac, expected_strategy in (
    (128, 128, 188, "retain"),
    (512, 4096, 187, "packed"),
):
    quantization.get_max_active_clusters = lambda _cluster, mac=mac: mac
    quantization.get_num_sm = lambda _device, mac=mac: mac
    source = torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    outputs = quantization.allocate_bf16_to_fp4_tma_outputs(M, K)
    launch = quantization.compile_bf16_to_fp4_tma(M, K, liveness_strategy=TUNING.default_config(
        Nvfp4QuantizationQuery(dtype="bfloat16", rows=M, columns=K), None,
    ).liveness_strategy)
    launch(source, global_scale, outputs.packed_a_flat, outputs.scale_flat)
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(outputs.packed_a_flat).item()) > 0
    assert int(torch.count_nonzero(outputs.scale_flat).item()) > 0
    records.append((M, K, expected_strategy, mac))

manifests = []
for path in sorted(Path(sys.argv[1]).glob("*/*.json")):
    manifest = json.loads(path.read_text())
    if manifest.get("kernel_id") == "quantization.bf16_to_fp4_tma":
        manifests.append(manifest)
assert len(manifests) == 2
assert len({manifest["cache_key"] for manifest in manifests}) == 2
assert len({manifest["semantic_key"] for manifest in manifests}) == 2
actual_facts = {
    tuple(tuple(item) for item in json.loads(manifest["compile_spec_json"])["facts"])
    for manifest in manifests
}
expected_facts = {
    (("M", M), ("K", K), ("liveness_strategy", strategy), ("mac", mac))
    for M, K, strategy, mac in records
}
assert actual_facts == expected_facts
assert {manifest["compile_spec_version"] for manifest in manifests} == {2}
info = compile_cache_info()
assert info["compile_misses"] == 2
print(json.dumps({
    "cache_keys": sorted(manifest["cache_key"] for manifest in manifests),
    "semantic_keys": sorted(manifest["semantic_key"] for manifest in manifests),
    "facts": sorted([list(facts) for facts in actual_facts]),
    "gpu_launches": len(records),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(cache_dir)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence["gpu_launches"] == 2
    assert len(evidence["cache_keys"]) == 2
    assert len(evidence["semantic_keys"]) == 2


def _random_source(
    device: torch.device,
    M: int,
    K: int,
    *,
    seed: int,
    divisor: float,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (
        torch.randn((M, K), generator=gen, dtype=torch.float32)
        .to(device)
        .div_(divisor)
        .to(torch.bfloat16)
        .contiguous()
    )


@pytest.mark.parametrize(("M", "K", "global_scale_value"), _RESOURCE_SHAPES)
def test_bf16_to_fp4_tma_eager_exact(
    M: int,
    K: int,
    global_scale_value: float,
) -> None:
    device = require_b12x()
    source = _random_source(
        device,
        M,
        K,
        seed=91_700 + M + K,
        divisor=4.0,
    )
    global_scale = torch.tensor(
        [global_scale_value],
        dtype=torch.float32,
        device=device,
    )
    outputs = allocate_bf16_to_fp4_tma_outputs(M, K, device=device)
    launch = compile_bf16_to_fp4_tma(
        M,
        K,
        liveness_strategy=TUNING.default_config(
            Nvfp4QuantizationQuery(dtype="bfloat16", rows=M, columns=K), None
        ).liveness_strategy,
    )

    assert outputs.packed_a_flat.numel() == M * K // 2
    assert outputs.scale_flat.numel() == M * K // 16
    assert not outputs.packed_a_storage.is_set_to(outputs.scale_storage)

    launch(source, global_scale, outputs.packed_a_flat, outputs.scale_flat)
    torch.cuda.synchronize(device)
    _assert_exact_outputs(
        source,
        global_scale,
        outputs.packed_a_storage,
        outputs.scale_flat,
    )


@pytest.mark.parametrize(("M", "K", "global_scale_value"), _RESOURCE_SHAPES)
def test_bf16_to_fp4_tma_graph_replay_exact(
    M: int,
    K: int,
    global_scale_value: float,
) -> None:
    device = require_b12x()
    source = _random_source(
        device,
        M,
        K,
        seed=91_800 + M + K,
        divisor=4.0,
    )
    global_scale = torch.tensor(
        [global_scale_value],
        dtype=torch.float32,
        device=device,
    )
    guard_bytes = 256
    packed_bytes = M * K // 2
    scale_bytes = M * K // 16
    packed_guard = 0xD3
    scale_guard = 0x6D
    packed_backing = torch.full(
        (packed_bytes + 2 * guard_bytes,),
        packed_guard,
        dtype=torch.uint8,
        device=device,
    )
    scale_backing = torch.full(
        (scale_bytes + 2 * guard_bytes,),
        scale_guard,
        dtype=torch.uint8,
        device=device,
    )
    packed_flat = packed_backing[guard_bytes : guard_bytes + packed_bytes]
    scale_flat = scale_backing[guard_bytes : guard_bytes + scale_bytes]
    packed_storage = packed_flat.view(1, M, K // 2)
    launch = compile_bf16_to_fp4_tma(
        M,
        K,
        liveness_strategy=TUNING.default_config(
            Nvfp4QuantizationQuery(dtype="bfloat16", rows=M, columns=K), None
        ).liveness_strategy,
    )
    packed_ptr = packed_flat.data_ptr()
    scale_ptr = scale_flat.data_ptr()
    initial_source = source.clone()
    initial_global_scale = global_scale.clone()

    # Compile and initialize all runtime state before capture. The graph must
    # reuse caller-owned input/output storage while observing mutated values.
    launch(source, global_scale, packed_flat, scale_flat)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(source, global_scale, packed_flat, scale_flat)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(source, initial_source, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        global_scale,
        initial_global_scale,
        rtol=0.0,
        atol=0.0,
    )

    replay_source = _random_source(
        device,
        M,
        K,
        seed=91_900 + M + K,
        divisor=3.0,
    )
    source.copy_(replay_source)
    global_scale.fill_(global_scale_value * 0.75)
    replay_source_snapshot = source.clone()
    replay_global_scale_snapshot = global_scale.clone()
    packed_flat.fill_(0xA5)
    scale_flat.fill_(0x5A)
    graph.replay()
    torch.cuda.synchronize(device)

    assert packed_flat.data_ptr() == packed_ptr
    assert scale_flat.data_ptr() == scale_ptr
    torch.testing.assert_close(
        source,
        replay_source_snapshot,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        global_scale,
        replay_global_scale_snapshot,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(packed_backing[:guard_bytes] == packed_guard).item()
    assert torch.all(packed_backing[-guard_bytes:] == packed_guard).item()
    assert torch.all(scale_backing[:guard_bytes] == scale_guard).item()
    assert torch.all(scale_backing[-guard_bytes:] == scale_guard).item()
    _assert_exact_outputs(
        source,
        global_scale,
        packed_storage,
        scale_flat,
    )


def test_bf16_to_fp4_tma_fp8_scale_boundaries_graph_exact() -> None:
    device = require_b12x()
    M = K = 128
    # Each group of 16 values targets one E4M3 scale encoding. This spans
    # zero, two subnormal values, the minimum normal value, and both sides of
    # the 7->8 (subnormal->normal) and 8->9 rounding transitions.
    block_values = torch.tensor(
        [
            0.0,
            0.01171875,
            0.03515625,
            0.09375,
            0.08740234375,
            0.08837890625,
            0.09912109375,
            0.10009765625,
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    source = block_values.repeat_interleave(16).expand(M, K).clone().contiguous()
    # A nonzero block below half the minimum E4M3 subnormal must canonicalize
    # to scale byte 0 and packed payload 0, rather than a saturated payload.
    source[1, :16] = 0.001953125
    global_scale = torch.ones((1,), dtype=torch.float32, device=device)
    outputs = allocate_bf16_to_fp4_tma_outputs(M, K, device=device)
    launch = compile_bf16_to_fp4_tma(
        M,
        K,
        liveness_strategy=TUNING.default_config(
            Nvfp4QuantizationQuery(dtype="bfloat16", rows=M, columns=K), None
        ).liveness_strategy,
    )
    packed_ptr = outputs.packed_a_flat.data_ptr()
    scale_ptr = outputs.scale_flat.data_ptr()
    initial_source = source.clone()
    initial_global_scale = global_scale.clone()

    launch(
        source,
        global_scale,
        outputs.packed_a_flat,
        outputs.scale_flat,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(source, initial_source, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        global_scale,
        initial_global_scale,
        rtol=0.0,
        atol=0.0,
    )
    _assert_exact_outputs(
        source,
        global_scale,
        outputs.packed_a_storage,
        outputs.scale_flat,
    )

    expected_first_row_codes = torch.tensor(
        [0, 1, 3, 8, 7, 8, 8, 9],
        dtype=torch.uint8,
        device=device,
    )
    first_row_codes = (
        (source[0].float().view(K // 16, 16).abs().amax(dim=-1) / 6.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    torch.testing.assert_close(
        first_row_codes,
        expected_first_row_codes,
        rtol=0.0,
        atol=0.0,
    )
    assert outputs.scale_flat[16].item() == 0
    assert torch.count_nonzero(outputs.packed_a_storage[0, 1, :8]).item() == 0

    packed_ref, scale_ref = _reference(source, global_scale)
    actual_scale_view = as_grouped_scale_view(
        outputs.scale_flat.view(1, -1),
        M,
        K,
    )
    reference_scale_view = as_grouped_scale_view(
        scale_ref.view(1, -1),
        M,
        K,
    )
    actual_dequant = dequantize_grouped_nvfp4(
        outputs.packed_a_storage,
        actual_scale_view,
        K,
        global_scale,
    )
    reference_dequant = dequantize_grouped_nvfp4(
        packed_ref.permute(2, 0, 1).contiguous(),
        reference_scale_view,
        K,
        global_scale,
    )
    assert torch.count_nonzero(actual_dequant[0, 1, :16]).item() == 0
    torch.testing.assert_close(
        actual_dequant,
        reference_dequant,
        rtol=0.0,
        atol=0.0,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(source, global_scale, outputs.packed_a_flat, outputs.scale_flat)
    torch.cuda.synchronize(device)

    # Exercise the runtime zero-global-scale branch through a captured launch,
    # while retaining the exact same caller-owned input/output allocations.
    global_scale.zero_()
    replay_source_snapshot = source.clone()
    replay_global_scale_snapshot = global_scale.clone()
    outputs.packed_a_flat.fill_(0xFF)
    outputs.scale_flat.fill_(0xFF)
    graph.replay()
    torch.cuda.synchronize(device)

    packed_ref, scale_ref = _reference(source, global_scale)
    packed_actual = outputs.packed_a_storage.permute(1, 2, 0)
    assert outputs.packed_a_flat.data_ptr() == packed_ptr
    assert outputs.scale_flat.data_ptr() == scale_ptr
    torch.testing.assert_close(
        source,
        replay_source_snapshot,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        global_scale,
        replay_global_scale_snapshot,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(packed_actual).item() == 0
    assert torch.count_nonzero(outputs.scale_flat).item() == 0
    torch.testing.assert_close(packed_actual, packed_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs.scale_flat, scale_ref, rtol=0.0, atol=0.0)


def test_bf16_to_fp4_tma_rejects_invalid_capacity_and_aliasing() -> None:
    device = require_b12x()
    with pytest.raises(ValueError, match="multiples"):
        compile_bf16_to_fp4_tma(
            127,
            128,
            liveness_strategy=TUNING.default_config(
                Nvfp4QuantizationQuery(dtype="bfloat16", rows=127, columns=128), None
            ).liveness_strategy,
        )
    with pytest.raises(ValueError, match="multiples"):
        allocate_bf16_to_fp4_tma_outputs(128, 192, device=device)

    M = K = 128
    source = torch.ones((M, K), dtype=torch.bfloat16, device=device)
    global_scale = torch.ones((1,), dtype=torch.float32, device=device)
    launch = compile_bf16_to_fp4_tma(
        M,
        K,
        liveness_strategy=TUNING.default_config(
            Nvfp4QuantizationQuery(dtype="bfloat16", rows=M, columns=K), None
        ).liveness_strategy,
    )
    packed_bytes = M * K // 2
    scale_bytes = M * K // 16
    backing = torch.empty(packed_bytes + scale_bytes, dtype=torch.uint8, device=device)
    packed = backing[:packed_bytes]

    with pytest.raises(ValueError, match="shape"):
        launch(source, global_scale, packed, backing[: scale_bytes - 1])
    with pytest.raises(ValueError, match="must not overlap"):
        launch(source, global_scale, packed, backing[:scale_bytes])

    outputs = allocate_bf16_to_fp4_tma_outputs(M, K, device=device)
    input_aliased_packed = source.view(torch.uint8).reshape(-1)[:packed_bytes]
    with pytest.raises(ValueError, match="bf16_input must not overlap"):
        launch(
            source,
            global_scale,
            input_aliased_packed,
            outputs.scale_flat,
        )

    output_aliased_global_scale = outputs.packed_a_flat[:4].view(torch.float32)
    with pytest.raises(ValueError, match="global_scale must not overlap"):
        launch(
            source,
            output_aliased_global_scale,
            outputs.packed_a_flat,
            outputs.scale_flat,
        )

    # Retain the argument-validation assertions, but finish with a valid launch
    # so this migration test also exercises the physical GPU implementation.
    launch(source, global_scale, outputs.packed_a_flat, outputs.scale_flat)
    torch.cuda.synchronize(device)
    _assert_exact_outputs(
        source,
        global_scale,
        outputs.packed_a_storage,
        outputs.scale_flat,
    )
