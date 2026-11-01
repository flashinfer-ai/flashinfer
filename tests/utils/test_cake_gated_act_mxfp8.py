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

Tests for the Cake fused SwiGLU + MXFP8 quantization training kernels
(forward and backward; SM100 / SM103 / SM107).

Every output byte -- the E4M3 codes of the rowwise and colwise values and the
E8M0 scale bytes in the swizzled 128 x 4 layout -- is compared exactly, with
no tolerance, against a plain-torch reference of the same math
(``tests/test_helpers/cake_gated_act_mxfp8_reference.py``).  Every program of
every route (forward / backward x rowwise / colwise / both) on the device's
architecture runs directly, whatever the selection rule would pick, on unit
normals, block-scaled normals, special values (NaN, Inf, zero and subnormal
blocks, overflowing products, gates around the reciprocal's 2^126 boundary)
and every BF16 bit pattern as a gate; the entry points run at sizes at which
the selection rule picks each program.  The suite also holds the selection rule,
argument validation, zero-size disabled outputs, CUDA Graph replay, the
absence of rebuilds on a changed shape and the ``build.json`` contract.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.gated_act_mxfp8`` facade is covered by
``test_cake_gated_act_mxfp8_api.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers import cake_gated_act_mxfp8_reference as reference
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_gated_act_mxfp8")
cake_build = cake_package("cake_gated_act_mxfp8", "cake_build")
loader = cake_package("cake_gated_act_mxfp8", "cake_jit")

ROUTES = {
    "fwd_row": (False, True, False),
    "fwd_col": (False, False, True),
    "fwd_both": (False, True, True),
    "bwd_row": (True, True, False),
    "bwd_col": (True, False, True),
    "bwd_both": (True, True, True),
}
# Entry-point shapes (M, K): M * K = 2^14, 2^16, 2^21, 2^22, 2^23, 2^24, 2^25,
# 2^26 and 2^28, at which the selection rule picks every program of each route
# on each architecture.
ENTRY_SHAPES = (
    (128, 128),
    (256, 256),
    (2048, 1024),
    (2048, 2048),
    (4096, 2048),
    (4096, 4096),
    (8192, 4096),
    (8192, 8192),
    (32768, 8192),
)
# (M, K): one tile row of scale tiles; odd counts of 128-column scale tiles.
PROGRAM_SHAPES = ((128, 128), (256, 384), (384, 256))


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(0))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and loader.arch_for_capability(_capability()) is not None
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_gated_act_mxfp8 needs an SM100 / SM103 / SM107 device",
)


def _arch() -> str:
    return loader.arch_for_capability(_capability())


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


def _expected(gated_input, grad_output, *, backward, rowwise, colwise):
    if backward:
        return reference.backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        )
    return reference.forward(gated_input, rowwise=rowwise, colwise=colwise)


def _run_program(name, gated_input, grad_output, *, backward, rowwise, colwise):
    """Run module ``name`` directly (not through the selection rule)."""

    M, K = pkg.check_inputs(
        gated_input,
        grad_output if backward else None,
        rowwise=rowwise,
        colwise=colwise,
    )
    outputs = pkg.empty_outputs(
        gated_input, backward=backward, rowwise=rowwise, colwise=colwise
    )
    row_output, col_output, row_scales, col_scales = outputs
    values = {
        "gated_input": gated_input,
        "row_output": row_output.view(torch.uint8),
        "col_output": (col_output.t() if colwise else col_output).view(torch.uint8),
        "row_scales": row_scales,
        "col_scales": col_scales,
        "M": M,
        "K": K,
    }
    if backward:
        values["grad_output"] = grad_output
    loader.run_module(name, values, device_index=0)
    return pkg.logical_outputs(outputs)


def _assert_bytes(outputs, expected, *, rowwise, colwise, what):
    got = reference.output_bytes(outputs, rowwise=rowwise, colwise=colwise)
    assert set(got) == set(expected), what
    for key, want in expected.items():
        mismatched = int((got[key] != want).sum())
        assert mismatched == 0, f"{what}: {key} differs in {mismatched} bytes"


@pytest.mark.parametrize("values", reference.VALUES)
@pytest.mark.parametrize("kind", tuple(ROUTES))
def test_every_program_matches_the_reference(kind, values) -> None:
    backward, rowwise, colwise = ROUTES[kind]
    shapes = ((256, 256),) if values == "bits" else PROGRAM_SHAPES
    for M, K in shapes:
        gated_input, grad_output = reference.make_inputs(
            M, K, values=values, seed=M + K
        )
        expected = _expected(
            gated_input,
            grad_output,
            backward=backward,
            rowwise=rowwise,
            colwise=colwise,
        )
        for name in loader.route_modules(_arch(), kind):
            outputs = _run_program(
                name,
                gated_input,
                grad_output,
                backward=backward,
                rowwise=rowwise,
                colwise=colwise,
            )
            _assert_bytes(
                outputs,
                expected,
                rowwise=rowwise,
                colwise=colwise,
                what=f"{kind} {name} M={M} K={K} {values}",
            )


@pytest.mark.parametrize("kind", tuple(ROUTES))
@pytest.mark.parametrize("shape", ENTRY_SHAPES)
def test_entry_points_match_the_reference(kind, shape) -> None:
    """The entry points through the selection rule, at shapes that select every
    program of the route on the device's architecture."""

    backward, rowwise, colwise = ROUTES[kind]
    M, K = shape
    gated_input, grad_output = reference.make_inputs(M, K, values="normal", seed=7)
    if backward:
        outputs = pkg.silu_and_mul_mxfp8_quantize_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        )
    else:
        outputs = pkg.silu_and_mul_mxfp8_quantize(
            gated_input, rowwise=rowwise, colwise=colwise
        )
    expected = _expected(
        gated_input, grad_output, backward=backward, rowwise=rowwise, colwise=colwise
    )
    _assert_bytes(outputs, expected, rowwise=rowwise, colwise=colwise, what=kind)


def test_output_layouts_and_disabled_outputs() -> None:
    M, K = 256, 384
    gated_input, grad_output = reference.make_inputs(M, K)
    for backward in (False, True):
        N = 2 * K if backward else K
        for rowwise, colwise in ((True, False), (False, True), (True, True)):
            if backward:
                outputs = pkg.silu_and_mul_mxfp8_quantize_backward(
                    gated_input, grad_output, rowwise=rowwise, colwise=colwise
                )
            else:
                outputs = pkg.silu_and_mul_mxfp8_quantize(
                    gated_input, rowwise=rowwise, colwise=colwise
                )
            row_output, col_output, row_scales, col_scales = outputs
            assert row_output.dtype == col_output.dtype == torch.float8_e4m3fn
            assert row_scales.dtype == col_scales.dtype == torch.float8_e8m0fnu
            for tensor in outputs:
                assert tensor.device == gated_input.device
            if rowwise:
                assert row_output.shape == (M, N) and row_output.stride() == (N, 1)
                assert row_scales.shape == (M, N // 32)
            else:
                assert row_output.numel() == 0 and row_scales.numel() == 0
            if colwise:
                assert col_output.shape == (M, N) and col_output.stride() == (1, M)
                assert col_scales.shape == (N * (M // 32),)
            else:
                assert col_output.numel() == 0 and col_scales.numel() == 0


def test_selection_rule() -> None:
    """Each route has one fallback per architecture (the record without a
    threshold there); of the records whose threshold ``M * K`` reaches, the one
    of the highest threshold runs."""

    for arch in loader.ARCHES:
        for kind in loader.KINDS:
            names = loader.route_modules(arch, kind)
            rules = {name: loader.select_rule(loader.MODULES[name]) for name in names}
            fallbacks = [
                name
                for name, rule in rules.items()
                if rule is None or arch not in rule["elements"]
            ]
            assert len(fallbacks) == 1, (arch, kind, fallbacks)
            positives = sorted(
                (int(rule["elements"][arch]), name)
                for name, rule in rules.items()
                if rule is not None and arch in rule["elements"]
            )
            thresholds = [threshold for threshold, _ in positives]
            assert thresholds == sorted(set(thresholds)), (arch, kind, positives)
            previous = fallbacks[0]
            for threshold, name in positives:
                assert threshold % (128 * 128) == 0, (arch, kind, threshold)
                K = 128
                below = threshold - 128 * 128
                assert loader.selected_module(arch, kind, M=threshold // K, K=K) == name
                assert loader.selected_module(arch, kind, M=below // K, K=K) == previous
                previous = name
            assert loader.selected_module(arch, kind, M=2**17, K=2**13) == previous
            assert loader.selected_module(arch, kind, M=128, K=128) == fallbacks[0]
            entry = {
                loader.selected_module(arch, kind, M=M, K=K) for M, K in ENTRY_SHAPES
            }
            assert entry == set(names), (arch, kind, sorted(set(names) - entry))


def test_argument_validation() -> None:
    M, K = 256, 256
    gated_input, grad_output = reference.make_inputs(M, K)
    forward = pkg.silu_and_mul_mxfp8_quantize
    backward = pkg.silu_and_mul_mxfp8_quantize_backward
    with pytest.raises(ValueError, match="CUDA tensor"):
        forward(gated_input.cpu())
    with pytest.raises(TypeError, match="bfloat16"):
        forward(gated_input.float())
    with pytest.raises(ValueError, match="contiguous"):
        forward(gated_input.t())
    with pytest.raises(ValueError, match="multiples of 128"):
        forward(gated_input[:200])
    with pytest.raises(ValueError, match="multiples of 128"):
        forward(gated_input[:, : 2 * 192].contiguous())
    shifted = torch.empty(M * 2 * K + 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="32-byte aligned"):
        forward(shifted[8:].view(M, 2 * K))
    with pytest.raises(ValueError, match="rowwise or colwise"):
        forward(gated_input, rowwise=False, colwise=False)
    with pytest.raises(ValueError, match="grad_output must be contiguous"):
        backward(gated_input, grad_output[:128])
    with pytest.raises(ValueError, match="grad_output must be contiguous"):
        backward(gated_input, grad_output.t())
    with pytest.raises(TypeError, match="bfloat16"):
        backward(gated_input, grad_output.float())
    with pytest.raises(ValueError, match="CUDA tensor"):
        backward(gated_input, grad_output.cpu())
    with pytest.raises(ValueError, match="CUDA tensor"):
        backward(gated_input, None)


def test_int32_index_bound() -> None:
    # The kernels index the last element of the input and of the backward
    # outputs, M * 2K - 1, in signed 32 bits: M * 2K = 2^31 is the largest
    # accepted size, and M * 2K = 2^31 + 2^15 (the next multiple of 128 * 256)
    # is rejected although its last gate element (M * 2K - K - 1) is in range.
    accepted = torch.empty(2**23, 256, dtype=torch.bfloat16, device="cuda")
    for rowwise, colwise in ((True, False), (False, True), (True, True)):
        m_k = pkg.check_inputs(accepted, None, rowwise=rowwise, colwise=colwise)
        assert m_k == (2**23, 128)
    del accepted
    M, K = 128, 2**23 + 128
    rejected = torch.empty(M, 2 * K, dtype=torch.bfloat16, device="cuda")
    assert M * 2 * K - K - 1 <= 2**31 - 1 < M * 2 * K - 1
    forward = pkg.silu_and_mul_mxfp8_quantize
    for rowwise, colwise in ((True, False), (False, True), (True, True)):
        with pytest.raises(ValueError, match="signed int32"):
            forward(rejected, rowwise=rowwise, colwise=colwise)


def test_cuda_graph_replays_match_eager() -> None:
    M, K = 512, 1024
    gated_input, grad_output = reference.make_inputs(M, K, values="scaled", seed=3)
    eager = [
        pkg.silu_and_mul_mxfp8_quantize(gated_input, rowwise=True, colwise=True),
        pkg.silu_and_mul_mxfp8_quantize_backward(
            gated_input, grad_output, rowwise=True, colwise=True
        ),
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            pkg.silu_and_mul_mxfp8_quantize(gated_input, rowwise=True, colwise=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [
            pkg.silu_and_mul_mxfp8_quantize(gated_input, rowwise=True, colwise=True),
            pkg.silu_and_mul_mxfp8_quantize_backward(
                gated_input, grad_output, rowwise=True, colwise=True
            ),
        ]
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for want, got in zip(eager, captured, strict=True):
            for a, b in zip(want, got, strict=True):
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_changing_shapes_do_not_rebuild() -> None:
    arch = _arch()
    for kind in loader.KINDS:
        for name in loader.route_modules(arch, kind):
            loader.load(name, arch)
    loaded_before = loader.load.cache_info().misses
    for M, K in ((128, 128), (256, 384), (4096, 2048), (1024, 7168), (8192, 8192)):
        gated_input, grad_output = reference.make_inputs(M, K)
        for backward, rowwise, colwise in ROUTES.values():
            if backward:
                pkg.silu_and_mul_mxfp8_quantize_backward(
                    gated_input, grad_output, rowwise=rowwise, colwise=colwise
                )
            else:
                pkg.silu_and_mul_mxfp8_quantize(
                    gated_input, rowwise=rowwise, colwise=colwise
                )
    torch.cuda.synchronize()
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed problem size rebuilt a module"
    )


def test_every_route_ships_on_every_architecture() -> None:
    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.KINDS), arch
        for kind in loader.KINDS:
            for name in loader.route_modules(arch, kind):
                assert loader.library_name(name, arch) in libraries, (arch, kind, name)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to
    compile; the manifest (schema ``cake-build/1``) carries exactly one entry per
    (library, architecture) the registry references, each with the kernel and
    binding sources under ``csrc/`` of this package and their digests."""

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    # The package the manifest was written for, also when this copy is imported
    # under another name.
    assert manifest["package"] == "flashinfer.cake_gated_act_mxfp8_backend"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    routed = {
        name
        for route in loader.ROUTES.values()
        for names in route.values()
        for name in names
    }
    assert set(loader.MODULES) == routed
    referenced = set()
    for arch, by_kind in loader.ROUTES.items():
        for kind, names in by_kind.items():
            assert kind in loader.KINDS, (arch, kind)
            for name in names:
                record = loader.MODULES[name]
                assert record["role"] == kind.split("_")[0], (arch, kind, name)
                assert record["route"] == kind.split("_")[1], (arch, kind, name)
                assert arch in record["arches"], (arch, kind, name)
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["role"] in loader.ROLES, name
        assert record["grid_rule"]["kind"] in loader.GRID_KINDS, name
        # 32 rows, or 64 or 128 for a program whose threads own two or four rows
        assert record["grid_rule"]["tile_rows"] in (32, 64, 128), name
        # 128 forward or 64 backward columns, or a narrower tile
        tile_cols = record["grid_rule"]["tile_cols"]
        widths = (128, 64, 32) if record["role"] == "fwd" else (64, 32)
        assert tile_cols in widths, name
        assert record["launch"]["cooperative"] is False, name
        assert record["launch"]["use_pdl"] is False, name
        # 256 threads for the forward's 128 columns, else 128
        assert record["launch"]["block"][0] == (
            256 if record["role"] == "fwd" and tile_cols == 128 else 128
        ), name
        plan = {key for kind, key in record["arg_plan"] if kind != "grid"}
        assert plan == set(record["logical"]), name
        assert set(record["logical"].values()) <= set(loader.LOGICAL_TENSORS) | set(
            loader.LOGICAL_SCALARS
        ), name
        assert ("grad_output" in record["logical"].values()) == (
            record["role"] == "bwd"
        ), name
        rule = loader.select_rule(record)
        if rule is not None:
            assert set(rule) == {"kind", "elements"}, name
            assert rule["kind"] in loader.SELECT_KINDS, name
            assert set(rule["elements"]) <= set(record["arches"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            assert library in libraries, f"{library} is not in build.json"
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
