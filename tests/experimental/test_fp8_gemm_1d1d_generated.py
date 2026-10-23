"""Prepared FP8 1D1D GEMM: random-input references at several shapes, both routes,
in-place accumulation, non-default-stream submission and CUDA-graph replay, the
host plan of the registry records and the ``build.json`` contract.

This suite exercises the package alone: it names the package once, through
``tests.test_helpers.cake_package``, and imports nothing else from flashinfer,
so a build that vendors the package under another import path runs it
unchanged after pointing that table entry at its package.
"""

import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("deepgemm_fp8_gemm")
_runtime = cake_package("deepgemm_fp8_gemm", "runtime")
loader = cake_package("deepgemm_fp8_gemm", "cake_jit")
cake_build = cake_package("deepgemm_fp8_gemm", "cake_build")

pack_ue8m0_words = pkg.pack_ue8m0_words
prepare_fp8_gemm_1d1d = pkg.prepare_fp8_gemm_1d1d

# The kernel's parameters in ``run`` order: the host vocabulary of every record.
PARAMETERS = (
    "A",
    "B",
    "SFA",
    "SFB",
    "C_tma",
    "M",
    "N",
    "K",
    "grid_m",
    "grid_n",
    "K_tiles",
)


def _capability():
    return (
        tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    )


def _supported():
    return (
        torch.cuda.is_available()
        and loader.arch_for_capability(_capability()) is not None
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        return _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries when they have no prebuilt file
    # before the first test, through the spec objects the package's loader itself
    # builds on first use (``cake_build.load_library``); a prebuilt or already
    # cached library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(loader.arch_for_capability(_capability())):
            spec.build_and_load()
    yield


def _ue8m0_exponent(amax):
    """UE8M0 exponent byte of the power-of-two scale covering ``amax`` at E4M3 range."""
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))
    return (scale.view(torch.int32) >> 23).to(torch.uint8), scale


def _quantize_rows(x, block_rows):
    """Cast BF16 ``x`` ``[R, K]`` to E4M3 with one UE8M0 scale per ``block_rows`` x 128 block."""
    rows, k = x.shape
    blocks = x.float().reshape(rows // block_rows, block_rows, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    exponent, scale = _ue8m0_exponent(amax)
    fp8 = (blocks / scale).to(torch.float8_e4m3fn)
    dequant = fp8.float() * scale
    return (
        fp8.reshape(rows, k).view(torch.uint8),
        exponent.reshape(rows // block_rows, k // 128),
        dequant.reshape(rows, k),
    )


def _case(M, N, K, accumulate, seed):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    pad_m = (-M) % 128
    pad_n = (-N) % 128
    a_u8, sfa_u8, a_deq = _quantize_rows(
        torch.nn.functional.pad(a, (0, 0, 0, pad_m)), 1
    )
    # Forward B carries one scale per 128 x 128 block; wgrad B is per-token cast.
    b_u8, sfb_u8, b_deq = _quantize_rows(
        torch.nn.functional.pad(b, (0, 0, 0, pad_n)), 1 if accumulate else 128
    )
    a_u8, sfa_u8, a_deq = a_u8[:M], sfa_u8[:M], a_deq[:M]
    b_u8, b_deq = b_u8[:N], b_deq[:N]
    if accumulate:
        sfb_u8 = sfb_u8[:N]
    sfa = pack_ue8m0_words(sfa_u8, M)
    sfb = pack_ue8m0_words(sfb_u8, N)
    reference = a_deq @ b_deq.T
    return a_u8.contiguous(), b_u8.contiguous(), sfa, sfb, reference


def _output(M, N, dtype):
    """``[M, N]`` output with a 16-byte row pitch: contiguous when ``N`` allows it,
    otherwise the leading columns of a wider buffer (the documented route for
    ``N`` values whose dense pitch TMA cannot encode)."""
    per_pitch = 16 // torch.empty((), dtype=dtype).element_size()
    pitch = (N + per_pitch - 1) // per_pitch * per_pitch
    return torch.empty(M, pitch, device="cuda", dtype=dtype)[:, :N]


SHAPES = [
    (4096, 7168, 4096),  # deployment row
    (1024, 2048, 1024),
    (3000, 4000, 1152),  # ragged M, N not a multiple of 224, 9 K tiles
    (128, 1024, 512),  # single M tile pair
    (300, 7168, 2048),
    (1, 7168, 512),  # decode row: M % 4 != 0 packs a padded scale pitch
    (3, 130, 256),  # M and N both off the 4-row pitch; out needs a padded pitch
]


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_fp8_gemm_1d1d_matches_dequantized_reference(M, N, K, accumulate):
    _skip_unless_supported()
    a, b, sfa, sfb, reference = _case(M, N, K, accumulate, seed=M * 31 + N * 7 + K)
    if accumulate:
        initial = torch.randn(M, N, device="cuda", dtype=torch.float32) * 32.0
        out = _output(M, N, torch.float32)
        out.copy_(initial)
        expected = reference + initial
    else:
        out = _output(M, N, torch.bfloat16)
        expected = reference
    address = out.data_ptr()
    plan = prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out, accumulate=accumulate)
    plan.run()
    torch.cuda.synchronize()
    assert out.data_ptr() == address
    if accumulate:
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-3)
        plan.run()
        torch.testing.assert_close(out, expected + reference, atol=2e-2, rtol=1e-3)
    else:
        torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_gemm_1d1d_replays_on_a_stream_and_in_a_graph(accumulate):
    _skip_unless_supported()
    M, N, K = 1024, 2048, 1024
    a, b, sfa, sfb, reference = _case(M, N, K, accumulate, seed=11)
    initial = (
        torch.randn(M, N, device="cuda", dtype=torch.float32) if accumulate else None
    )
    out = torch.empty(
        M, N, device="cuda", dtype=torch.float32 if accumulate else torch.bfloat16
    )
    plan = prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out, accumulate=accumulate)

    def reset():
        if accumulate:
            out.copy_(initial)
        else:
            out.zero_()

    expected = reference + initial if accumulate else reference
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        reset()
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        reset()
        graph.replay()
    stream.synchronize()
    tolerance = dict(atol=1e-2, rtol=1e-3) if accumulate else dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out.float(), expected, **tolerance)


def test_fp8_gemm_1d1d_rejects_invalid_operands():
    _skip_unless_supported()
    a, b, sfa, sfb, _ = _case(256, 448, 512, False, seed=3)
    out = torch.empty(256, 448, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        prepare_fp8_gemm_1d1d(a, b, sfa, sfb, out.float())
    with pytest.raises(TypeError):
        prepare_fp8_gemm_1d1d(a, b, sfa[:, :128], sfb, out)
    with pytest.raises(ValueError):
        prepare_fp8_gemm_1d1d(a[:, :500], b[:, :500], sfa, sfb, out)
    # K=512 packs into one word per row; row counts that are multiples of four
    # keep a dense pitch, and a 4-byte pitch is rejected.
    assert tuple(sfa.stride()) == (256, 1) and tuple(sfb.stride()) == (448, 1)
    with pytest.raises(ValueError):
        prepare_fp8_gemm_1d1d(
            a, b, sfa.expand(1, 256).as_strided((1, 256), (1, 1)), sfb, out
        )


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 130, 256])
def test_pack_ue8m0_words_pads_the_row_pitch(rows):
    _skip_unless_supported()
    scales = torch.randint(100, 140, (rows, 9), device="cuda", dtype=torch.uint8)
    packed = pack_ue8m0_words(scales, rows)
    assert tuple(packed.shape) == (3, rows)
    assert packed.stride(1) == 1 and packed.stride(0) % 4 == 0
    assert packed.stride(0) == (rows + 3) // 4 * 4 and packed.data_ptr() % 16 == 0
    dense = torch.nn.functional.pad(scales, (0, 3)).contiguous().view(torch.uint32)
    assert torch.equal(
        packed.contiguous().view(torch.int32),
        dense.transpose(0, 1).contiguous().view(torch.int32),
    )


# --- The host plan and the build manifest --------------------------------------


def test_host_plan_of_the_records():
    assert loader.ARCHES == ("sm_100a", "sm_103a")
    assert loader.ARCH_BY_CAPABILITY == {(10, 0): "sm_100a", (10, 3): "sm_103a"}
    assert loader.arch_for_capability((10, 0)) == "sm_100a"
    assert loader.arch_for_capability((10, 3)) == "sm_103a"
    assert loader.arch_for_capability((9, 0)) is None
    assert loader.arch_for_capability((10,)) is None
    # One record per route; the route's build keyword selects it.
    assert set(loader.KERNELS) == {"forward", "wgrad"}
    forward, wgrad = loader.KERNELS["forward"], loader.KERNELS["wgrad"]
    assert forward != wgrad and set(loader.MODULES) == {forward, wgrad}
    assert loader.MODULES[forward]["build"]["accumulate"] is False
    assert loader.MODULES[wgrad]["build"]["accumulate"] is True
    assert loader.MODULES[forward]["workspace_rule"] is None
    assert loader.MODULES[wgrad]["workspace_rule"] == {
        "kind": "in_place_accumulate",
        "output": "C_tma",
    }
    for name, record in loader.MODULES.items():
        for arch in record["arches"]:
            assert loader.library_name(name, arch) == f"{name}_{arch}"
        with pytest.raises(RuntimeError, match="not delivered for sm_90a"):
            loader.library_name(name, "sm_90a")
        assert record["grid_rule"] == _runtime.GRID_RULE
    # The grid rule the runtime evaluates: 128 x 224 tiles, an even M tile count,
    # an even persistent CTA count of at most one per tile, K in 128-wide tiles.
    assert loader.CLUSTER_CTAS == 2
    assert _runtime.launch_geometry(4096, 7168, 4096, 148) == dict(
        grid_m=32, grid_n=32, K_tiles=32, grid_x=148, sf_words=8
    )
    assert _runtime.launch_geometry(3000, 4000, 1152, 148) == dict(
        grid_m=24, grid_n=18, K_tiles=9, grid_x=148, sf_words=3
    )
    assert _runtime.launch_geometry(1, 7168, 512, 148) == dict(
        grid_m=2, grid_n=32, K_tiles=4, grid_x=64, sf_words=1
    )
    assert _runtime.launch_geometry(3, 130, 256, 148) == dict(
        grid_m=2, grid_n=1, K_tiles=2, grid_x=2, sf_words=1
    )
    assert _runtime.launch_geometry(128, 1024, 512, 149) == dict(
        grid_m=2, grid_n=5, K_tiles=4, grid_x=10, sf_words=1
    )
    with pytest.raises(ValueError, match="must be a multiple of 128"):
        _runtime.launch_geometry(256, 256, 200, 148)
    with pytest.raises(ValueError, match="must be positive"):
        _runtime.launch_geometry(0, 256, 256, 148)
    with pytest.raises(ValueError, match="raster group of 8 M tiles"):
        _runtime.launch_geometry(256, 256, 256, 64)
    # The argument plan: the launch values by logical name, then the grid axes.
    record = loader.MODULES[forward]
    values = {key: object() for kind, key in record["arg_plan"] if kind != "grid"}
    args = loader.launch_args(record, values, (148, 1, 1))
    assert args[-3:] == (148, 1, 1)
    assert args[:-3] == tuple(
        values[key] for kind, key in record["arg_plan"] if kind != "grid"
    )
    with pytest.raises(ValueError, match="needs logical argument 'K_tiles'"):
        loader.launch_args(
            record, {k: v for k, v in values.items() if k != "K_tiles"}, (2, 1, 1)
        )
    with pytest.raises(ValueError, match="has no logical name"):
        loader.launch_args({**record, "logical": {}}, values, (2, 1, 1))


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` libraries and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests, a closure digest and
    no nvcc flag of its own (the programs compile with nvcc's defaults); the
    registry itself records the host plan only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == {"sm_100a", "sm_103a"}
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert re.fullmatch(r"cake_deepgemm_fp8_gemm_[0-9a-f]{20}", name)
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert not {
            "sources",
            "source_sha256",
            "compile_flags",
            "closure_sha256",
            "role",
        } & set(record), name
        assert tuple(record["arches"]) == loader.ARCHES, name
        assert set(record["build"]) == {
            "accumulate",
            "backend",
            "num_sms",
            "runtime_shape",
        }
        assert record["build"]["runtime_shape"] is True
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False
        assert tuple(launch["block"]) == (256, 1, 1)
        assert tuple(launch["cluster"]) == (loader.CLUSTER_CTAS, 1, 1)
        assert launch["dynamic_smem_bytes"] > 0
        assert record["grid_rule"] == _runtime.GRID_RULE
        parameters = [key for kind, key in record["arg_plan"] if kind != "grid"]
        assert tuple(parameters) == PARAMETERS
        assert [kind for kind, _ in record["arg_plan"]] == ["buffer"] * 5 + [
            "parameter"
        ] * 6 + ["grid"] * 3
        assert record["logical"] == {key: key for key in parameters}
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
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
            assert entry["nvcc_flags"] == [], library
            assert re.fullmatch(r"\d+\.\d+", entry["min_cuda"]), library
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    # The DeepGEMM derivation notice travels at the package root, where every
    # generated source's license comment points.
    assert (package / "DEEPGEMM_NOTICE.txt").is_file()
    for source in package.glob("csrc/*_kernel.cu"):
        head = source.read_text(encoding="utf-8").split("*/", 1)[0]
        assert "by FlashInfer team" in head and "DEEPGEMM_NOTICE.txt" in head, source
