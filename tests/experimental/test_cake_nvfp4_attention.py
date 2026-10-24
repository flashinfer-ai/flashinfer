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

Tests of the experimental NVFP4 attention backend package (SM103): the packing
oracle of its operand layouts, the validated shape set against
``scaled_dot_product_attention``, the host plan of its registry record and the
``build.json`` contract.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.prefill.prepare_nvfp4_attention`` public entry is covered by
``test_cake_nvfp4_attention_api.py``.
"""

import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("nvfp4_attention")
backend = cake_package("nvfp4_attention", "cake_backend")
loader = cake_package("nvfp4_attention", "cake_jit")
cake_build = cake_package("nvfp4_attention", "cake_build")

# The kernel's parameters in ``run`` order: the host vocabulary of the record.
PARAMETERS = (
    "Q",
    "K",
    "Vt",
    "SFQ",
    "SFK",
    "SFVtLo",
    "SFVtHi",
    "O",
    "seqlen_q",
    "seqlen_kv",
    "q_stride",
    "kv_stride",
    "softmax_scale_log2",
    "total_bh",
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


def _skip_unless_sm103():
    if not _supported():
        pytest.skip("SM103 required")


def _devices():
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's library when it has no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(loader.arch_for_capability(_capability())):
            spec.build_and_load()
    yield


# --- Reference packing (the eager implementation the backend replaced) -------
# Kept verbatim as the bit-exact oracle of the packed operand layouts.


def _reference_quantize_nvfp4(x):
    x_f32 = x.float()
    groups = x_f32.shape[-1] // 16
    blocks = x_f32.reshape(*x_f32.shape[:-1], groups, 16)
    raw_scale = (blocks.abs().amax(dim=-1) / 6.0).clamp(
        min=2.0**-9, max=torch.finfo(torch.float8_e4m3fn).max
    )
    scale_fp8 = raw_scale.to(torch.float8_e4m3fn)
    scale = scale_fp8.float()
    normalized = blocks / scale.unsqueeze(-1)
    ax = normalized.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(ax, boundaries).to(torch.uint8)
    code |= ((normalized < 0) & (code != 0)).to(torch.uint8) << 3
    code = code.reshape(*x_f32.shape)
    pairs = code.reshape(*code.shape[:-1], code.shape[-1] // 2, 2)
    packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)
    return packed.contiguous(), scale_fp8.view(torch.uint8).contiguous()


def _reference_prepack_sf_tiles(scale_u8):
    tiles = scale_u8.shape[0]
    sf = scale_u8.reshape(tiles, 4, 32, 1, 8)
    sf = sf.reshape(tiles, 4, 4, 8, 1, 2, 4)
    sf = sf.permute(0, 4, 2, 5, 3, 1, 6).contiguous()
    return sf.reshape(tiles * 32, 32)


def _reference_prepack_sf_tiles_split_ksets(scale_u8):
    tiles = scale_u8.shape[0]

    def pack_half(values):
        sf = values.contiguous().reshape(tiles, 4, 32, 1, 4)
        sf = sf.reshape(tiles, 4, 4, 8, 1, 1, 4)
        sf = sf.permute(0, 4, 2, 5, 3, 1, 6).contiguous()
        return sf.reshape(tiles, 512)

    lo = pack_half(scale_u8[..., :4]).reshape(tiles * 16, 32)
    hi = pack_half(scale_u8[..., 4:]).reshape(tiles * 16, 32)
    return lo, hi


def _reference_pack(q, k, v):
    batch, heads, seqlen, dim = q.shape
    bh = batch * heads
    sources = [x.reshape(bh, seqlen, dim).contiguous() for x in (q, k, v)]
    packed_q, sq = _reference_quantize_nvfp4(sources[0])
    packed_k, sk = _reference_quantize_nvfp4(sources[1])
    blocks = seqlen // 128
    sq = _reference_prepack_sf_tiles(sq.reshape(bh, blocks, 128, 8).reshape(-1, 128, 8))
    sk = _reference_prepack_sf_tiles(sk.reshape(bh, blocks, 128, 8).reshape(-1, 128, 8))
    packed_v, sv = _reference_quantize_nvfp4(sources[2].transpose(-1, -2).contiguous())
    scales = sv.reshape(bh, 128, blocks, 8).permute(0, 2, 1, 3).contiguous()
    lo, hi = _reference_prepack_sf_tiles_split_ksets(scales.reshape(-1, 128, 8))
    return dict(
        Q=packed_q.reshape(bh * seqlen, 64),
        K=packed_k.reshape(bh * seqlen, 64),
        Vt=packed_v.reshape(bh * 128, seqlen // 2),
        SFQ=sq,
        SFK=sk,
        SFVtLo=lo,
        SFVtHi=hi,
    )


def _inputs_with_edge_values(batch, heads, seqlen, device, seed=7):
    """Random BF16 inputs seeded with the values that exercise every code path:
    zeros, exact E2M1 thresholds times a scale, saturated magnitudes and
    infinities."""
    generator = torch.Generator().manual_seed(seed)
    q, k, v = (
        torch.randn((batch, heads, seqlen, 128), generator=generator).to(
            dtype=torch.bfloat16, device=device
        )
        for _ in range(3)
    )
    largest = torch.finfo(torch.bfloat16).max
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            5.0,
            -5.0,
            6.0,
            -6.0,
            2688.0,
            -2688.0,
            4096.0,
            largest,
            -largest,
            float("inf"),
            -float("inf"),
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    for x in (q, k, v):
        x[0, 0, : specials.numel(), 0] = specials
        x[0, 0, 0, : specials.numel()] = specials
        x[0, 0, :16, 1] = 0.0
    return q, k, v


def _assert_same_pack(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name].dtype == torch.uint8, name
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].is_contiguous(), name
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def _nhd_view(x):
    """``[B,H,S,128]`` view of a tensor stored as ``[B,S,H,128]``."""
    storage = x.transpose(1, 2).contiguous()
    view = storage.transpose(1, 2)
    assert not view.is_contiguous()
    return view


# --- The validated shape set --------------------------------------------------


@pytest.mark.parametrize(
    "batch,heads,seqlen", [(4, 8, 4096), (1, 8, 32768), (8, 32, 8192)]
)
def test_nvfp4_attention(batch, heads, seqlen):
    _skip_unless_sm103()
    torch.manual_seed(42)
    q = torch.randn((batch, heads, seqlen, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    attention = backend.prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention() is out
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1.0, rtol=0.1)
    snapshot = out.clone()
    out.zero_()
    assert attention() is out
    torch.testing.assert_close(out, snapshot, atol=0, rtol=0)


def test_nvfp4_attention_accepts_strided_qkv():
    """``[B,S,H,128]`` storage viewed as ``[B,H,S,128]`` is quantized in place and
    produces the output of the contiguous copies bit for bit."""
    _skip_unless_sm103()
    torch.manual_seed(3)
    q = torch.randn((2, 4, 2048, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    expected = torch.empty_like(q)
    backend.prepare_nvfp4_attention(q, k, v, expected, backend="cake")()
    out = torch.empty_like(q)
    attention = backend.prepare_nvfp4_attention(
        _nhd_view(q), _nhd_view(k), _nhd_view(v), out, backend="cake"
    )
    assert attention() is out
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_nvfp4_attention_requires_contiguous_output():
    _skip_unless_sm103()
    q = torch.randn((1, 2, 512, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        backend.prepare_nvfp4_attention(
            q, q, q, _nhd_view(torch.empty_like(q)), backend="cake"
        )


def test_nvfp4_runner_binds_the_records_argument_plan():
    """The runner's arguments are the record's ``arg_plan`` over ``main_kwargs``
    and its grid is the registry's rule on the device's SM count."""
    _skip_unless_sm103()
    q = torch.randn((1, 2, 512, 128), dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(q)
    attention = backend.prepare_nvfp4_attention(q, q, q, out, backend="cake")
    arch = loader.arch_for_capability(_capability())
    record = loader.MODULES[loader.record_for(arch)]
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = loader.launch_grid(record, total_bh=2, seqlen_q=512, sm_count=sm_count)
    assert attention.main_kwargs["grid"] == grid == (2 * min(sm_count // 2, 2), 1, 1)
    assert len(attention.arguments) == len(record["arg_plan"])
    assert attention.arguments[-3:] == grid
    for argument, (kind, key) in zip(
        attention.arguments[:-3], record["arg_plan"][:-3], strict=True
    ):
        assert kind in ("buffer", "parameter")
        assert argument is attention.main_kwargs[record["logical"][key]]
    assert attention.out is out and attention() is out


def test_nvfp4_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        backend.prepare_nvfp4_attention(None, None, None, None, backend="unknown")


# --- The packing oracle --------------------------------------------------------


@pytest.mark.parametrize("device", _devices())
def test_nvfp4_packing_matches_reference_bitwise(device):
    q, k, v = _inputs_with_edge_values(1, 2, 1024, device)
    _assert_same_pack(
        backend.pack_nvfp4_attention_inputs(q, k, v), _reference_pack(q, k, v)
    )


@pytest.mark.parametrize("device", _devices())
def test_nvfp4_packing_is_stride_independent(device):
    q, k, v = _inputs_with_edge_values(2, 3, 512, device, seed=11)
    expected = backend.pack_nvfp4_attention_inputs(q, k, v)
    strided = backend.pack_nvfp4_attention_inputs(
        _nhd_view(q), _nhd_view(k), _nhd_view(v)
    )
    _assert_same_pack(strided, expected)


@pytest.mark.parametrize("device", _devices())
def test_nvfp4_quantization_saturates_finite_scales(device):
    largest = torch.finfo(torch.bfloat16).max
    values = torch.tensor(
        [0.0, 1.0, 2688.0, 4096.0, -4096.0, largest, -largest],
        dtype=torch.bfloat16,
        device=device,
    )
    x = values[:, None].expand(-1, 16).contiguous()
    packed, scale_bytes = backend.quantize_nvfp4_rows(x)
    scales = scale_bytes.view(torch.float8_e4m3fn).float()
    expected_scales = torch.tensor(
        [2.0**-9, 0.171875, 448.0, 448.0, 448.0, 448.0, 448.0], device=device
    )[:, None]
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    expected_packed = torch.tensor(
        [0x00, 0x77, 0x77, 0x77, 0xFF, 0x77, 0xFF],
        dtype=torch.uint8,
        device=device,
    )[:, None].expand(-1, 8)
    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)


# --- The host plan and the build manifest --------------------------------------


def test_host_plan_of_the_record():
    (name,) = loader.MODULES
    record = loader.MODULES[name]
    assert loader.ARCHES == ("sm_103a",) and loader.ARCH_BY_CAPABILITY == {
        (10, 3): "sm_103a"
    }
    assert loader.arch_for_capability((10, 3)) == "sm_103a"
    assert loader.arch_for_capability((10, 0)) is None
    assert loader.arch_for_capability((10,)) is None
    assert loader.record_for("sm_103a") == name
    with pytest.raises(RuntimeError, match="delivers 0 programs for sm_100a"):
        loader.record_for("sm_100a")
    assert loader.library_name(name, "sm_103a") == f"{name}_sm_103a"
    with pytest.raises(RuntimeError, match="not delivered for sm_100a"):
        loader.library_name(name, "sm_100a")
    # The grid: a persistent cluster of two CTAs per 512-row tile, at most one per SM pair.
    assert loader.TILE_ROWS == 512 and loader.CLUSTER_CTAS == 2
    assert loader.launch_grid(record, total_bh=32, seqlen_q=4096, sm_count=148) == (
        148,
        1,
        1,
    )
    assert loader.launch_grid(record, total_bh=3, seqlen_q=1536, sm_count=148) == (
        18,
        1,
        1,
    )
    assert loader.launch_grid(record, total_bh=1, seqlen_q=512, sm_count=148) == (
        2,
        1,
        1,
    )
    assert loader.launch_grid(record, total_bh=1, seqlen_q=513, sm_count=148) == (
        4,
        1,
        1,
    )
    assert loader.launch_grid(record, total_bh=1, seqlen_q=512, sm_count=3) == (2, 1, 1)
    with pytest.raises(ValueError, match="at least 2 SMs"):
        loader.launch_grid(record, total_bh=1, seqlen_q=512, sm_count=1)
    with pytest.raises(ValueError, match="positive heads and rows"):
        loader.launch_grid(record, total_bh=0, seqlen_q=512, sm_count=148)
    with pytest.raises(ValueError, match="unknown Cake NVFP4 attention grid rule"):
        loader.launch_grid(
            {"grid_rule": {"kind": "linear"}}, total_bh=1, seqlen_q=512, sm_count=148
        )
    # The argument plan: ``main_kwargs`` by logical name, then the grid axes.
    values = {key: object() for kind, key in record["arg_plan"] if kind != "grid"}
    args = loader.launch_args(record, values, (24, 1, 1))
    assert args[-3:] == (24, 1, 1)
    assert args[:-3] == tuple(
        values[key] for kind, key in record["arg_plan"] if kind != "grid"
    )
    with pytest.raises(ValueError, match="needs logical argument 'total_bh'"):
        loader.launch_args(
            record, {k: v for k, v in values.items() if k != "total_bh"}, (2, 1, 1)
        )


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests, a closure digest and
    the program's ``--use_fast_math``; the registry itself records the host plan
    only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == {"sm_103a"}
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert re.fullmatch(r"cake_nvfp4_attention_[0-9a-f]{20}", name)
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert record["arches"] and set(record["arches"]) <= set(loader.ARCHES), name
        assert set(record["build"]) == {"groupwise", "has_kv_tail", "is_causal"}
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is True
        assert tuple(launch["block"]) == (512, 1, 1)
        assert tuple(launch["cluster"]) == (loader.CLUSTER_CTAS, 1, 1)
        assert launch["dynamic_smem_bytes"] > 0
        assert record["grid_rule"] == {
            "kind": "persistent_tiles",
            "tile_rows": loader.TILE_ROWS,
            "ctas_per_tile": loader.CLUSTER_CTAS,
        }
        assert record["workspace_rule"] is None
        parameters = [key for kind, key in record["arg_plan"] if kind != "grid"]
        assert tuple(parameters) == PARAMETERS
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
            assert entry["nvcc_flags"] == ["--use_fast_math"], library
            assert re.fullmatch(r"\d+\.\d+", entry["min_cuda"]), library
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
