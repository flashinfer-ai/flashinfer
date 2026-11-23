from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear import _tuning, api
from b12x.gemm.trellis_linear._tuning import TrellisConfig, TrellisQuery
from b12x.preparation import PreparedCall, PreparationSession
from b12x.preparation.types import require_prepared
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table_cpu
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.w4a16.kernel import (
    _trellis256_dense_launch_geometry,
    _trellis256_dense_split_k_slices,
    run_trellis256_dense,
)

_MCG = np.uint64(0xCBAC1FED)
_MUL1 = np.uint64(0x83DCD12D)
_MASK = np.uint32(0x8FFF8FFF)
_ORC = np.uint32(0x3B603B60)


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 12 and minor in (0, 1)


def _decode_3inst_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    value = np.uint32((value & _MASK) ^ _ORC)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)


def _decode_mul1_e4m3_fp16(window: np.ndarray) -> np.ndarray:
    product = ((window.astype(np.uint64) * _MUL1) & np.uint64(0xFFFFFFFF)).astype(
        np.uint32
    )
    byte_sum = (
        (product & np.uint32(0xFF)).astype(np.uint32)
        + ((product >> np.uint32(8)) & np.uint32(0xFF))
        + ((product >> np.uint32(16)) & np.uint32(0xFF))
        + ((product >> np.uint32(24)) & np.uint32(0xFF))
    )
    accumulator = (byte_sum + np.uint32(0x6400)).astype(np.uint16).view(np.float16)
    inv = np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0]
    bias = np.array([0xC931], dtype=np.uint16).view(np.float16)[0]
    reconstructed = (
        accumulator.astype(np.float64) * np.float64(inv) + np.float64(bias)
    ).astype(np.float16)
    return (
        torch.from_numpy(np.asarray(reconstructed))
        .to(torch.float8_e4m3fn)
        .to(torch.float16)
        .numpy()
    )


@lru_cache(maxsize=None)
def _lut_e4m3_table(bits: int) -> np.ndarray:
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported lut_e4m3 test rate K{bits}")
    rate_index = bits - 2
    labels = lut_e4m3_direct_table_cpu()[rate_index << 16 : (rate_index + 1) << 16]
    return labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


def _decode_lut_e4m3_fp16(window: np.ndarray, bits: int) -> np.ndarray:
    indices = np.asarray(window, dtype=np.uint32) & np.uint32(0xFFFF)
    return _lut_e4m3_table(bits)[indices]


def _decode_lane(
    tile_words: np.ndarray,
    lane: int,
    bits: int,
    *,
    codebook: str = "mcg",
) -> np.ndarray:
    width = 8 * bits
    values = []
    for weight in range(8):
        end_bit = (lane * 8 + weight + 257) * bits
        start_bit = end_bit - 16
        first_word = start_bit // 32
        last_word = (end_bit - 1) // 32
        shift = (last_word + 1) * 32 - end_bit
        first = tile_words[..., first_word % width].astype(np.uint64)
        last = tile_words[..., last_word % width].astype(np.uint64)
        merged = (first << np.uint64(32)) | last
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint32)
        if codebook == "mcg":
            values.append(_decode_3inst_fp16(window))
        elif codebook == "mul1-e4m3":
            values.append(_decode_mul1_e4m3_fp16(window))
        elif codebook == "lut_e4m3":
            values.append(_decode_lut_e4m3_fp16(window, bits))
        else:
            raise ValueError(f"unsupported test codebook {codebook!r}")
    return np.stack(values, axis=-1).astype(np.float16)


def _reconstruct_native(
    trellis: torch.Tensor, *, codebook: str = "mcg"
) -> torch.Tensor:
    native = trellis.detach().cpu().numpy()
    bits = int(native.shape[-1]) // 16
    k_tiles, n_tiles, _ = native.shape
    packed = native.view(np.uint16).reshape(k_tiles, n_tiles, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (
        packed[..., 1].astype(np.uint32) << np.uint32(16)
    )
    output = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.float16)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            lanes = np.stack(
                [
                    _decode_lane(words[k_tile, n_tile], lane, bits, codebook=codebook)
                    for lane in range(32)
                ]
            )
            block = np.zeros((16, 16), dtype=np.float16)
            for lane in range(32):
                row0 = (lane % 4) * 2
                rows = (row0, row0 + 1, row0 + 8, row0 + 9)
                col0 = lane // 8
                col1 = col0 + 4
                parity = (lane >> 2) & 1
                for weight in range(8):
                    block[
                        rows[weight % 4],
                        2 * (col0 if weight < 4 else col1) + parity,
                    ] = lanes[lane, weight]
            output[
                k_tile * 16 : (k_tile + 1) * 16,
                n_tile * 16 : (n_tile + 1) * 16,
            ] = block
    return torch.from_numpy(output)


def _reference_mxfp8_rows(source: torch.Tensor) -> torch.Tensor:
    """Reference the UE8M0/K32 E4M3 activation quantizer in logical K order."""
    m, k = source.shape
    blocks = source.float().reshape(m, k // 32, 32)
    max_abs = blocks.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, device=source.device), exponent)
    scale = torch.where(max_abs > 0, scale, torch.ones_like(scale))
    return (blocks / scale).to(torch.float8_e4m3fn).float().mul(scale).reshape(m, k)


@contextmanager
def _prepared_trellis(
    x,
    weight,
    *,
    c_tmp,
    hadamard_128=None,
    allow_split_k=False,
    override=None,
    **run_kwargs,
):
    query = trellis_linear.query_from_weight(
        weight,
        max_rows=x.shape[0],
        input_dtype=x.dtype,
        output_mode="provided"
        if run_kwargs.get("output") is not None
        else "functional",
        c_tmp_mode="provided",
        hadamard_128=hadamard_128,
        allow_split_k=allow_split_k,
        **{
            f"{name}_provided": run_kwargs.get(name) is not None
            for name in (
                "gemm_output",
                "input_f16",
                "rotated_f16",
                "rotated_compute",
                "gemm_output_f16",
                "output_f16",
            )
        },
    )
    declaration = trellis_linear.plan(
        query,
        weight=weight,
        c_tmp=c_tmp,
        hadamard_128=hadamard_128,
        override=override,
    )
    request = declaration.request(
        name="trellis",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(x, **run_kwargs)),
    )
    with PreparationSession(
        device=x.device, autotune=False, compile_workers=2
    ) as session:
        result = session.prepare((request,))
        yield declaration, session, result


@pytest.mark.parametrize(
    ("pair_kind", "rate_axis", "match"),
    [
        ("P25", "n", "pair_kind must be P24 or P33"),
        ("P24", "x", "rate_axis must be 'k' or 'n'"),
    ],
)
def test_prepare_pair_weight_rejects_malformed_descriptor_before_cuda(
    pair_kind: str,
    rate_axis: str,
    match: str,
) -> None:
    payload = torch.empty(0, dtype=torch.int16)
    scale = torch.empty(0, dtype=torch.float16)

    with pytest.raises(ValueError, match=match):
        trellis_linear.prepare_pair_weight(
            payload,
            scale,
            scale,
            pair_kind=pair_kind,
            rate_axis=rate_axis,
        )


def test_is_supported_uses_standard_sm12x_gate(monkeypatch) -> None:
    seen = {}

    def fake_gate(device, *, requires):
        seen["device"] = device
        seen["requires"] = requires
        return True

    monkeypatch.setattr(api, "default_is_supported", fake_gate)
    assert trellis_linear.is_supported("cuda:3")
    assert seen == {"device": "cuda:3", "requires": trellis_linear.META.requires}


@pytest.mark.parametrize(
    ("size_m", "size_k", "size_n", "expected"),
    [
        # Narrow profile: avoid a short spill by increasing both M and N work.
        (512, 16384, 6144, (48, (64, 128))),
        (1024, 16384, 6144, (48, (64, 256))),
        # Wide model projection: N already supplies enough work. Splitting N is
        # useful for the second wave, while M48 only adds scheduler overhead.
        (192, 2048, 16384, (64, (64, 128))),
        (384, 2048, 16384, (48, (64, 256))),
        # A deeper K makes M48 scheduler overhead more expensive.
        (384, 6144, 16384, (64, (64, 256))),
        # Full waves retain the throughput-oriented default geometry.
        (2048, 4096, 6144, (64, (64, 256))),
    ],
)
def test_dense_launch_geometry_avoids_short_spill_waves(
    size_m: int,
    size_k: int,
    size_n: int,
    expected: tuple[int, tuple[int, int]],
) -> None:
    assert (
        _trellis256_dense_launch_geometry(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            sms=188,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("size_m", "size_k", "size_n", "block_rows", "tile_n", "expected"),
    [
        # Decode/verify projections from the issue: 4, 24 and 16 whole-tile
        # CTAs on 188 SMs. At most 8 slices that split the K-tiles evenly,
        # at least 512 of K each, within one SM wave (24 tiles: 7 -> 4).
        (4, 6144, 1024, 64, 256, 8),
        (4, 4096, 6144, 64, 256, 4),
        (4, 2048, 4096, 64, 256, 4),
        (32, 6144, 1024, 64, 256, 8),
        # A full 64-row tile needs 1024 of K per slice.
        (64, 6144, 1024, 64, 256, 6),
        (64, 2048, 4096, 64, 256, 2),
        # One SM wave caps the slices: 48 tiles -> 3, which 64 K-tiles round to 2.
        (128, 4096, 6144, 64, 256, 2),
        # Whole tiles already fill the device (prefill keeps its schedule).
        (2048, 4096, 6144, 64, 256, None),
        (512, 4096, 16384, 64, 256, None),
        # Too little K for two 512-wide slices.
        (4, 512, 1024, 64, 256, None),
        # The M8 decode schedule computes one whole tile per CTA.
        (4, 6144, 1024, 8, 256, None),
    ],
)
def test_dense_split_k_slices_fill_underused_decode_launches(
    size_m: int,
    size_k: int,
    size_n: int,
    block_rows: int,
    tile_n: int,
    expected: int | None,
) -> None:
    assert (
        _trellis256_dense_split_k_slices(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            block_rows=block_rows,
            tile_k=64,
            tile_n=tile_n,
            sms=188,
        )
        == expected
    )


def _query(*, max_rows: int, allow_split_k: bool) -> TrellisQuery:
    return TrellisQuery(
        max_rows=max_rows,
        in_features=6144,
        out_features=1024,
        input_dtype="bfloat16",
        compute_dtype="bfloat16",
        output_dtype="bfloat16",
        codebook="mcg",
        bits=4,
        allow_split_k=allow_split_k,
    )


def test_query_defaults_to_the_whole_tile_schedule() -> None:
    query = TrellisQuery(
        max_rows=4,
        in_features=6144,
        out_features=1024,
        input_dtype="bfloat16",
        compute_dtype="bfloat16",
        output_dtype="bfloat16",
        codebook="mcg",
        bits=4,
    )
    assert query.allow_split_k is False
    config = _tuning.default_config(query, SimpleNamespace(sm_count=188))
    assert config.split_k_slices is None


def test_default_config_splits_k_only_when_allowed() -> None:
    device = SimpleNamespace(sm_count=188)
    whole = _tuning.default_config(_query(max_rows=4, allow_split_k=False), device)
    split = _tuning.default_config(_query(max_rows=4, allow_split_k=True), device)
    assert whole.split_k_slices is None
    assert split.split_k_slices == 8
    assert (split.block_rows, split.tile_k, split.tile_n) == (
        whole.block_rows,
        whole.tile_k,
        whole.tile_n,
    )
    prefill = _tuning.default_config(_query(max_rows=8192, allow_split_k=True), device)
    assert prefill.split_k_slices is None


def test_tuning_space_searches_split_k_only_when_allowed() -> None:
    device = SimpleNamespace(sm_count=188)
    whole = _tuning._tuning_parameters(_query(max_rows=4, allow_split_k=False), device)
    split = _tuning._tuning_parameters(_query(max_rows=4, allow_split_k=True), device)
    assert whole["split_k_slices"] == (None,)
    assert None in split["split_k_slices"]
    assert len(split["split_k_slices"]) > 1


@pytest.mark.parametrize(
    ("allow_split_k", "block_rows", "split_k_slices", "match"),
    [
        (False, 64, 4, "allow_split_k=True"),
        (True, 8, 4, "block_rows >= 16"),
        (True, 64, 1, "integer >= 2"),
        (True, 64, 97, "exceeds the K-tile count"),
    ],
)
def test_validate_config_rejects_unsupported_split_k(
    allow_split_k: bool,
    block_rows: int,
    split_k_slices: int,
    match: str,
) -> None:
    config = TrellisConfig(
        backend="cutedsl",
        block_rows=block_rows,
        tile_k=64,
        tile_n=256,
        split_k_slices=split_k_slices,
    )
    with pytest.raises(ValueError, match=match):
        _tuning.validate_config(
            _query(max_rows=4, allow_split_k=allow_split_k), config, None
        )


def test_validate_config_accepts_allowed_split_k() -> None:
    config = TrellisConfig(
        backend="cutedsl", block_rows=64, tile_k=64, tile_n=256, split_k_slices=12
    )
    _tuning.validate_config(_query(max_rows=4, allow_split_k=True), config, None)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_trellis_dense_cuda_graph_replay_is_stable() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    m = 3
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones((features,), dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
        params_dtype=torch.float16,
    )
    x = torch.randn((m, features), dtype=torch.float16, device=device)
    output = torch.empty_like(x)
    gemm_output = torch.empty_like(x)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    kwargs = {
        "output": output,
        "gemm_output": gemm_output,
        "rotated_f16": rotated_f16,
    }
    with _prepared_trellis(x, weight, c_tmp=c_tmp, **kwargs) as (plan, session, result):
        expected = trellis_linear.run(x, plan=plan, **kwargs).clone()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = trellis_linear.run(x, plan=plan, **kwargs)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured, expected)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [2, 3])
def test_dense_bf16_reuses_all_scratch_during_cuda_graph_capture(bits: int) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    m = 2
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones((features,), dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
        params_dtype=torch.bfloat16,
    )
    x = torch.randn((m, features), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(x)
    gemm_output = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    input_f16 = torch.empty_like(x, dtype=torch.float16)
    rotated_f16 = torch.empty_like(input_f16)
    rotated_compute = torch.empty_like(x)
    gemm_output_f16 = torch.empty_like(input_f16)
    output_f16 = torch.empty_like(input_f16)

    def hadamard_128(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    kwargs = {
        "output": output,
        "gemm_output": gemm_output,
        "input_f16": input_f16,
        "rotated_f16": rotated_f16,
        "rotated_compute": rotated_compute,
        "gemm_output_f16": gemm_output_f16,
        "output_f16": output_f16,
    }
    with _prepared_trellis(
        x,
        weight,
        c_tmp=c_tmp,
        hadamard_128=hadamard_128,
        **kwargs,
    ) as (plan, session, result):
        expected = trellis_linear.run(x, plan=plan, **kwargs).clone()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = trellis_linear.run(x, plan=plan, **kwargs)
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured, expected)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_dense_lut_e4m3_matches_reference(bits: int) -> None:
    """Close the runtime lut_e4m3 labels through the W4A16 GEMM fragment path."""

    torch.manual_seed(0x535147 + bits)
    device = torch.device("cuda", torch.cuda.current_device())
    m = 2
    features = 128
    trellis = torch.randint(
        -32768,
        32767,
        (features // 16, features // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    scale = torch.ones(features, dtype=torch.float16, device=device)
    weight = trellis_linear.prepare_weight(
        trellis,
        scale,
        scale.clone(),
        codebook="lut_e4m3",
        params_dtype=torch.float16,
    )
    assert weight.trellis_codebook == "lut_e4m3"
    reference_weight = _reconstruct_native(trellis, codebook="lut_e4m3").to(device)
    x = (torch.randn((m, features), device=device) * 1.0e-3).to(torch.float16)

    def identity_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    output = torch.empty_like(x)
    gemm_output = torch.empty_like(x)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    kwargs = {
        "output": output,
        "gemm_output": gemm_output,
        "rotated_f16": rotated_f16,
    }
    with _prepared_trellis(
        x,
        weight,
        c_tmp=c_tmp,
        hadamard_128=identity_hadamard,
        **kwargs,
    ) as (plan, _, _):
        actual = trellis_linear.run(x, plan=plan, **kwargs).clone()
        torch.cuda.synchronize(device)
        expected = (x.float() @ reference_weight.float()).to(torch.float16)
        relative_error = (actual - expected).float().norm() / expected.float().norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert float(relative_error) <= 2.0e-2
        assert float(cosine) >= 0.999


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(("pair_kind", "bits"), [("P24", (2, 4)), ("P33", (3, 3))])
@pytest.mark.parametrize("rate_axis", ["k", "n"])
@pytest.mark.parametrize(
    "orthogonal_tiles",
    [16, 224],
    ids=["square-proof", "wide-axis"],
)
@pytest.mark.parametrize("codebook", ["mcg", "lut_e4m3"])
def test_dense_pair_matches_independent_reference_and_captures(
    pair_kind: str,
    bits: tuple[int, int],
    rate_axis: str,
    orthogonal_tiles: int,
    codebook: str,
) -> None:
    torch.manual_seed(20260801)
    device = torch.device("cuda", torch.cuda.current_device())
    low_bits, high_bits = bits
    if rate_axis == "n":
        low = torch.randint(
            -32768,
            32767,
            (orthogonal_tiles, 8, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (orthogonal_tiles, 8, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        reference_weight = torch.cat(
            (
                _reconstruct_native(low, codebook=codebook),
                _reconstruct_native(high, codebook=codebook),
            ),
            dim=1,
        ).to(device)
    else:
        low = torch.randint(
            -32768,
            32767,
            (8, orthogonal_tiles, 16 * low_bits),
            dtype=torch.int16,
            device=device,
        )
        high = torch.randint(
            -32768,
            32767,
            (8, orthogonal_tiles, 16 * high_bits),
            dtype=torch.int16,
            device=device,
        )
        reference_weight = torch.cat(
            (
                _reconstruct_native(low, codebook=codebook),
                _reconstruct_native(high, codebook=codebook),
            ),
            dim=0,
        ).to(device)
    payload = torch.cat((low.reshape(-1), high.reshape(-1))).contiguous()
    suh = torch.ones(reference_weight.shape[0], dtype=torch.float16, device=device)
    svh = torch.ones(reference_weight.shape[1], dtype=torch.float16, device=device)
    codebook_kwargs = (
        {"mcg": torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device)}
        if codebook == "mcg"
        else {"codebook": codebook}
    )
    weight = trellis_linear.prepare_pair_weight(
        payload,
        suh,
        svh,
        pair_kind=pair_kind,
        rate_axis=rate_axis,
        params_dtype=torch.float16,
        **codebook_kwargs,
    )
    assert weight.trellis_codebook == codebook
    assert weight.trellis.numel() * weight.trellis.element_size() == (
        payload.numel() * payload.element_size()
    )
    if orthogonal_tiles == 224:
        assert payload.numel() * payload.element_size() == 344064
    if rate_axis == "k":
        assert weight.trellis.data_ptr() == payload.data_ptr()
    else:
        assert weight.trellis.data_ptr() != payload.data_ptr()

    x = (torch.randn((2, reference_weight.shape[0]), device=device) * 1.0e-3).to(
        torch.float16
    )

    def identity_hadamard(
        source: torch.Tensor,
        destination: torch.Tensor,
        _left_scale,
        _right_scale,
        _scale: float,
    ) -> None:
        destination.copy_(source)

    output = torch.empty(
        (x.shape[0], reference_weight.shape[1]), dtype=x.dtype, device=device
    )
    gemm_output = torch.empty_like(output)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    kwargs = {
        "output": output,
        "gemm_output": gemm_output,
        "rotated_f16": rotated_f16,
    }
    with _prepared_trellis(
        x,
        weight,
        c_tmp=c_tmp,
        hadamard_128=identity_hadamard,
        **kwargs,
    ) as (plan, session, result):
        actual = trellis_linear.run(x, plan=plan, **kwargs).clone()
        torch.cuda.synchronize(device)
        expected = (x.float() @ reference_weight.float()).to(torch.float16)
        relative_error = (actual - expected).float().norm() / expected.float().norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert float(relative_error) <= 2.0e-2
        assert float(cosine) >= 0.999
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = trellis_linear.run(x, plan=plan, **kwargs)
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured, actual)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_prepare_pair_weight_rejects_wrong_tp12_payload_length() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    payload = torch.empty(172032 - 1, dtype=torch.int16, device=device)
    suh = torch.ones(3584, dtype=torch.float16, device=device)
    svh = torch.ones(256, dtype=torch.float16, device=device)

    with pytest.raises(ValueError, match="payload length mismatch"):
        trellis_linear.prepare_pair_weight(
            payload,
            suh,
            svh,
            pair_kind="P24",
            rate_axis="n",
        )


def _identity_hadamard(
    source: torch.Tensor,
    destination: torch.Tensor,
    _left_scale,
    _right_scale,
    _scale: float,
) -> None:
    destination.copy_(source)


@lru_cache(maxsize=None)
def _split_k_reference_case(size_k: int, size_n: int, bits: int):
    """One random native weight and its independently decoded fp16 matrix."""
    generator = torch.Generator().manual_seed(0x9381 + size_k + size_n + bits)
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 16 * bits),
        dtype=torch.int16,
        generator=generator,
    )
    return trellis, _reconstruct_native(trellis)


def _mcg_weight(trellis: torch.Tensor, params_dtype: torch.dtype):
    device = trellis.device
    size_k = trellis.shape[0] * 16
    size_n = trellis.shape[1] * 16
    return trellis_linear.prepare_weight(
        trellis,
        torch.ones(size_k, dtype=torch.float16, device=device),
        torch.ones(size_n, dtype=torch.float16, device=device),
        mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
        params_dtype=params_dtype,
    )


def _split_k_override(device, *, size_m, size_k, size_n, split_k_slices):
    block_rows, (tile_k, tile_n) = _trellis256_dense_launch_geometry(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        sms=torch.cuda.get_device_properties(device).multi_processor_count,
    )
    return TrellisConfig(
        backend="cutedsl",
        block_rows=block_rows,
        tile_k=tile_k,
        tile_n=tile_n,
        split_k_slices=split_k_slices,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize(
    ("m", "split_k_slices"),
    [
        (1, 2),
        # Three slices over eight K-tiles: uneven slices, and with two N tiles
        # one CTA's stripe straddles the end of tile 0 and the start of tile 1.
        (4, 3),
        # One K-tile per slice: the longest lock-ordered finalize chain.
        (4, 8),
        (16, 5),
        # Two route blocks.
        (65, 3),
    ],
)
def test_dense_split_k_matches_whole_tile_and_reference(
    m: int, split_k_slices: int
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    size_k, size_n = 512, 512
    trellis_cpu, reference_cpu = _split_k_reference_case(size_k, size_n, 4)
    weight = _mcg_weight(trellis_cpu.to(device), torch.float16)
    reference_weight = reference_cpu.to(device)
    x = (torch.randn((m, size_k), device=device) * 1.0e-2).to(torch.float16)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)

    with _prepared_trellis(x, weight, c_tmp=c_tmp, hadamard_128=_identity_hadamard) as (
        plan,
        _,
        _,
    ):
        assert require_prepared(plan, "gemm.trellis_linear").split_k_slices is None
        whole = trellis_linear.run(x, plan=plan).clone()

    override = _split_k_override(
        device,
        size_m=m,
        size_k=size_k,
        size_n=size_n,
        split_k_slices=split_k_slices,
    )
    with _prepared_trellis(
        x,
        weight,
        c_tmp=c_tmp,
        hadamard_128=_identity_hadamard,
        allow_split_k=True,
        override=override,
    ) as (plan, _, _):
        state = require_prepared(plan, "gemm.trellis_linear")
        assert state.split_k_slices == split_k_slices
        assert not state.launch.kernel.schedule_whole_tiles
        split = trellis_linear.run(x, plan=plan).clone()
        repeated = trellis_linear.run(x, plan=plan).clone()
    torch.cuda.synchronize(device)

    # The lock-ordered finalize fixes the partial-sum order: bit-stable.
    assert torch.equal(split, repeated)
    assert _relative_error(split, whole) <= 2.0e-3
    expected = (x.float() @ reference_weight.float()).to(torch.float16)
    assert _relative_error(split, expected) <= 2.0e-2
    cosine = torch.nn.functional.cosine_similarity(
        split.float().flatten(), expected.float().flatten(), dim=0
    )
    assert float(cosine) >= 0.999


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_dense_split_k_cuda_graph_replay_is_stable() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    m, size_k, size_n = 4, 1024, 512
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    weight = _mcg_weight(trellis, torch.float16)
    x = torch.randn((m, size_k), dtype=torch.float16, device=device)
    output = torch.empty((m, size_n), dtype=torch.float16, device=device)
    gemm_output = torch.empty_like(output)
    rotated_f16 = torch.empty_like(x)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    kwargs = {
        "output": output,
        "gemm_output": gemm_output,
        "rotated_f16": rotated_f16,
    }
    override = _split_k_override(
        device, size_m=m, size_k=size_k, size_n=size_n, split_k_slices=4
    )
    with _prepared_trellis(
        x,
        weight,
        c_tmp=c_tmp,
        hadamard_128=_identity_hadamard,
        allow_split_k=True,
        override=override,
        **kwargs,
    ) as (plan, session, _):
        assert require_prepared(plan, "gemm.trellis_linear").split_k_slices == 4
        expected = trellis_linear.run(x, plan=plan, **kwargs).clone()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = trellis_linear.run(x, plan=plan, **kwargs)
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(captured, expected)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("m", [4, 32])
def test_dense_allowed_split_k_decode_plan_splits_k(m: int) -> None:
    """The issue's 6144x1024 K6 projection: the allowed plan splits K by default."""
    device = torch.device("cuda", torch.cuda.current_device())
    size_k, size_n = 6144, 1024
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    weight = _mcg_weight(trellis, torch.bfloat16)
    x = (torch.randn((m, size_k), device=device) * 1.0e-2).to(torch.bfloat16)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)

    with _prepared_trellis(x, weight, c_tmp=c_tmp, hadamard_128=_identity_hadamard) as (
        plan,
        _,
        _,
    ):
        assert require_prepared(plan, "gemm.trellis_linear").split_k_slices is None
        whole = trellis_linear.run(x, plan=plan).clone()

    with _prepared_trellis(
        x,
        weight,
        c_tmp=c_tmp,
        hadamard_128=_identity_hadamard,
        allow_split_k=True,
    ) as (plan, _, _):
        state = require_prepared(plan, "gemm.trellis_linear")
        kernel = state.launch.kernel
        expected_slices = _trellis256_dense_split_k_slices(
            size_m=m,
            size_k=size_k,
            size_n=size_n,
            block_rows=kernel.moe_block_size,
            tile_k=kernel.tile_k,
            tile_n=kernel.tile_n,
            sms=torch.cuda.get_device_properties(device).multi_processor_count,
        )
        assert expected_slices is not None
        assert state.split_k_slices == expected_slices
        assert not kernel.schedule_whole_tiles
        split = trellis_linear.run(x, plan=plan).clone()
    torch.cuda.synchronize(device)

    assert torch.isfinite(split).all()
    assert _relative_error(split, whole) <= 5.0e-3


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_dense_whole_tile_rows_do_not_depend_on_planned_m() -> None:
    """allow_split_k=False keeps a row's bits independent of the planned M.

    The whole-tile default configs fold K in one order for every M (the
    intra-CTA fold depends on tile_k, which the default keeps at 64); split-K
    must not leak into these plans.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    size_k, size_n = 1024, 512
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    weight = _mcg_weight(trellis, torch.bfloat16)
    x_all = (torch.randn((512, size_k), device=device) * 1.0e-2).to(torch.bfloat16)
    c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
    outputs = {}
    for m in (1, 4, 65, 512):
        x = x_all[:m]
        with _prepared_trellis(
            x, weight, c_tmp=c_tmp, hadamard_128=_identity_hadamard
        ) as (plan, _, _):
            assert require_prepared(plan, "gemm.trellis_linear").split_k_slices is None
            outputs[m] = trellis_linear.run(x, plan=plan).clone()
    torch.cuda.synchronize(device)
    for m, out in outputs.items():
        assert torch.equal(out, outputs[512][:m])


def _split_k_pair(device):
    """One weight, two split-K plans of the same M: the same lock slots."""
    size_k, size_n = 4096, 1024
    trellis = torch.randint(
        -32768,
        32767,
        (size_k // 16, size_n // 16, 96),
        dtype=torch.int16,
        device=device,
    )
    weight = _mcg_weight(trellis, torch.bfloat16)
    xs = [
        (torch.randn((4, size_k), device=device) * 1.0e-2).to(torch.bfloat16)
        for _ in range(2)
    ]
    c_tmps = [
        torch.empty((1 << 20,), dtype=torch.float32, device=device) for _ in range(2)
    ]
    return weight, xs, c_tmps


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_split_k_plans_own_their_lock_counters() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    weight, xs, c_tmps = _split_k_pair(device)
    with (
        _prepared_trellis(
            xs[0],
            weight,
            c_tmp=c_tmps[0],
            hadamard_128=_identity_hadamard,
            allow_split_k=True,
        ) as (first, _, _),
        _prepared_trellis(
            xs[1],
            weight,
            c_tmp=c_tmps[1],
            hadamard_128=_identity_hadamard,
            allow_split_k=True,
        ) as (second, _, _),
        _prepared_trellis(
            xs[0], weight, c_tmp=c_tmps[0], hadamard_128=_identity_hadamard
        ) as (whole, _, _),
    ):
        states = [require_prepared(p, "gemm.trellis_linear") for p in (first, second)]
        assert all(state.split_k_slices is not None for state in states)
        pointers = {state.locks.data_ptr() for state in states}
        assert len(pointers) == 2
        assert weight.workspace.data_ptr() not in pointers
        assert require_prepared(whole, "gemm.trellis_linear").locks is None
        for plan, x in zip((first, second), xs):
            trellis_linear.run(x, plan=plan)
        torch.cuda.synchronize(device)
        for state in states:
            assert int(state.locks.abs().sum()) == 0

        state = states[0]
        common = {
            "launch": state.launch,
            "execution_lut": state.execution_lut,
            "grid_cap": state.grid_cap,
            "hadamard_128": state.hadamard_128,
            "split_k_slices": state.split_k_slices,
        }
        with pytest.raises(ValueError, match="its own lock storage"):
            run_trellis256_dense(xs[0], weight, **common)
        with pytest.raises(ValueError, match="shared workspace"):
            run_trellis256_dense(xs[0], weight, locks=weight.workspace, **common)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
def test_concurrent_split_k_plans_on_one_weight_match_sequential() -> None:
    """Two split-K plans of one weight on two streams: no shared lock counters."""
    device = torch.device("cuda", torch.cuda.current_device())
    weight, xs, c_tmps = _split_k_pair(device)
    with (
        _prepared_trellis(
            xs[0],
            weight,
            c_tmp=c_tmps[0],
            hadamard_128=_identity_hadamard,
            allow_split_k=True,
        ) as (first, _, _),
        _prepared_trellis(
            xs[1],
            weight,
            c_tmp=c_tmps[1],
            hadamard_128=_identity_hadamard,
            allow_split_k=True,
        ) as (second, _, _),
    ):
        plans = (first, second)
        expected = [trellis_linear.run(x, plan=p).clone() for p, x in zip(plans, xs)]
        torch.cuda.synchronize(device)
        streams = [torch.cuda.Stream(device=device) for _ in plans]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream(device))
        results = [[], []]
        for _ in range(32):
            for index, (stream, plan, x) in enumerate(zip(streams, plans, xs)):
                with torch.cuda.stream(stream):
                    results[index].append(trellis_linear.run(x, plan=plan))
        torch.cuda.synchronize(device)
        for index in range(2):
            for out in results[index]:
                assert torch.equal(out, expected[index])
