from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import numpy as np
import pytest
import torch

from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear import api
from b12x.preparation import PreparedCall, PreparationSession
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table_cpu
from b12x.gemm._shared.wo_mxfp8 import empty_mxfp8_rows_for_dense_gemm
from b12x.moe._shared.kernels.activations import (
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.w4a16.kernel import _trellis256_dense_launch_geometry
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
    labels = lut_e4m3_direct_table_cpu()[
        rate_index << 16 : (rate_index + 1) << 16
    ]
    return labels.view(torch.float8_e4m3fn).to(torch.float16).numpy()


def _decode_lut_e4m3_fp16(
    window: np.ndarray, bits: int
) -> np.ndarray:
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
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(
            np.uint32
        )
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
                    _decode_lane(
                        words[k_tile, n_tile], lane, bits, codebook=codebook
                    )
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
    return (
        (blocks / scale)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scale)
        .reshape(m, k)
    )

@contextmanager
def _prepared_trellis(x, weight, *, c_tmp, hadamard_128=None, **run_kwargs):
    query = trellis_linear.query_from_weight(
        weight,
        max_rows=x.shape[0],
        input_dtype=x.dtype,
        output_mode="provided" if run_kwargs.get("output") is not None else "functional",
        c_tmp_mode="provided",
        hadamard_128=hadamard_128,
        **{
            f"{name}_provided": run_kwargs.get(name) is not None
            for name in (
                "gemm_output", "input_f16", "rotated_f16", "rotated_compute",
                "gemm_output_f16", "output_f16",
            )
        },
    )
    declaration = trellis_linear.plan(
        query, weight=weight, c_tmp=c_tmp, hadamard_128=hadamard_128,
    )
    request = declaration.request(
        name="trellis",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(x, **run_kwargs)),
    )
    with PreparationSession(device=x.device, autotune=False, compile_workers=2) as session:
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
        x, weight, c_tmp=c_tmp, hadamard_128=hadamard_128, **kwargs,
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
    reference_weight = _reconstruct_native(
        trellis, codebook="lut_e4m3"
    ).to(device)
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
        x, weight, c_tmp=c_tmp, hadamard_128=identity_hadamard, **kwargs,
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
        {
            "mcg": torch.tensor(
                0xCBAC1FED, dtype=torch.uint32, device=device
            )
        }
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
        x, weight, c_tmp=c_tmp, hadamard_128=identity_hadamard, **kwargs,
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
