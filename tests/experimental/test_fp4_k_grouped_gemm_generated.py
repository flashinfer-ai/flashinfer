"""Grouped packed FP4 GEMM: any group layout, exact empty groups, current-stream graph replay."""

import pytest
import torch
from flashinfer.experimental.deepgemm_kgroup_gemm import kgroup_gemm as _runtime
from flashinfer.fp4_k_grouped_gemm import prepare_fp4_k_grouped_gemm

# E2M1 magnitudes by code; bit 3 is the sign.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

_CASES = [
    # catalogued model geometries (dynamic tile counts on the N256 / swapped schedules)
    dict(m=4608, N=5120, group_ks=[8192, 0, 4096], output_dtype="bf16"),
    dict(
        m=4608, N=5120, group_ks=[12288, 0, 6144], output_dtype="fp32", accumulate=True
    ),
    dict(m=4608, N=5120, group_ks=[16384, 0, 8192], output_dtype="fp32"),
    dict(m=5120, N=2304, group_ks=[8192, 0, 4096], output_dtype="bf16"),
    dict(
        m=5120, N=2304, group_ks=[12288, 0, 6144], output_dtype="fp32", accumulate=True
    ),
    dict(m=5120, N=2304, group_ks=[16384, 0, 8192], output_dtype="fp32"),
    # group counts, K values and alignments outside the former catalog
    dict(m=4608, N=5120, group_ks=[4096, 4096, 4096, 4096], output_dtype="bf16"),
    dict(m=5120, N=2304, group_ks=[9216, 7168], output_dtype="fp32"),
    dict(
        m=4608,
        N=5120,
        group_ks=[8192, 0, 4096],
        output_dtype="fp32",
        use_psum_layout=False,
    ),
    dict(m=300, N=384, group_ks=[1000], output_dtype="bf16"),
    dict(
        m=1024,
        N=512,
        group_ks=[513, 0, 2047, 256, 31],
        k_alignment=512,
        use_psum_layout=False,
        output_dtype="fp32",
    ),
    dict(
        m=512,
        N=256,
        group_ks=[257, 0, 511, 768, 1, 1024, 300],
        k_alignment=768,
        output_dtype="fp32",
        accumulate=True,
    ),
    dict(m=256, N=128, group_ks=[8192, 4096], output_dtype="bf16"),
    dict(m=256, N=128, group_ks=[8192, 8192], output_dtype="fp32"),
    dict(m=256, N=128, group_ks=[8192, 4096], output_dtype="fp32", accumulate=True),
    dict(m=1024, N=384, group_ks=[8192, 4096], output_dtype="bf16"),
    dict(m=1024, N=384, group_ks=[8192, 8192], output_dtype="fp32"),
    dict(m=1024, N=384, group_ks=[8192, 4096], output_dtype="fp32", accumulate=True),
    dict(m=256, N=128, group_ks=[257, 0, 511], output_dtype="bf16"),
    dict(
        m=256,
        N=128,
        group_ks=[257, 0, 511],
        k_alignment=768,
        use_psum_layout=False,
        output_dtype="fp32",
        accumulate=True,
    ),
    dict(m=256, N=128, group_ks=[257, 511, 0], output_dtype="fp32"),
    dict(m=256, N=128, group_ks=[2049, 0, 511], output_dtype="bf16"),
    dict(
        m=256,
        N=128,
        group_ks=[2049, 0, 511],
        use_psum_layout=False,
        output_dtype="fp32",
    ),
    dict(m=256, N=128, group_ks=[2049, 0, 511], output_dtype="fp32", accumulate=True),
    # the 2 x 8 x 3 tile geometry of those rows on layouts the former catalog never had
    dict(m=200, N=128, group_ks=[1024, 3000, 0], output_dtype="bf16"),
    dict(
        m=256,
        N=128,
        group_ks=[0, 2560, 300],
        k_alignment=512,
        use_psum_layout=False,
        output_dtype="fp32",
        accumulate=True,
    ),
]


# (arch, SM count) of the SM100a / SM103a parts the host may run on: the catalog
# devices (B200, GB300) and GB200, whose 152 SMs reach the BM240 schedule the
# export never measured on SM100a.
_DEVICE_PROFILES = {
    "B200": ("sm_100a", 148),
    "GB200": ("sm_100a", 152),
    "GB300": ("sm_103a", 152),
}
_MODES = [("bf16", False), ("fp32", False), ("fp32", True)]


@pytest.mark.parametrize("device", sorted(_DEVICE_PROFILES))
def test_every_device_profile_resolves_a_measured_program(device):
    """CPU-only: every case in every output mode resolves to a program the export measured on the
    device's architecture, taking the preferred schedule whenever that architecture carries it."""
    arch, sm_count = _DEVICE_PROFILES[device]
    table = _runtime.ROUTES.get(arch, _runtime.ROUTES)
    assert {route.split(":")[0] for route in table} >= {"general_s1", "general_s2"}
    seen = set()
    for case in _CASES:
        for output_dtype, accumulate in _MODES:
            args = (
                case["m"],
                case["N"],
                case["group_ks"],
                sm_count,
                output_dtype,
                accumulate,
                case.get("k_alignment", 256),
            )
            candidates = _runtime.route_candidates(*args)
            route, program = _runtime.select_route(arch, *args)
            assert candidates[0] == _runtime.route_key(*args)
            assert route == next(
                candidate for candidate in candidates if candidate in table
            )
            assert table[route] == program
            assert arch in _runtime.MODULES[program]["arches"]
            assert candidates[-1].split(":")[0].startswith("general_s")
            seen.add(route.split(":")[0])
    assert seen >= {"general_s1", "general_s2", "n256", "small", "small_2x8x3"}


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_facts(torch.cuda.current_device())
    except RuntimeError as error:
        pytest.skip(str(error))


def _layout(case):
    m, n, ks = case["m"], case["N"], case["group_ks"]
    alignment = case.get("k_alignment", 256)
    padded = [(k + alignment - 1) // alignment * alignment for k in ks]
    return (m + 255) // 256 * 256, n, ks, padded, sum(padded)


def _decode(codes):
    """Packed E2M1 bytes -> float values, even K in the low nibble."""
    low, high = codes & 0xF, codes >> 4
    table = _E2M1.to(codes.device)
    values = torch.stack(
        (
            table[(low & 7).long()] * (1 - 2 * (low >> 3).float()),
            table[(high & 7).long()] * (1 - 2 * (high >> 3).float()),
        ),
        dim=-1,
    )
    return values.flatten(-2)


def fixture(case, *, random_codes):
    physical_m, n, ks, padded, total = _layout(case)
    generator = torch.Generator(device="cuda").manual_seed(432)
    a = torch.zeros((physical_m, total // 2), dtype=torch.uint8, device="cuda")
    b = torch.zeros((n, total // 2), dtype=torch.uint8, device="cuda")
    cursor = 0
    for k, pk in zip(ks, padded, strict=True):
        for value in (a, b):
            if random_codes:
                block = torch.randint(
                    0,
                    256,
                    (value.shape[0], k // 2),
                    dtype=torch.int32,
                    generator=generator,
                    device="cuda",
                )
                value[:, cursor // 2 : (cursor + k) // 2] = block.to(torch.uint8)
                if k % 2:
                    tail = torch.randint(
                        0,
                        16,
                        (value.shape[0],),
                        dtype=torch.int32,
                        generator=generator,
                        device="cuda",
                    )
                    value[:, (cursor + k) // 2] = tail.to(torch.uint8)
            else:
                value[:, cursor // 2 : (cursor + k) // 2] = 0x22
                if k % 2:
                    value[:, (cursor + k) // 2] = 0x02
        cursor += pk
    # UE8M0 scale 2^0 for every 32-wide K block.
    sfa = torch.full(
        (total // 128, physical_m), 0x7F7F7F7F, dtype=torch.int32, device="cuda"
    )
    sfb = torch.full((total // 128, n), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    dtype = (
        torch.float32 if case.get("output_dtype", "bf16") == "fp32" else torch.bfloat16
    )
    accumulate = case.get("accumulate", False)
    initial = torch.full(
        (len(ks), physical_m, n), 0.25 if accumulate else 0, dtype=dtype, device="cuda"
    )
    out = initial.clone()
    options = {k: v for k, v in case.items() if k != "N"}
    plan = prepare_fp4_k_grouped_gemm(a, b, sfa, sfb, **options, out=out)
    assert plan.storage is out
    return a, b, initial, plan


def exact_reference(a, b, initial, case):
    """Products are multiples of 0.25 and sums stay below 2^24, so FP32 is exact
    and the BF16 output is one rounding of that exact value."""
    physical_m, n, ks, padded, _ = _layout(case)
    av, bv = _decode(a).float(), _decode(b).float()
    expected = initial.clone()
    cursor = 0
    with _tf32_off():
        for group, (k, pk) in enumerate(zip(ks, padded, strict=True)):
            if k:
                product = av[:, cursor : cursor + k] @ bv[:, cursor : cursor + k].T
                expected[group] = (product + initial[group].float()).to(expected.dtype)
            cursor += pk
    return expected


class _tf32_off:
    def __enter__(self):
        self.previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.previous


@pytest.mark.parametrize("case", _CASES)
def test_random_codes_match_exact_reference(case):
    _skip_unless_supported()
    a, b, initial, plan = fixture(case, random_codes=True)
    plan.run()
    torch.cuda.synchronize()
    expected = exact_reference(a, b, initial, case)
    torch.testing.assert_close(plan.storage, expected, atol=0, rtol=0)


@pytest.mark.parametrize("case", _CASES)
def test_grouped_values_stream_changed_input_replay(case):
    _skip_unless_supported()
    a, _, initial, plan = fixture(case, random_codes=False)
    positive = a.clone()
    negative = (
        positive | 0x88
    )  # Sign of both FP4 nibbles; padded zero becomes signed zero.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    accumulate = case.get("accumulate", False)
    for sign, bytes_ in ((1, positive), (-1, negative), (1, positive)):
        with torch.cuda.stream(stream):
            a.copy_(bytes_)
            plan.storage.copy_(initial)
        for replay_count in (1, 2, 3):
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            multiplier = replay_count if accumulate else 1
            for group, k in enumerate(case["group_ks"]):
                expected = initial[group] + sign * multiplier * k
                torch.testing.assert_close(
                    plan.storage[group], expected, atol=0, rtol=0
                )


@pytest.mark.parametrize(
    "dtype,accumulate", [("bf16", False), ("fp32", False), ("fp32", True)]
)
@pytest.mark.parametrize("psum", [False, True])
def test_all_empty_zero_or_preserve_graph(dtype, accumulate, psum):
    _skip_unless_supported()
    case = dict(
        m=256,
        N=128,
        group_ks=[0, 0, 0],
        output_dtype=dtype,
        accumulate=accumulate,
        use_psum_layout=psum,
    )
    _, _, initial, plan = fixture(case, random_codes=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    stream.synchronize()
    for value in (0.25, -2.0, 8.0):
        with torch.cuda.stream(stream):
            plan.storage.fill_(value)
            graph.replay()
        stream.synchronize()
        expected = torch.full_like(plan.storage, value if accumulate else 0)
        torch.testing.assert_close(plan.storage, expected, atol=0, rtol=0)
