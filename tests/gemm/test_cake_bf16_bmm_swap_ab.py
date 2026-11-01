"""Tests for the generated swap-AB programs of ``bmm_bf16(backend="cake")``.

``flashinfer.jit.cake_bf16_bmm_swap_ab`` holds tcgen05 programs that serve the
decode rows of ``_CAKE_BF16_BMM_SWAP_AB_ROUTES`` on sm_100a and sm_103a; every
other row keeps the CAKE dispatcher.  Both accumulate in FP32, in different
orders, so the outputs are checked against an FP32 ``torch.bmm`` reference with
the tolerance of ``tests/gemm/test_bmm_bf16.py``.
"""

import pytest
import torch

import flashinfer.gemm.gemm_base as gemm_base
from flashinfer import bmm_bf16
from flashinfer.utils import get_compute_capability

BF16, F16, F32 = torch.bfloat16, torch.float16, torch.float32
CAPABILITIES = {"sm_100a": (10, 0), "sm_103a": (10, 3)}
DISABLE = "FLASHINFER_DISABLE_CAKE_BF16_BMM_SWAP_AB"

# (target, batch, M, N, K, output dtype, program): every routed row.
ROUTED = [
    ("sm_100a", 1, 8, 40, 1024, F32, "W64_T8_S2_SK8_PF_F32"),
    ("sm_100a", 2, 33, 1032, 1024, BF16, "W64_T16_S8_SK2_PF_BF16"),
    ("sm_100a", 1, 16, 64, 1024, F16, "W64_T8_S2_SK8_CP_F16"),
    ("sm_100a", 2, 8, 1024, 1024, F16, "W64_T8_S4_SK4_PF_F16"),
    ("sm_100a", 2, 8, 1024, 1024, F32, "W64_T8_S4_SK4_PF_F32"),
    ("sm_100a", 2, 8, 1024, 1024, BF16, "W64_T8_S4_SK4_PF_BF16"),
    ("sm_100a", 4, 16, 1024, 1024, F16, "W64_T16_S8_SK2_PF_L2_F16"),
    ("sm_100a", 4, 16, 1024, 1024, F32, "W64_T16_S8_SK2_PF_F32"),
    ("sm_100a", 4, 16, 1024, 1024, BF16, "W64_T16_S8_SK2_PF_L2_BF16"),
    ("sm_103a", 1, 8, 40, 1024, F32, "W64_T8_S2_SK8_CP_F32"),
    ("sm_103a", 2, 33, 1032, 1024, BF16, "W64_T32_S8_SK2_PF_BF16"),
    ("sm_103a", 1, 16, 64, 1024, F16, "W64_T8_S2_SK8_CP_F16"),
    ("sm_103a", 2, 8, 1024, 1024, F16, "W64_T8_S4_SK4_PF_F16"),
    ("sm_103a", 2, 8, 1024, 1024, F32, "W64_T8_S4_SK4_PF_F32"),
    ("sm_103a", 2, 8, 1024, 1024, BF16, "W64_T8_S4_SK4_PF_L2_BF16"),
    ("sm_103a", 4, 16, 1024, 1024, F16, "W64_T16_S8_SK2_PF_F16"),
    ("sm_103a", 4, 16, 1024, 1024, F32, "W64_T16_S8_SK2_PF_F32"),
    ("sm_103a", 4, 16, 1024, 1024, BF16, "W64_T16_S8_SK2_PF_BF16"),
]
EXPECTED = {route[:-1]: route[-1] for route in ROUTED}
# The routed rows, each once, and the rows next to them (another output dtype,
# batch, M or K), which keep the dispatcher.
ROWS = list(dict.fromkeys(route[1:-1] for route in ROUTED))
UNROUTED = [
    (1, 8, 40, 1024, BF16),
    (1, 8, 40, 1024, F16),
    (2, 33, 1032, 1024, F16),
    (2, 33, 1032, 1024, F32),
    (1, 16, 64, 1024, BF16),
    (1, 16, 64, 1024, F32),
    (3, 8, 1024, 1024, BF16),
    (2, 16, 1024, 1024, F16),
    (4, 16, 1024, 256, F32),
    (2, 8, 1024, 64, BF16),
]


def _operands(batch, m, n, k, dtype, device="cpu"):
    a = torch.empty(batch, m, k, dtype=BF16, device=device)
    mat2 = torch.empty(batch, n, k, dtype=BF16, device=device).transpose(1, 2)
    out = torch.empty(batch, m, n, dtype=dtype, device=device)
    return a, mat2, out


def _program(monkeypatch, capability, a, mat2, out):
    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda _: capability)
    route = gemm_base._cake_bf16_bmm_swap_ab_route(a, mat2, out)
    if route is None:
        return None
    target, swap_ab = route
    assert capability == CAPABILITIES[target]
    return swap_ab.program


def test_cake_bf16_bmm_swap_ab_routes_are_the_tested_rows():
    """The route table holds exactly ``ROUTED``."""
    table = {
        (target, *row, swap_ab.program)
        for target, routes in gemm_base._CAKE_BF16_BMM_SWAP_AB_ROUTES.items()
        for row, swap_ab in routes.items()
    }
    assert table == set(ROUTED)


@pytest.mark.parametrize(
    "target,batch,m,n,k,dtype,program",
    ROUTED + [(target, *row, None) for target in CAPABILITIES for row in UNROUTED],
)
def test_cake_bf16_bmm_swap_ab_selection(
    monkeypatch, target, batch, m, n, k, dtype, program
):
    """Each routed row selects its program on its target; the rows next to them
    keep the dispatcher."""
    monkeypatch.delenv(DISABLE, raising=False)
    operands = _operands(batch, m, n, k, dtype)
    assert _program(monkeypatch, CAPABILITIES[target], *operands) == program


def test_cake_bf16_bmm_swap_ab_keeps_the_dispatcher(monkeypatch):
    """The escape hatch, another target, and operands the dispatcher's binding
    rejects (dtypes, shapes or strides, a data pointer off 16 bytes, an output
    overlapping an input) keep the dispatcher."""
    monkeypatch.delenv(DISABLE, raising=False)
    a, mat2, out = _operands(2, 8, 1024, 1024, BF16)
    assert _program(monkeypatch, (10, 0), a, mat2, out) == "W64_T8_S4_SK4_PF_BF16"
    monkeypatch.setenv(DISABLE, "1")
    assert _program(monkeypatch, (10, 0), a, mat2, out) is None
    monkeypatch.delenv(DISABLE)
    for capability in ((10, 7), (12, 0)):
        assert _program(monkeypatch, capability, a, mat2, out) is None

    # skip_check=True leaves these to the binding: a routed (batch, M, N, K) with
    # another input dtype, a shorter B or output batch, or K differing from B's.
    assert _program(monkeypatch, (10, 0), a.half(), mat2, out) is None
    assert _program(monkeypatch, (10, 0), a, mat2[:1], out) is None
    assert _program(monkeypatch, (10, 0), a, mat2, out[:1]) is None
    assert _program(monkeypatch, (10, 0), a, mat2[:, :512], out) is None

    shifted = torch.empty(a.numel() + 1, dtype=BF16)[1:].view_as(a)
    assert _program(monkeypatch, (10, 0), shifted, mat2, out) is None
    padded = torch.empty(2, 1024, 1024 + 64, dtype=BF16)[..., :1024].transpose(1, 2)
    assert _program(monkeypatch, (10, 0), a, padded, out) is None
    assert _program(monkeypatch, (10, 0), a, mat2, a) is None
    over_b = mat2.transpose(1, 2).reshape(-1)[: out.numel()].view_as(out)
    assert _program(monkeypatch, (10, 0), a, mat2, over_b) is None


def test_cake_bf16_bmm_swap_ab_programs_match_routes():
    """Every route names a generated program that ships for its target, takes the
    host's arguments and launches the route's cluster; every program is routed."""
    from flashinfer.jit import cake_bf16_bmm_swap_ab as programs

    suffix = {BF16: "_BF16", F16: "_F16", F32: "_F32"}
    routed = set()
    for target, routes in gemm_base._CAKE_BF16_BMM_SWAP_AB_ROUTES.items():
        for (_, _, _, _, dtype), swap_ab in routes.items():
            record = getattr(programs, swap_ab.program)
            routed.add(record)
            assert target in record.targets, swap_ab.program
            assert record.args == gemm_base._CAKE_BF16_BMM_SWAP_AB_ARGS
            assert record.block == (256, 1, 1)
            assert record.cluster == (swap_ab.split_k, 1, 1), swap_ab.program
            assert swap_ab.program.startswith(f"W{swap_ab.tile_w}_T{swap_ab.tile_t}_S")
            assert f"_SK{swap_ab.split_k}_" in swap_ab.program
            assert swap_ab.program.endswith(suffix[dtype])
    assert routed == set(programs.PROGRAMS)


def _target():
    if not torch.cuda.is_available():
        pytest.skip("the swap-AB programs require a CUDA device")
    capability = get_compute_capability(torch.device("cuda"))
    targets = {cap: target for target, cap in CAPABILITIES.items()}
    cc = capability[0] * 10 + capability[1]
    if capability not in targets or not bmm_bf16.is_backend_supported("cake", cc):
        pytest.skip("the swap-AB programs require SM100 or SM103")
    return targets[capability]


def _spy(monkeypatch):
    """Records the program of each call, ``None`` for the dispatcher."""
    calls = []
    run = gemm_base._run_cake_bf16_bmm_swap_ab
    get_module = gemm_base.get_blackwell_bf16_bmm_module

    def run_swap_ab(target, swap_ab, *args):
        calls.append(swap_ab.program)
        run(target, swap_ab, *args)

    def get_dispatcher(*args, **kwargs):
        calls.append(None)
        return get_module(*args, **kwargs)

    monkeypatch.setattr(gemm_base, "_run_cake_bf16_bmm_swap_ab", run_swap_ab)
    monkeypatch.setattr(gemm_base, "get_blackwell_bf16_bmm_module", get_dispatcher)
    return calls


def _case(batch, m, n, k, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((batch, m, k), device="cuda", dtype=BF16, generator=generator)
    mat2 = torch.randn(
        (batch, n, k), device="cuda", dtype=BF16, generator=generator
    ).transpose(1, 2)
    expected = torch.bmm(a.float(), mat2.float()).to(dtype)
    return a, mat2, expected


@pytest.mark.parametrize("batch,m,n,k,dtype", ROWS + UNROUTED)
@pytest.mark.parametrize("preallocated", [True, False])
def test_cake_bf16_bmm_swap_ab_rows(monkeypatch, batch, m, n, k, dtype, preallocated):
    """Every row runs its program on this target (the rows next to them the
    dispatcher) and matches FP32 torch.bmm, twice with the same bits."""
    target = _target()
    monkeypatch.delenv(DISABLE, raising=False)
    calls = _spy(monkeypatch)
    a, mat2, expected = _case(batch, m, n, k, dtype)

    results = []
    for _ in range(2):
        out = None
        if preallocated:
            out = torch.full((batch, m, n), float("nan"), device="cuda", dtype=dtype)
        result = bmm_bf16(a, mat2, out=out, out_dtype=dtype, backend="cake")
        assert out is None or result is out
        results.append(result)
    torch.cuda.synchronize()

    program = EXPECTED.get((target, batch, m, n, k, dtype))
    assert calls == [program, program]
    assert results[0].dtype == dtype
    assert results[0].data_ptr() != results[1].data_ptr()
    torch.testing.assert_close(results[0], expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("misaligned", ["A", "B", "out"])
def test_cake_bf16_bmm_swap_ab_rejects_misaligned_data_pointer(misaligned):
    """A routed row keeps the dispatcher's alignment errors."""
    _target()
    batch, m, n, k = 2, 8, 1024, 1024
    a, mat2, _ = _case(batch, m, n, k, BF16)
    out = torch.empty((batch, m, n), device="cuda", dtype=BF16)
    if misaligned == "A":
        a = torch.empty(a.numel() + 1, device="cuda", dtype=BF16)[1:].view_as(a)
    elif misaligned == "B":
        storage = torch.empty(batch * n * k + 1, device="cuda", dtype=BF16)[1:]
        mat2 = storage.view(batch, n, k).transpose(1, 2)
    else:
        out = torch.empty(out.numel() + 1, device="cuda", dtype=BF16)[1:].view_as(out)

    expected_error = f"{misaligned} data pointer must be 16-byte aligned"
    with pytest.raises(ValueError, match=expected_error):
        bmm_bf16(a, mat2, out=out, backend="cake")


def test_cake_bf16_bmm_swap_ab_rejects_output_input_overlap():
    """A routed row keeps the dispatcher's overlap errors."""
    _target()
    a, mat2, _ = _case(2, 8, 1024, 1024, BF16)
    with pytest.raises(ValueError, match="out must not overlap A"):
        bmm_bf16(a, mat2, out=a, backend="cake")

    over_b = mat2.transpose(1, 2).reshape(-1)[: a.numel()].view_as(a)
    with pytest.raises(ValueError, match="out must not overlap B"):
        bmm_bf16(a, mat2, out=over_b, backend="cake")


if __name__ == "__main__":
    pytest.main([__file__])
