# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""cuDNN Frost SwiGLU with launch-time alpha/beta/limit (``swiglu_param``).

DeepSeek-V4/V4.1-Flash and GLM-5.3-Flash clamp SwiGLU at 10; gpt-oss and
MiniMax-M3 use OpenAI's ``alpha=1.702, beta=1, limit=7`` form. The SM120 sources
read these values as scalar kernel inputs, so one kernel serves every model.
SM120 BF16 and NVFP4 also admit gpt-oss's H = I = 2880 (BF16 in 32-element
blocks, NVFP4 in 64s); the canonical MXFP8 weight scales still need 128s.
"""

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFrostBf16Config,
    CutlassNvfp4Config,
    CudnnFrostMxfp8Config,
    CudnnFrostMxfp8Mxfp4Config,
    CudnnFrostNvfp4Config,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.api import ReLU2
from flashinfer.fused_moe.backends.cudnn_frost.mxfp8 import support as mxfp8_support
from flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4 import (
    support as mixed_support,
)
from flashinfer.fused_moe.backends.cudnn_frost.nvfp4 import support as nvfp4_support
from flashinfer.fused_moe.backends.cudnn_frost.activations import (
    SWIGLU_PARAM_SCALARS,
    activation_name,
    activation_scalars,
    launch_scalars,
)
from flashinfer.fused_moe.backends.cudnn_frost.bf16 import moe as bf16_moe
from flashinfer.fused_moe.backends.cudnn_frost.cache import (
    require_graph_resource_retention,
)
from flashinfer.fused_moe.backends.cudnn_frost.dimension_catalog import (
    SWIGLU_PARAM_TAIL,
)
from flashinfer.fused_moe.backends.cudnn_frost.manifest import read_manifest
from flashinfer.fused_moe.backends.cudnn_frost.runtime import artifact_root
from tests.moe.test_cudnn_frost_kernels import (
    _bf16_moe_reference,
    _mxfp8_moe_reference,
    _nvfp4_moe_reference,
)

CLAMPED = SwiGLU(alpha=1.0, beta=0.0, limit=10.0)  # DeepSeek-V4, GLM-5.3
OAI = SwiGLU(alpha=1.702, beta=1.0, limit=7.0)  # gpt-oss, MiniMax-M3
FORMATS = {
    "bf16": (CudnnFrostBf16Config, QuantFormat.BF16, QuantFormat.BF16),
    "mxfp8": (CudnnFrostMxfp8Config, QuantFormat.MXFP8, QuantFormat.MXFP8),
    "nvfp4": (CudnnFrostNvfp4Config, QuantFormat.NVFP4, QuantFormat.NVFP4),
    "mxfp8_mxfp4": (CudnnFrostMxfp8Mxfp4Config, QuantFormat.MXFP4, QuantFormat.MXFP8),
}


def _records(dtype):
    path = artifact_root(dtype) / "cudnn_frost_selected_kernels.json"
    return read_manifest(path)["kernels"]


def test_activation_names_and_launch_scalars():
    assert activation_name(SwiGLU()) == "swiglu"
    assert activation_name(CLAMPED) == "swiglu_param"
    assert activation_name(OAI) == "swiglu_param"
    assert activation_scalars(SwiGLU()) == {}
    assert activation_scalars(OAI) == dict(
        swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0, swiglu_lower=-7.0
    )
    assert launch_scalars(CLAMPED) == (1.0, 0.0, 10.0, -10.0)
    assert launch_scalars(SwiGLU())[:2] == (1.0, 0.0)
    # Only SwiGLU has scalar-parameterized sources.
    for other in (SiTU(gate_scale=5.0), SwiGLUStep(limit=5.0)):
        with pytest.raises(NotImplementedError):
            activation_name(other)


@pytest.mark.parametrize("dtype", FORMATS)
def test_swiglu_param_catalog(dtype):
    records = _records(dtype)
    param = [r for r in records if r["contract"]["activation"] == "swiglu_param"]
    assert param and {r["arch"] for r in param} == {"sm_120a"}
    # Every SM120 SwiGLU tile/output/orientation also has its swiglu_param kernel.

    def axes(r):
        return (r["tactic"]["tile"], r["contract"]["output_dtype"])

    swiglu = {
        axes(r)
        for r in records
        if r["arch"] == "sm_120a" and r["contract"]["activation"] == "swiglu"
    }
    assert {axes(r) for r in param} == swiglu
    for r in param:
        tail = r["launch"]["tail"]
        assert set(SWIGLU_PARAM_SCALARS) <= set(tail)
        scale = tail.index("scale")
        assert tuple(tail[scale - len(SWIGLU_PARAM_TAIL) : scale]) == SWIGLU_PARAM_TAIL
    # SM120 kernels take whole 32-element N/K blocks (e.g. 2880).
    for r in records:
        if r["arch"] == "sm_120a":
            assert r["contract"]["n"]["multiple_of"] == 32
            assert r["contract"]["k"]["multiple_of"] == 32


def test_swiglu_param_rejected_where_no_kernels_are_packaged(monkeypatch):
    """SM107 packages no swiglu_param sources: the runner declines, not crashes."""
    config = MoEConfig(
        routing=RoutingConfig(num_experts=8, top_k=2),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=256),
        backend=BackendOptions((CudnnFrostBf16Config(),)),
        activation=OAI,
    )
    runner = bf16_moe.CudnnFrostBf16MoeRunner.__new__(bf16_moe.CudnnFrostBf16MoeRunner)
    runner.config, runner.device = config, torch.device("cuda", 0)
    monkeypatch.setattr(bf16_moe, "get_compute_capability", lambda device: (10, 7))
    with pytest.raises(NotImplementedError, match="no swiglu_param kernels"):
        runner._check_support()


@pytest.mark.parametrize(
    "dtype,eligible_on_sm120",
    [("nvfp4", True), ("mxfp8", False), ("mxfp8_mxfp4", False)],
)
def test_block_scale_gpt_oss_geometry_admission(dtype, eligible_on_sm120):
    """H = I = 2880: NVFP4 takes 64s on SM120; canonical MX weight scales are
    int32-packed per 128 K, so MXFP8 still needs 128s. SM107 needs 128s."""
    config = MoEConfig(
        routing=RoutingConfig(num_experts=32, top_k=4),
        quant=QuantConfig(weight=FORMATS[dtype][1], activation=FORMATS[dtype][2]),
        experts=ExpertConfig(intermediate_size=2880),
        execution=ExecutionConfig(tune_max_num_tokens=64),
        activation=OAI,
    )
    if dtype == "nvfp4":
        x, module = torch.empty(8, 1440, dtype=torch.uint8), nvfp4_support
    else:
        x = torch.empty(8, 2880, dtype=torch.float8_e4m3fn)
        module = mxfp8_support if dtype == "mxfp8" else mixed_support
    ids = torch.zeros(8, 4, dtype=torch.int32)
    act = MoEActivationPack(x, None, ids, torch.full((8, 4), 0.25))
    assert module.is_eligible(config, act, 120) is eligible_on_sm120
    assert not module.is_eligible(config, act, 107)


def test_nvfp4_per_branch_fc1_scales():
    """I % 128 == 64: the gate rows start mid-atom in the canonical [up; gate]
    scale matrix, so the Frost view swizzles each branch on its own."""
    if not torch.cuda.is_available():
        pytest.skip("fp4_quantize requires CUDA")
    from flashinfer.fused_moe.prepare import (
        _nvfp4_swizzle_scales,
        _nvfp4_unswizzle_scales,
    )

    e, h, i = 3, 128, 192
    gen = torch.Generator(device="cuda").manual_seed(5)
    w1 = torch.randn(e, 2 * i, h, device="cuda", generator=gen).bfloat16()
    w2 = torch.randn(e, h, i, device="cuda", generator=gen).bfloat16()
    common = dict(num_local_experts=e, hidden_size=h, intermediate_size=i)
    canonical = CutlassNvfp4Config.prepare_weights(w1, w2, **common)
    frost = CudnnFrostNvfp4Config.prepare_weights(w1, w2, **common)
    for name, value in canonical.items():
        if name != "fc1_weight_block_scale":
            assert torch.equal(frost[name], value), name
    sf = frost["fc1_weight_block_scale"]
    assert sf.shape == (2, e, 256, h // 16) and sf.is_contiguous()
    linear = _nvfp4_unswizzle_scales(canonical["fc1_weight_block_scale"], 2 * i, 8)
    for branch in range(2):
        rows = _nvfp4_unswizzle_scales(sf[branch], 256, 8)
        assert torch.equal(rows[:, :i], linear[:, branch * i : (branch + 1) * i])
        assert not rows[:, i:].any()
    # The pair is an exact inverse, padding aside.
    assert torch.equal(
        _nvfp4_swizzle_scales(linear), canonical["fc1_weight_block_scale"]
    )
    # Gated FC1 with I % 128 == 0, and non-gated FC1, keep the canonical view.
    for activation, size in ((SwiGLU(), 256), (ReLU2(), 192)):
        rows = 2 * size if activation.is_gated else size
        a = torch.randn(e, rows, h, device="cuda", generator=gen).bfloat16()
        b = torch.randn(e, h, size, device="cuda", generator=gen).bfloat16()
        kwargs = dict(common, intermediate_size=size, activation=activation)
        assert torch.equal(
            CudnnFrostNvfp4Config.prepare_weights(a, b, **kwargs)[
                "fc1_weight_block_scale"
            ],
            CutlassNvfp4Config.prepare_weights(a, b, **kwargs)[
                "fc1_weight_block_scale"
            ],
        )


def _sm120_case():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("swiglu_param sources are packaged for SM120a")
    try:
        require_graph_resource_retention()
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def _reference(dtype, act, weights, activation):
    if dtype == "bf16":
        return _bf16_moe_reference(act, weights, activation)
    if dtype == "nvfp4":
        return _nvfp4_moe_reference(act, weights, activation)
    return _mxfp8_moe_reference(act, weights, activation, mixed=dtype == "mxfp8_mxfp4")


def _run(dtype, activation, geometry, tokens, seed=17):
    cls, weight_format, act_format = FORMATS[dtype]
    e, h, i, k = geometry
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=k),
        quant=QuantConfig(weight=weight_format, activation=act_format),
        experts=ExpertConfig(intermediate_size=i),
        backend=BackendOptions((cls(),)),
        activation=activation,
        execution=ExecutionConfig(tune_max_num_tokens=max(tokens, 16)),
    )
    layer = MoELayer(config)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # |gate|, |up| ~ 6: both clamps and the sigmoid knee are exercised.
    scale = 3.0 / h**0.5
    w1 = torch.randn(e, 2 * i, h, device="cuda", generator=gen) * scale
    w2 = torch.randn(e, h, i, device="cuda", generator=gen) * (0.5 / i**0.5)
    w1, w2 = w1.bfloat16(), w2.bfloat16()
    view = cls.prepare_weights(
        w1,
        w2,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
    )
    x = torch.randn(tokens, h, device="cuda", generator=gen).mul(2.0).bfloat16()
    xq, xsf = cls.prepare_activations(x, quant=config.quant)
    ids = torch.rand(tokens, e, device="cuda", generator=gen).topk(k).indices.int()
    scores = torch.rand(tokens, k, device="cuda", generator=gen).softmax(-1)
    act = MoEActivationPack(xq, xsf, ids, scores)
    own = MoEWeightPack({f"cudnn_frost_{dtype}": view})
    if dtype == "nvfp4":
        # The reference reads the canonical [up; gate] scale matrix.
        view = CutlassNvfp4Config.prepare_weights(
            w1, w2, num_local_experts=e, hidden_size=h, intermediate_size=i
        )
    shared = MoEWeightPack({f"cutlass_{dtype}": view})
    with autotune(tuning_buckets=(tokens,)):
        out = layer(act, own).clone()
    assert layer.winner_backend == f"cudnn_frost_{dtype}"
    reference = _reference(dtype, act, shared, activation)
    default = _reference(dtype, act, shared, SwiGLU())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = layer(act, own)
    replayed.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    return out, replayed, reference, default


def _relative(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("dtype", FORMATS)
@pytest.mark.parametrize("activation", [CLAMPED, OAI], ids=["limit10", "oai"])
@pytest.mark.parametrize("tokens", [3, 200])
def test_swiglu_param_moe_layer(dtype, activation, tokens):
    _sm120_case()
    out, replayed, reference, default = _run(
        dtype, activation, (16, 1024, 512, 4), tokens
    )
    tolerance = 0.005 if dtype == "bf16" else 0.03
    assert torch.isfinite(out).all()
    assert _relative(out, reference) < tolerance
    torch.testing.assert_close(replayed, out)
    # The parameters matter: the default SwiGLU result is far from this one.
    assert _relative(default, reference) > 10 * tolerance


@pytest.mark.parametrize("dtype", ("bf16", "nvfp4"))
@pytest.mark.parametrize("tokens", [1, 37, 512])
def test_gpt_oss_geometry(dtype, tokens):
    """gpt-oss-20b: E=32, H=I=2880, top-4, SwiGLU(1.702, 1, 7) on SM120."""
    _sm120_case()
    out, replayed, reference, default = _run(dtype, OAI, (32, 2880, 2880, 4), tokens)
    tolerance = 0.005 if dtype == "bf16" else 0.03
    assert torch.isfinite(out).all()
    assert _relative(out, reference) < tolerance
    torch.testing.assert_close(replayed, out)
    assert _relative(default, reference) > 10 * tolerance


@pytest.mark.parametrize("activation", [CLAMPED, SwiGLU()], ids=["limit10", "default"])
def test_nvfp4_half_atom_intermediate(activation):
    """H = I = 576 (4.5 atoms): per-branch FC1 scales and padded FC2 scales."""
    _sm120_case()
    out, replayed, reference, _ = _run("nvfp4", activation, (8, 576, 576, 2), 77)
    assert torch.isfinite(out).all()
    assert _relative(out, reference) < 0.03
    torch.testing.assert_close(replayed, out)
