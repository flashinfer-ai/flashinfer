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

Experimental MNNVL all-reduce fused with the mHC post-map and the split-K pre-norm partials
(flashinfer.comm.allreduce_mhc_post_prenorm).

* The host tests (supported surface, guards, capture safety, the composable path against the
  reference, containment of the experimental package) run anywhere, without a GPU.
* The kernel tests need 8 ranks on SM100 / SM103 GPUs in one MNNVL domain, e.g.
  ``mpirun -np 8 pytest tests/experimental/test_allreduce_mhc_post_prenorm.py`` or torchrun with
  8 processes; they skip otherwise.
"""

import subprocess
import sys
import traceback
import warnings

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.comm.trtllm_mnnvl_ar import (
    MNNVLAllReduceFusionWorkspace,
    MNNVLAllreduceFusionStrategy,
    trtllm_mnnvl_allreduce,
)
from flashinfer.experimental.allreduce_mhc_post_prenorm import backend

WORLD_SIZE = 8
HIDDEN = 4096
HC = 4
MIX = HC * (2 + HC)
N_SPLITS = 8
MAX_TOKENS = 32

# fp32 summation-order noise of the split-K partials passes; bf16 / TF32 operands or a rounded
# new_r fail.
YP_ATOL, YP_RTOL = 1e-5, 1e-5
RP_ATOL, RP_RTOL = 1e-4, 1e-5
# residual_out: <= 1 bf16 ulp everywhere and >= 99.5% bit-exact against the PyTorch fp32 post-map
# (the kernel's fused multiply-adds round once).
RES_RTOL, RES_MIN_EXACT = 2.0**-7, 0.995


# ---------------------------------------------------------------------------
# Inputs and reference
# ---------------------------------------------------------------------------


def _make_inputs(num_tokens, hidden_size=HIDDEN, hc_mult=HC, device="cuda", seed=0):
    """One rank's inputs. ``post_layer_mix`` is a sigmoid gate scaled by 2 and ``comb_res_mix`` a
    Sinkhorn-normalized (doubly stochastic) mix, as an mHC pre block produces them."""
    g = torch.Generator(device=device).manual_seed(seed)
    mix = hc_mult * (2 + hc_mult)
    inp = 0.5 * torch.randn(num_tokens, hidden_size, device=device, generator=g)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, generator=g)
    post_logits = torch.randn(num_tokens, hc_mult, device=device, generator=g)
    comb = torch.randn(
        num_tokens, hc_mult, hc_mult, device=device, generator=g
    ).softmax(dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(dim=-2, keepdim=True)
        comb = comb / comb.sum(dim=-1, keepdim=True)
    fn = torch.randn(mix, hc_mult * hidden_size, device=device, generator=g) / (
        (hc_mult * hidden_size) ** 0.5
    )
    return {
        "input": inp.to(torch.bfloat16),
        "residual": residual.to(torch.bfloat16),
        "post_layer_mix": (2.0 * torch.sigmoid(post_logits)).contiguous(),
        "comb_res_mix": comb.contiguous(),
        "fn": fn.contiguous(),
    }


@torch.no_grad()
def _reference(x, residual, post_layer_mix, comb_res_mix, fn):
    """Epilogue on the already-reduced ``x``: fp32 post-map, partials accumulated in fp64."""
    num_tokens, hc_mult, hidden_size = residual.shape
    post = post_layer_mix.reshape(num_tokens, hc_mult).float()
    new_r = post.unsqueeze(-1) * x.float().unsqueeze(1)
    for k in range(hc_mult):
        new_r = new_r + comb_res_mix[:, k, :].float().unsqueeze(-1) * residual[
            :, k, :
        ].float().unsqueeze(1)
    width = hidden_size // N_SPLITS
    r64 = new_r.double().view(num_tokens, hc_mult, N_SPLITS, width)
    f64 = fn.double().view(fn.shape[0], hc_mult, N_SPLITS, width)
    yp = torch.einsum("mjsh,njsh->smn", r64, f64).float()
    rp = r64.square().sum(dim=(1, 3)).transpose(0, 1).float()
    return yp, rp, new_r.to(residual.dtype)


# ---------------------------------------------------------------------------
# Host tests (no GPU needed)
# ---------------------------------------------------------------------------


def _stub_workspace(tp_size=WORLD_SIZE, sufficient=True):
    ws = MNNVLAllReduceFusionWorkspace.__new__(MNNVLAllReduceFusionWorkspace)
    ws.tp_size = tp_size
    ws.rank = 0
    ws._destroyed = True  # nothing to free
    ws.is_buffer_size_sufficient = lambda *args, **kwargs: sufficient
    return ws


@pytest.fixture
def sm10x(monkeypatch):
    monkeypatch.setattr(backend, "_device_cc_major", lambda device_index: 10)


def test_supported_surface():
    assert backend.WORLD_SIZE == WORLD_SIZE
    assert backend.HIDDEN_SIZE == HIDDEN
    assert backend.HC_MULT == HC
    assert backend.MIX == MIX
    assert backend.NUM_SPLITS == N_SPLITS
    assert backend.MAX_TOKENS == MAX_TOKENS
    assert backend.DTYPE is torch.bfloat16
    assert sorted(backend.COMPUTE_CAPABILITY_MAJORS) == [10]


def test_entry_points_are_experimental():
    for fn in (
        comm.allreduce_mhc_post_prenorm,
        comm.allreduce_mhc_post_prenorm_supported,
        comm.allreduce_mhc_post_prenorm_prepare,
    ):
        assert fn.is_experimental is True
        assert "experimental" in fn.__doc__


def test_import_does_not_load_the_backend():
    # Core reaches the experimental package only through deferred imports.
    code = (
        "import sys, flashinfer, flashinfer.comm; "
        "loaded = [m for m in sys.modules "
        "if m.startswith('flashinfer.experimental.allreduce_mhc_post_prenorm')]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_calling_is_the_opt_in_and_warns_once():
    # No environment variable: the first call warns once with ExperimentalWarning.
    code = (
        "import warnings, torch, flashinfer.comm as c\n"
        "from flashinfer.api_logging import ExperimentalWarning\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    for _ in range(2):\n"
        "        assert not c.allreduce_mhc_post_prenorm_supported(\n"
        "            object(), 1, 4096, 4, torch.bfloat16, torch.device('cpu'))\n"
        "hits = [w for w in caught if issubclass(w.category, ExperimentalWarning)]\n"
        "assert len(hits) == 1, [str(w.message) for w in caught]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_jit_spec_is_numerics_preserving(monkeypatch):
    from flashinfer.experimental.allreduce_mhc_post_prenorm import jit as jit_mod
    from flashinfer.jit import core

    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        jit_mod.current_compilation_context,
        "get_nvcc_flags_list",
        lambda **kwargs: ["-gencode=arch=compute_103a,code=sm_103a"],
    )
    spec = jit_mod.gen_allreduce_mhc_post_prenorm_module()
    assert "-use_fast_math" not in spec.extra_cuda_cflags
    assert all(source.is_file() for source in spec.sources)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 8, 9, 16, 17, 32])
def test_guard_accepts_supported_surface(sm10x, num_tokens):
    cuda0 = torch.device("cuda", 0)
    assert backend._shape_supported(
        _stub_workspace(), num_tokens, HIDDEN, HC, torch.bfloat16, cuda0
    )


@pytest.mark.parametrize(
    "workspace,num_tokens,hidden,hc,dtype,device",
    [
        ("ws", 0, HIDDEN, HC, torch.bfloat16, "cuda"),
        ("ws", 33, HIDDEN, HC, torch.bfloat16, "cuda"),
        ("ws", 1, 7168, HC, torch.bfloat16, "cuda"),
        ("ws", 1, HIDDEN, 2, torch.bfloat16, "cuda"),
        ("ws", 1, HIDDEN, HC, torch.float16, "cuda"),
        ("ws", 1, HIDDEN, HC, torch.bfloat16, "cpu"),
        ("tp4", 1, HIDDEN, HC, torch.bfloat16, "cuda"),
        ("small", 1, HIDDEN, HC, torch.bfloat16, "cuda"),
        ("other", 1, HIDDEN, HC, torch.bfloat16, "cuda"),
    ],
)
def test_guard_rejects(sm10x, workspace, num_tokens, hidden, hc, dtype, device):
    ws = {
        "ws": _stub_workspace(),
        "tp4": _stub_workspace(tp_size=4),
        "small": _stub_workspace(sufficient=False),
        "other": object(),
    }[workspace]
    dev = torch.device(device, 0) if device == "cuda" else torch.device(device)
    assert not backend._shape_supported(ws, num_tokens, hidden, hc, dtype, dev)


@pytest.mark.parametrize("major", [8, 9, 11, 12])
def test_guard_rejects_other_architectures(monkeypatch, major):
    monkeypatch.setattr(backend, "_device_cc_major", lambda device_index: major)
    assert not backend._shape_supported(
        _stub_workspace(), 1, HIDDEN, HC, torch.bfloat16, torch.device("cuda", 0)
    )


def test_capturing_call_never_builds(sm10x, monkeypatch):
    def fail_build():
        raise AssertionError("the module must not be built during graph capture")

    monkeypatch.setattr(backend, "_module", None)
    monkeypatch.setattr(backend, "_build_module", fail_build)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    before = backend.stats()["capture_unready_count"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert not comm.allreduce_mhc_post_prenorm_supported(
            _stub_workspace(), 1, HIDDEN, HC, torch.bfloat16, torch.device("cuda", 0)
        )
        # prepare() during capture reports readiness and never builds either.
        assert not comm.allreduce_mhc_post_prenorm_prepare()
    assert backend.stats()["capture_unready_count"] == before + 1
    assert backend.stats()["load_error"] is None


def test_stats_keys():
    stats = backend.stats()
    for key in (
        "module_loaded",
        "load_error",
        "dispatch_count",
        "dispatch_by_num_tokens",
        "fallback_count",
        "capture_unready_count",
    ):
        assert key in stats


@pytest.mark.parametrize("num_tokens", [1, 3, 33])
def test_composable_path_matches_reference(monkeypatch, num_tokens):
    # Outside the supported surface the API all-reduces with the stock MNNVL kernel and applies
    # the epilogue in PyTorch; an identity all-reduce checks the epilogue and the preallocated
    # output contract without GPUs.
    monkeypatch.setattr(backend, "is_supported", lambda *args: False)
    monkeypatch.setattr(
        backend, "trtllm_mnnvl_allreduce", lambda x, ws, pdl, strategy: x
    )
    inputs = _make_inputs(num_tokens, hidden_size=256, device="cpu", seed=num_tokens)
    yp = torch.empty(N_SPLITS, num_tokens, MIX)
    rp = torch.empty(N_SPLITS, num_tokens)
    residual_out = torch.empty_like(inputs["residual"])
    before = backend.stats()["fallback_count"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out = comm.allreduce_mhc_post_prenorm(
            **inputs,
            workspace=_stub_workspace(),
            yp=yp,
            rp=rp,
            residual_out=residual_out,
        )
    x = inputs.pop("input")
    ref = _reference(x, **inputs)
    assert out[0] is yp and out[1] is rp and out[2] is residual_out
    for got, want in zip(out, ref, strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert backend.stats()["fallback_count"] == before + 1


def _packed_mix_views(num_tokens, hc_mult=HC, device="cpu", seed=0):
    """``post_layer_mix`` / ``comb_res_mix`` as an mHC pre block often produces them: views
    into one packed ``[T, hc * (2 + hc)]`` mix tensor (pre, post, comb), hence non-contiguous."""
    g = torch.Generator(device=device).manual_seed(seed)
    packed = torch.rand(num_tokens, MIX, device=device, generator=g)
    post = packed[:, hc_mult : 2 * hc_mult].unsqueeze(
        -1
    )  # [T, hc, 1], strides (MIX, 1, 1)
    comb = packed[:, 2 * hc_mult :].view(
        num_tokens, hc_mult, hc_mult
    )  # strides (MIX, hc, 1)
    if num_tokens > 1:  # a single row is contiguous whatever its row stride
        assert not post.reshape(num_tokens, hc_mult).is_contiguous()
        assert not comb.is_contiguous()
    return post, comb


def _strided_outputs(num_tokens, hidden_size, hc_mult=HC, device="cpu"):
    yp = torch.empty(num_tokens, N_SPLITS, MIX, device=device).transpose(0, 1)
    rp = torch.empty(num_tokens, N_SPLITS, device=device).t()
    residual_out = torch.empty(
        num_tokens, hidden_size, hc_mult, dtype=torch.bfloat16, device=device
    ).permute(0, 2, 1)
    assert not (
        yp.is_contiguous() or rp.is_contiguous() or residual_out.is_contiguous()
    )
    return yp, rp, residual_out


class _ContiguityCheckingModule:
    """Stands in for the JIT module: rejects non-contiguous tensors like the binding's
    ``CHECK_INPUT_AND_TYPE`` and writes recognisable values into the outputs."""

    def __init__(self):
        self.seen = None
        self.ptrs = None
        self.launch_with_pdl = None

    def allreduce_mhc_post_prenorm(
        self, input, residual, post, comb, fn, yp, rp, residual_out, *rest
    ):
        tensors = {
            "input": input,
            "residual": residual,
            "post": post,
            "comb": comb,
            "fn": fn,
            "yp": yp,
            "rp": rp,
            "residual_out": residual_out,
        }
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        self.seen = {name: tensor.clone() for name, tensor in tensors.items()}
        self.ptrs = {name: tensor.data_ptr() for name, tensor in tensors.items()}
        self.launch_with_pdl = rest[-1]
        yp.fill_(1.0)
        rp.fill_(2.0)
        residual_out.fill_(3.0)


def _fused_stub(monkeypatch):
    module = _ContiguityCheckingModule()
    monkeypatch.setattr(backend, "is_supported", lambda *args, **kwargs: True)
    monkeypatch.setattr(backend, "_get_module", lambda allow_build: module)
    workspace = _stub_workspace()
    workspace._require_handles_attached = lambda: None
    workspace.mc_ptr = 0
    workspace.uc_ptr_local = 0
    workspace.buffer_flags = None
    return module, workspace


@pytest.mark.parametrize(
    "num_tokens", [2, 3, 32]
)  # one token is contiguous at any stride
def test_fused_path_accepts_strided_inputs_and_outputs(monkeypatch, num_tokens):
    # A call the support query accepts must not fail on layout: strided inputs reach the
    # kernel as contiguous copies with the same values, and strided preallocated outputs
    # receive the results.
    module, workspace = _fused_stub(monkeypatch)
    hidden = 256
    inputs = _make_inputs(num_tokens, hidden_size=hidden, device="cpu", seed=num_tokens)
    post, comb = _packed_mix_views(num_tokens, seed=num_tokens)
    inputs["post_layer_mix"], inputs["comb_res_mix"] = post, comb
    inputs["residual"] = (
        inputs["residual"].permute(0, 2, 1).contiguous().permute(0, 2, 1)
    )  # same values, non-contiguous
    assert not inputs["residual"].is_contiguous()
    yp, rp, residual_out = _strided_outputs(num_tokens, hidden)
    before = backend.stats()["dispatch_count"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out = comm.allreduce_mhc_post_prenorm(
            **inputs, workspace=workspace, yp=yp, rp=rp, residual_out=residual_out
        )
    assert backend.stats()["dispatch_count"] == before + 1
    seen = module.seen
    assert torch.equal(seen["post"], post.reshape(num_tokens, HC))
    assert torch.equal(seen["comb"], comb)
    assert torch.equal(seen["residual"], inputs["residual"])
    assert out[0] is yp and out[1] is rp and out[2] is residual_out
    assert bool((yp == 1.0).all()) and bool((rp == 2.0).all())
    assert bool((residual_out == 3.0).all())


def test_off_contract_dtypes_take_the_composable_path(monkeypatch):
    # The support query only sees the input dtype; a mix tensor in another dtype must fall
    # back to the composable path instead of reaching the binding's dtype check.
    module, workspace = _fused_stub(monkeypatch)
    monkeypatch.setattr(
        backend, "trtllm_mnnvl_allreduce", lambda x, ws, pdl, strategy: x
    )
    inputs = _make_inputs(2, hidden_size=256, device="cpu", seed=7)
    inputs["post_layer_mix"] = inputs["post_layer_mix"].to(torch.bfloat16)
    before = backend.stats()["fallback_count"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out = comm.allreduce_mhc_post_prenorm(**inputs, workspace=workspace)
    assert module.seen is None
    assert backend.stats()["fallback_count"] == before + 1
    x = inputs.pop("input")
    for got, want in zip(out, _reference(x, **inputs), strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def _recording_allreduce(calls):
    """Identity all-reduce that records the layout of its input and the strategy."""

    def allreduce(x, ws, pdl, strategy):
        calls.append((x.is_contiguous(), strategy))
        return x

    return allreduce


def test_composable_path_passes_a_dense_input(monkeypatch):
    # The stock all-reduce reads dense rows: a strided input reaches it as a contiguous copy.
    calls = []
    monkeypatch.setattr(backend, "is_supported", lambda *args: False)
    monkeypatch.setattr(backend, "trtllm_mnnvl_allreduce", _recording_allreduce(calls))
    num_tokens, hidden = 3, 256
    inputs = _make_inputs(num_tokens, hidden_size=hidden, device="cpu", seed=11)
    wide = torch.zeros(num_tokens, 2 * hidden, dtype=torch.bfloat16)
    wide[:, :hidden] = inputs["input"]
    inputs["input"] = wide[:, :hidden]
    assert not inputs["input"].is_contiguous()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out = comm.allreduce_mhc_post_prenorm(**inputs, workspace=_stub_workspace())
    assert [contiguous for contiguous, _ in calls] == [True]
    x = inputs.pop("input")
    for got, want in zip(out, _reference(x, **inputs), strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize(
    "num_tokens, hidden, strategy",
    [
        (1, HIDDEN, MNNVLAllreduceFusionStrategy.ONESHOT),
        (16, HIDDEN, MNNVLAllreduceFusionStrategy.ONESHOT),
        (17, HIDDEN, MNNVLAllreduceFusionStrategy.ONESHOT),
        (32, HIDDEN, MNNVLAllreduceFusionStrategy.ONESHOT),
        (33, HIDDEN, MNNVLAllreduceFusionStrategy.AUTO),
        (17, 256, MNNVLAllreduceFusionStrategy.AUTO),
    ],
)
def test_fallback_all_reduce_matches_the_fused_protocol(
    sm10x, monkeypatch, num_tokens, hidden, strategy
):
    # A rank whose module failed to load falls back while the other ranks launch the fused
    # one-shot kernel, so inside the surface its all-reduce is the one-shot kernel too, also
    # where the stock choice is two-shot (17 to 32 tokens). Outside it the stock choice applies.
    calls = []
    shape_supported = backend._shape_supported
    monkeypatch.setattr(
        backend,
        "_shape_supported",
        lambda ws, m, h, hc, dtype, device: shape_supported(
            ws, m, h, hc, dtype, torch.device("cuda", 0)
        ),
    )
    monkeypatch.setattr(backend, "_module", None)
    monkeypatch.setattr(backend, "_stats", dict(backend._stats, load_error="failed"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(backend, "trtllm_mnnvl_allreduce", _recording_allreduce(calls))
    inputs = _make_inputs(num_tokens, hidden_size=hidden, device="cpu", seed=13)
    before = backend.stats()["fallback_count"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        comm.allreduce_mhc_post_prenorm(**inputs, workspace=_stub_workspace())
    assert calls == [(True, strategy)]
    assert backend.stats()["fallback_count"] == before + 1


def test_overlap_check():
    buf = torch.empty(2, 3, 4, 8)
    assert backend._overlaps(buf[0], buf[0])
    assert backend._overlaps(buf[0], buf[0, 1:])
    assert backend._overlaps(buf[:, 0], buf[:, 1])  # interleaved spans: conservative
    assert not backend._overlaps(buf[0], buf[1])
    assert not backend._overlaps(buf[0], torch.empty(3, 4, 8))


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 32])
def test_in_place_residual_update(monkeypatch, num_tokens):
    # residual_out may be residual. In the 1- to 4-token kernels several CTAs read the residual
    # slice that one of them overwrites, so they get a separate output that is copied back; from
    # 5 tokens the kernel updates residual in place.
    module, workspace = _fused_stub(monkeypatch)
    inputs = _make_inputs(num_tokens, hidden_size=256, device="cpu", seed=num_tokens)
    residual = inputs["residual"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        out = comm.allreduce_mhc_post_prenorm(
            **inputs, workspace=workspace, residual_out=residual
        )
    assert out[2] is residual and bool((residual == 3.0).all())
    assert module.ptrs["residual"] == residual.data_ptr()
    in_place = module.ptrs["residual_out"] == residual.data_ptr()
    assert in_place == (num_tokens >= backend.IN_PLACE_MIN_TOKENS)


def test_disjoint_residual_out_in_the_same_buffer_is_written_directly(monkeypatch):
    module, workspace = _fused_stub(monkeypatch)
    inputs = _make_inputs(1, hidden_size=256, device="cpu", seed=3)
    buf = torch.empty((2, *inputs["residual"].shape), dtype=torch.bfloat16)
    buf[0] = inputs["residual"]
    inputs["residual"] = buf[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        comm.allreduce_mhc_post_prenorm(
            **inputs, workspace=workspace, residual_out=buf[1]
        )
    assert module.ptrs["residual_out"] == buf[1].data_ptr()


@pytest.mark.parametrize("strided_fn", [False, True])
def test_copied_fn_launches_without_pdl(monkeypatch, strided_fn):
    # The kernel loads fn before griddepcontrol.wait. A copy of fn is written by the kernel
    # launched right before, so that launch must not start early.
    module, workspace = _fused_stub(monkeypatch)
    inputs = _make_inputs(2, hidden_size=256, device="cpu", seed=5)
    if strided_fn:
        inputs["fn"] = inputs["fn"].t().contiguous().t()  # same values, non-contiguous
        assert not inputs["fn"].is_contiguous()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        comm.allreduce_mhc_post_prenorm(
            **inputs, workspace=workspace, launch_with_pdl=True
        )
    assert module.launch_with_pdl is (not strided_fn)
    assert torch.equal(module.seen["fn"], inputs["fn"])


# ---------------------------------------------------------------------------
# Kernel tests (8 ranks, SM100 / SM103, MNNVL)
# ---------------------------------------------------------------------------


def _rank_inputs(num_tokens, rank, seed):
    """The same tensors on every rank; this rank's un-reduced partial is ``parts[rank]``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    parts = 0.35 * torch.randn(
        WORLD_SIZE, num_tokens, HIDDEN, device="cuda", generator=g
    )
    parts[:, :, ::509] *= 8.0  # outlier channels
    # Cancellation channels: the last rank nearly cancels the other seven, so a reduction order
    # other than 0..7 in fp32 shows up after the bf16 rounding.
    noise = 1e-3 * torch.randn(
        num_tokens, parts[0, :, 3::7].shape[-1], device="cuda", generator=g
    )
    parts[WORLD_SIZE - 1, :, 3::7] = -parts[: WORLD_SIZE - 1, :, 3::7].sum(0) + noise
    parts = parts.to(torch.bfloat16)
    epilogue = _make_inputs(num_tokens, device="cuda", seed=seed + 1)
    epilogue.pop("input")
    return parts, epilogue


def _rank_order_allreduce(parts):
    acc = torch.zeros(parts.shape[1:], dtype=torch.float32, device=parts.device)
    for r in range(parts.shape[0]):
        acc += parts[r].float()
    return acc.to(torch.bfloat16)


def _check_epilogue(out, ref):
    yp, rp, res = out
    yp_ref, rp_ref, res_ref = ref
    assert torch.isfinite(yp).all() and torch.isfinite(rp).all()
    assert ((yp - yp_ref).abs() <= YP_ATOL + YP_RTOL * yp_ref.abs()).all()
    assert ((rp - rp_ref).abs() <= RP_ATOL + RP_RTOL * rp_ref.abs()).all()
    err = (res.float() - res_ref.float()).abs()
    assert (err <= 1e-6 + RES_RTOL * res_ref.float().abs()).all()
    assert (res == res_ref).float().mean().item() >= RES_MIN_EXACT


@pytest.fixture(scope="module")
def mnnvl_workspace():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import TorchDistBackend
    from tests.test_helpers.comm import init_torch_distributed_from_mpi

    try:
        init_torch_distributed_from_mpi()
    except Exception as exc:  # no launcher / single process
        pytest.skip(f"requires an MPI / torchrun launch with {WORLD_SIZE} ranks: {exc}")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if world_size != WORLD_SIZE:
        pytest.skip(f"the fused kernel needs {WORLD_SIZE} ranks, got {world_size}")
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=world_size,
    )
    torch.cuda.set_device(mapping.local_rank)
    majors = [None] * world_size
    dist.all_gather_object(majors, torch.cuda.get_device_capability()[0])
    if any(m != 10 for m in majors):
        pytest.skip("the fused kernel needs SM100 / SM103 on every rank")
    workspace = MNNVLAllReduceFusionWorkspace(
        mapping,
        max_num_tokens=MAX_TOKENS,
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        comm_backend=TorchDistBackend(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert comm.allreduce_mhc_post_prenorm_prepare(), backend.stats()["load_error"]
    yield workspace, rank
    dist.barrier()
    workspace.destroy()


def _run_ranks(fn):
    """Run ``fn`` on every rank; fail on every rank if any rank failed."""
    failure = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            fn()
    except Exception as exc:  # noqa: BLE001
        failure = f"rank {dist.get_rank()}: {exc}\n{traceback.format_exc()}"
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    failed = [f for f in failures if f is not None]
    assert not failed, "\n".join(failed)


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 24, 25, 31, 32]
)
def test_fused_matches_stock_allreduce_and_reference(mnnvl_workspace, num_tokens):
    workspace, rank = mnnvl_workspace

    def body():
        parts, epi = _rank_inputs(num_tokens, rank, seed=1000 + num_tokens)
        x_local = parts[rank].contiguous()
        # The stock one-shot all-reduce is the deterministic rank-order sum.
        x = trtllm_mnnvl_allreduce(
            x_local,
            workspace,
            launch_with_pdl=True,
            strategy=MNNVLAllreduceFusionStrategy.ONESHOT,
        )
        assert torch.equal(x, _rank_order_allreduce(parts))
        ref = _reference(x, **epi)
        assert comm.allreduce_mhc_post_prenorm_supported(
            workspace, num_tokens, HIDDEN, HC, torch.bfloat16, x_local.device
        )
        before = backend.stats()["dispatch_count"]
        out = comm.allreduce_mhc_post_prenorm(x_local, workspace=workspace, **epi)
        torch.cuda.synchronize()
        assert backend.stats()["dispatch_count"] == before + 1
        _check_epilogue(out, ref)

    _run_ranks(body)


@pytest.mark.parametrize("num_tokens", [2, 5, 17, 32])
def test_fused_accepts_packed_mix_and_strided_outputs(mnnvl_workspace, num_tokens):
    # post / comb as views into one packed [T, 24] mix tensor, a strided input and fn, and
    # strided preallocated outputs take the fused path and give exactly the results of the
    # contiguous call.
    workspace, rank = mnnvl_workspace

    def body():
        parts, epi = _rank_inputs(num_tokens, rank, seed=3000 + num_tokens)
        x_local = parts[rank].contiguous()
        dense = comm.allreduce_mhc_post_prenorm(x_local, workspace=workspace, **epi)
        packed = torch.zeros(num_tokens, MIX, device="cuda")
        packed[:, HC : 2 * HC] = epi["post_layer_mix"]
        packed[:, 2 * HC :] = epi["comb_res_mix"].reshape(num_tokens, HC * HC)
        strided = dict(epi)
        strided["post_layer_mix"] = packed[:, HC : 2 * HC].unsqueeze(-1)
        strided["comb_res_mix"] = packed[:, 2 * HC :].view(num_tokens, HC, HC)
        strided["fn"] = epi["fn"].t().contiguous().t()
        x_wide = torch.zeros(
            num_tokens, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        x_wide[:, :HIDDEN] = x_local
        yp, rp, residual_out = _strided_outputs(num_tokens, HIDDEN, device="cuda")
        before = backend.stats()["dispatch_count"]
        out = comm.allreduce_mhc_post_prenorm(
            x_wide[:, :HIDDEN],
            workspace=workspace,
            yp=yp,
            rp=rp,
            residual_out=residual_out,
            **strided,
        )
        torch.cuda.synchronize()
        assert backend.stats()["dispatch_count"] == before + 1
        assert out[0] is yp and out[1] is rp and out[2] is residual_out
        for got, want in zip(out, dense, strict=True):
            assert torch.equal(got, want)

    _run_ranks(body)


def test_interleaves_with_stock_allreduce(mnnvl_workspace):
    # Same workspace and Lamport buffers as the stock one-shot / two-shot kernels.
    workspace, rank = mnnvl_workspace

    def body():
        for step, num_tokens in enumerate([1, 32, 2, 4, 3, 17, 8, 1]):
            parts, epi = _rank_inputs(num_tokens, rank, seed=2000 + step)
            x = _rank_order_allreduce(parts)
            out = comm.allreduce_mhc_post_prenorm(
                parts[rank].contiguous(), workspace=workspace, **epi
            )
            _check_epilogue(out, _reference(x, **epi))
            stock = trtllm_mnnvl_allreduce(
                parts[rank].contiguous(), workspace, launch_with_pdl=True
            )
            assert torch.equal(stock, x)
        torch.cuda.synchronize()

    _run_ranks(body)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 32])
def test_in_place_residual_update_on_ranks(mnnvl_workspace, num_tokens):
    # residual_out = residual gives exactly the results of the out-of-place call.
    workspace, rank = mnnvl_workspace

    def body():
        parts, epi = _rank_inputs(num_tokens, rank, seed=5000 + num_tokens)
        x_local = parts[rank].contiguous()
        dense = comm.allreduce_mhc_post_prenorm(x_local, workspace=workspace, **epi)
        for _ in range(10):
            in_place = dict(epi, residual=epi["residual"].clone())
            out = comm.allreduce_mhc_post_prenorm(
                x_local,
                workspace=workspace,
                residual_out=in_place["residual"],
                **in_place,
            )
            torch.cuda.synchronize()
            assert out[2] is in_place["residual"]
            for got, want in zip(out, dense, strict=True):
                assert torch.equal(got, want)

    _run_ranks(body)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 17, 32])
def test_fallback_rank_completes_with_fused_ranks(
    mnnvl_workspace, monkeypatch, num_tokens
):
    # One rank cannot load the module and takes the composable path while the others launch the
    # fused kernel: the call completes on every rank with the same all-reduce, also from 17
    # tokens, where the stock all-reduce would choose two-shot.
    workspace, rank = mnnvl_workspace
    fallback_rank = 3
    if rank == fallback_rank:
        monkeypatch.setattr(backend, "_get_module", lambda allow_build: None)

    def body():
        parts, epi = _rank_inputs(num_tokens, rank, seed=6000 + num_tokens)
        before = backend.stats()["dispatch_count"]
        out = comm.allreduce_mhc_post_prenorm(
            parts[rank].contiguous(), workspace=workspace, **epi
        )
        torch.cuda.synchronize()
        dispatched = backend.stats()["dispatch_count"] - before
        assert dispatched == (0 if rank == fallback_rank else 1)
        _check_epilogue(out, _reference(_rank_order_allreduce(parts), **epi))

    _run_ranks(body)


@pytest.mark.parametrize("strided_input", [False, True])
def test_composable_path_on_ranks(mnnvl_workspace, monkeypatch, strided_input):
    # Outside the supported surface: the stock all-reduce (bit-exact) and the PyTorch epilogue,
    # also for a strided input.
    workspace, rank = mnnvl_workspace
    monkeypatch.setattr(backend, "is_supported", lambda *args: False)

    def body():
        parts, epi = _rank_inputs(2, rank, seed=3000)
        x_local = parts[rank].contiguous()
        if strided_input:
            x_wide = torch.zeros(2, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda")
            x_wide[:, :HIDDEN] = x_local
            x_local = x_wide[:, :HIDDEN]
        before = backend.stats()["dispatch_count"]
        out = comm.allreduce_mhc_post_prenorm(x_local, workspace=workspace, **epi)
        torch.cuda.synchronize()
        assert backend.stats()["dispatch_count"] == before
        ref = _reference(_rank_order_allreduce(parts), **epi)
        assert torch.equal(out[2], ref[2])
        _check_epilogue(out, ref)

    _run_ranks(body)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 32])
def test_cuda_graph_capture_and_replay(mnnvl_workspace, num_tokens):
    workspace, rank = mnnvl_workspace

    def body():
        parts, epi = _rank_inputs(num_tokens, rank, seed=4000 + num_tokens)
        x_local = parts[rank].contiguous()
        yp = torch.empty(N_SPLITS, num_tokens, MIX, device="cuda")
        rp = torch.empty(N_SPLITS, num_tokens, device="cuda")
        res = torch.empty_like(epi["residual"])
        eager = [
            t.clone()
            for t in comm.allreduce_mhc_post_prenorm(
                x_local, workspace=workspace, **epi
            )
        ]
        graph = torch.cuda.CUDAGraph()
        before = backend.stats()["dispatch_count"]
        with torch.cuda.graph(graph):
            comm.allreduce_mhc_post_prenorm(
                x_local, workspace=workspace, yp=yp, rp=rp, residual_out=res, **epi
            )
        # The captured call dispatched the fused kernel (no composable path captured).
        assert backend.stats()["dispatch_count"] == before + 1
        for _ in range(3):
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip((yp, rp, res), eager, strict=True):
                assert torch.equal(got, want)

    _run_ranks(body)
