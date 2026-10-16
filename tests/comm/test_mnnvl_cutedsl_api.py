# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side checks of the ``backend="mnnvl-cutedsl"`` factory and support probe.

These need neither a GPU nor a process group: the factory rejects its
arguments before any allocation, and the probe's device queries are patched.
The collective path is covered by test_mnnvl_cutedsl_allreduce_api.py.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="requires nvidia-cutlass-dsl")

import flashinfer.comm as comm
import flashinfer.comm.mnnvl_cutedsl_ar as cutedsl_ar
from flashinfer.comm import (
    create_allreduce_fusion_workspace,
    is_mnnvl_cutedsl_supported,
)
from flashinfer.comm.mnnvl_cutedsl import (
    DEFAULT_CONFIG,
    HT_ONLY_CONFIG,
    NO_NORM_CONFIG,
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
    LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT,
    LLFinalizeTuning,
)

_SHAPE = dict(world_size=4, rank=0, max_token_num=64, hidden_dim=8192)


def _ll_route(preset) -> MRangeDispatch:
    return MRangeDispatch(
        upper_bounds=(None,), targets=(KernelTarget(ProtocolKind.LL, preset),)
    )


_TWO_SHOT_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=4,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=_ll_route(LLFinalizeTuning()),
            all_reduce_routes=_ll_route(LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT),
        ),
    ),
)


@pytest.mark.parametrize("backend", ("auto", "trtllm", "mnnvl"))
@pytest.mark.parametrize(
    "option",
    (
        {"top_k": 10},
        {"rms_eps": 1e-5},
        {"weight_bias": 1.0},
        {"routed_scaling_factor": 2.0},
        {"include_shared_expert": False},
        {"add_residual": False},
        {"write_residual_output": True},
        {"apply_rms_norm": False},
        {"output_dtype": torch.float8_e4m3fn},
        {"dynamic_fp8": True},
        {"write_norm_output": True},
        {"config": DEFAULT_CONFIG},
    ),
    ids=lambda option: next(iter(option)),
)
def test_other_backends_reject_cutedsl_options(backend, option):
    with pytest.raises(ValueError, match="Only backend='mnnvl-cutedsl' accepts"):
        create_allreduce_fusion_workspace(
            backend=backend, dtype=torch.bfloat16, **_SHAPE, **option
        )


@pytest.mark.parametrize(
    "option,name",
    (
        ({"gpus_per_node": 4}, "gpus_per_node"),
        ({"comm_backend": object()}, "comm_backend"),
        ({"force_oneshot_support": True}, "force_oneshot_support"),
    ),
)
def test_cutedsl_backend_rejects_unused_arguments(option, name):
    with pytest.raises(ValueError, match=f"does not use: {name}"):
        create_allreduce_fusion_workspace(
            backend="mnnvl-cutedsl", dtype=torch.bfloat16, **_SHAPE, **option
        )


def test_cutedsl_backend_forwards_options(monkeypatch):
    captured = {}

    class _Workspace:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cutedsl_ar, "MNNVLCuteDSLAllReduceFusionWorkspace", _Workspace)
    group = object()
    workspace = create_allreduce_fusion_workspace(
        backend="mnnvl-cutedsl",
        dtype=torch.bfloat16,
        group=group,
        apply_rms_norm=False,
        add_residual=False,
        rms_eps=1e-5,
        **_SHAPE,
    )
    assert isinstance(workspace, _Workspace)
    assert captured == dict(
        tp_size=4,
        tp_rank=0,
        max_token_num=64,
        hidden_dim=8192,
        dtype=torch.bfloat16,
        group=group,
        apply_rms_norm=False,
        add_residual=False,
        rms_eps=1e-5,
    )


def test_probe_is_exported():
    assert comm.is_mnnvl_cutedsl_supported is is_mnnvl_cutedsl_supported


@pytest.fixture
def fake_device(monkeypatch):
    """Pretend the current device is an MNNVL-capable GPU of a chosen arch."""
    state = {"capability": (10, 3), "symm_mem": True, "multicast": True}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: state["capability"]
    )
    monkeypatch.setattr(
        cutedsl_ar.symm_mem,
        "get_backend",
        lambda device: "NVSHMEM" if state["symm_mem"] else None,
    )
    monkeypatch.setattr(
        cutedsl_ar, "is_multicast_supported", lambda index: state["multicast"]
    )
    return state


def _probe(**overrides):
    kwargs = dict(
        tp_size=4,
        hidden_dim=8192,
        dtype=torch.bfloat16,
        max_token_num=16384,
    )
    kwargs.update(overrides)
    tp_size = kwargs.pop("tp_size")
    hidden_dim = kwargs.pop("hidden_dim")
    dtype = kwargs.pop("dtype")
    return is_mnnvl_cutedsl_supported(tp_size, hidden_dim, dtype, **kwargs)


@pytest.mark.parametrize("capability", ((10, 3), (10, 7)))
@pytest.mark.parametrize("apply_rms_norm", (True, False))
def test_probe_accepts_profiled_architectures(fake_device, capability, apply_rms_norm):
    fake_device["capability"] = capability
    assert _probe(apply_rms_norm=apply_rms_norm)
    assert _probe(apply_rms_norm=apply_rms_norm, device=0)
    assert _probe(apply_rms_norm=apply_rms_norm, device="cuda:0")


@pytest.mark.parametrize(
    "overrides",
    (
        {"tp_size": 3},
        {"dtype": torch.float16},
        {"max_token_num": 0},
        {"hidden_dim": 4096},
        {"top_k": 8},
        {"device": "cpu"},
        # A norm-on config paired with a norm-free workspace.
        {"apply_rms_norm": False, "config": DEFAULT_CONFIG},
        # HT cannot compile the norm out.
        {"apply_rms_norm": False, "config": HT_ONLY_CONFIG, "tp_size": 8},
        {"apply_rms_norm": False, "write_residual_output": False},
        {"apply_rms_norm": False, "output_dtype": torch.float8_e4m3fn},
        {"dynamic_fp8": True},
        {"write_norm_output": True},
        # The LL two-shot all-reduce has no dynamic FP8 epilogue.
        {
            "output_dtype": torch.float8_e4m3fn,
            "dynamic_fp8": True,
            "config": _TWO_SHOT_CONFIG,
            "max_token_num": 64,
        },
    ),
    ids=(
        "tp",
        "dtype",
        "capacity",
        "hidden",
        "top_k",
        "cpu",
        "config-norm-mismatch",
        "ht-no-norm",
        "no-output",
        "no-norm-fp8",
        "dynamic-bf16",
        "norm-output-bf16",
        "two-shot-dynamic-fp8",
    ),
)
def test_probe_rejects_unsupported_requests(fake_device, overrides):
    assert not _probe(**overrides)


@pytest.mark.parametrize("capability", ((10, 3), (10, 7)))
@pytest.mark.parametrize("dynamic_fp8", (False, True))
@pytest.mark.parametrize("write_norm_output", (False, True))
def test_probe_accepts_fp8_outputs(
    fake_device, capability, dynamic_fp8, write_norm_output
):
    fake_device["capability"] = capability
    assert _probe(
        output_dtype=torch.float8_e4m3fn,
        dynamic_fp8=dynamic_fp8,
        write_norm_output=write_norm_output,
    )


@pytest.mark.parametrize(
    "device_state",
    (
        {"capability": (9, 0)},
        {"capability": (10, 0)},
        {"symm_mem": False},
        {"multicast": False},
    ),
    ids=("hopper", "no-profile-arch", "no-symm-mem", "no-multicast"),
)
def test_probe_rejects_unsupported_devices(fake_device, device_state):
    fake_device.update(device_state)
    assert not _probe()


def test_probe_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not _probe()


def test_probe_defaults_config_by_norm_mode(fake_device):
    fake_device["capability"] = (10, 7)
    assert _probe(apply_rms_norm=False) == _probe(
        apply_rms_norm=False, config=NO_NORM_CONFIG
    )
    assert _probe(apply_rms_norm=True) == _probe(config=DEFAULT_CONFIG)
