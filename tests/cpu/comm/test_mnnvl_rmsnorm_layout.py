# Copyright (c) 2026, FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer.comm import trtllm_mnnvl_ar
from flashinfer.comm.allreduce import AllReduceFusionPattern, allreduce_fusion
from flashinfer.comm.trtllm_mnnvl_ar import (
    MNNVLAllReduceFusionWorkspace,
    MNNVLAllreduceFusionStrategy,
    trtllm_mnnvl_fused_allreduce_add_rmsnorm,
    trtllm_mnnvl_fused_allreduce_rmsnorm,
)


def _no_dispatch():
    pytest.fail("The invalid tensor reached workspace access or kernel loading")


def _padded_like(x):
    if x.ndim == 1:
        return torch.zeros((x.shape[0], 2), dtype=x.dtype)[:, 0]
    return torch.zeros((x.shape[0], 2 * x.shape[1]), dtype=x.dtype)[:, : x.shape[1]]


@pytest.mark.parametrize(
    ("field", "provided_outputs"),
    [
        ("input", False),
        ("residual_in", False),
        ("gamma", False),
        ("input", True),
        ("residual_in", True),
        ("gamma", True),
        ("output", True),
        ("residual_out", True),
    ],
)
def test_mnnvl_nonquantized_rmsnorm_requires_contiguous_tensors(
    field, provided_outputs
):
    x = torch.zeros((2, 16), dtype=torch.bfloat16)
    norm_args = {"input": x, "residual_in": torch.zeros_like(x), "gamma": x[0].clone()}
    if provided_outputs:
        norm_args.update(output=torch.empty_like(x), residual_out=torch.empty_like(x))
    norm_args[field] = _padded_like(norm_args[field])
    assert not norm_args[field].is_contiguous()

    workspace = SimpleNamespace(_require_handles_attached=_no_dispatch)
    with pytest.raises(ValueError, match=rf"^{field} must be contiguous"):
        trtllm_mnnvl_fused_allreduce_add_rmsnorm(workspace=workspace, **norm_args)


@pytest.mark.parametrize("use_oneshot", [None, True, False])
def test_mnnvl_unified_rmsnorm_rejects_padded_output(monkeypatch, use_oneshot):
    # Exercise workspace-type dispatch without allocating symmetric device memory.
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace._destroyed = True
    monkeypatch.setattr(workspace, "_require_handles_attached", _no_dispatch)
    x = torch.zeros((2, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="^output must be contiguous"):
        allreduce_fusion(
            input=x,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            residual_in=torch.zeros_like(x),
            rms_gamma=x[0].clone(),
            norm_out=_padded_like(x),
            use_oneshot=use_oneshot,
        )


@pytest.mark.parametrize(
    ("pattern", "route"),
    [
        (AllReduceFusionPattern.kAllReduce, "trtllm_mnnvl_allreduce"),
        (
            AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
            "trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant",
        ),
    ],
)
def test_mnnvl_unified_other_patterns_keep_their_dispatch(monkeypatch, pattern, route):
    allreduce_module = import_module("flashinfer.comm.allreduce")
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace._destroyed = True
    x = torch.zeros((2, 16), dtype=torch.bfloat16)
    quant_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    dispatch = Mock(return_value=(quant_out, None, None, None))
    monkeypatch.setattr(allreduce_module, route, dispatch)
    monkeypatch.setattr(
        allreduce_module, "trtllm_mnnvl_fused_allreduce_add_rmsnorm", _no_dispatch
    )

    result = allreduce_fusion(
        input=x,
        workspace=workspace,
        pattern=pattern,
        residual_in=torch.zeros_like(x),
        rms_gamma=x[0].clone(),
        quant_out=quant_out,
    )
    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["input"] is x
    if pattern == AllReduceFusionPattern.kAllReduce:
        assert result is dispatch.call_args.kwargs["output"]
    else:
        assert result is quant_out


@pytest.mark.parametrize("provided_outputs", [False, True])
@pytest.mark.parametrize("strategy", list(MNNVLAllreduceFusionStrategy))
def test_mnnvl_rmsnorm_dispatches_contiguous_tensors(
    monkeypatch, provided_outputs, strategy
):
    # A contiguous view need not start at the beginning of its storage.
    x = torch.zeros((3, 16), dtype=torch.bfloat16)[1:]
    args = {"input": x, "residual_in": torch.zeros_like(x), "gamma": x[0].clone()}
    if provided_outputs:
        args.update(output=torch.empty_like(x), residual_out=torch.empty_like(x))
    module = SimpleNamespace(trtllm_mnnvl_allreduce_fusion=Mock())
    monkeypatch.setattr(trtllm_mnnvl_ar, "get_trtllm_mnnvl_comm_module", lambda: module)
    workspace = SimpleNamespace(
        _require_handles_attached=Mock(),
        is_buffer_size_sufficient=Mock(return_value=True),
        tp_size=2,
        rank=0,
        mc_ptr=1,
        uc_ptrs_dev=2,
        uc_ptr_local=3,
        buffer_flags=None,
    )
    result = trtllm_mnnvl_fused_allreduce_add_rmsnorm(
        workspace=workspace, strategy=strategy, **args
    )
    module.trtllm_mnnvl_allreduce_fusion.assert_called_once()
    launch = module.trtllm_mnnvl_allreduce_fusion.call_args.args
    assert workspace._require_handles_attached.call_count == 1
    assert launch[0] is x
    assert launch[9] is (strategy != MNNVLAllreduceFusionStrategy.TWOSHOT)
    assert launch[10] is result[0] and launch[11] is result[1]
    assert all(t.is_contiguous() and t.shape == x.shape for t in result)
    if provided_outputs:
        assert result[0] is args["output"] and result[1] is args["residual_out"]


@pytest.mark.parametrize(
    "field", ["shard_input", "residual", "gamma", "normed_output", "prenorm_output"]
)
def test_mnnvl_legacy_rmsnorm_rejects_padded_layout(monkeypatch, field):
    monkeypatch.setattr(trtllm_mnnvl_ar, "get_trtllm_mnnvl_comm_module", _no_dispatch)
    x = torch.zeros((2, 16), dtype=torch.bfloat16)
    args = dict(
        shard_input=x,
        residual=torch.zeros_like(x),
        gamma=x[0].clone(),
        normed_output=torch.empty_like(x),
        prenorm_output=torch.empty_like(x),
    )
    args[field] = _padded_like(args[field])
    with (
        pytest.warns(DeprecationWarning, match="is deprecated"),
        pytest.raises(ValueError, match=rf"^{field} must be contiguous"),
    ):
        trtllm_mnnvl_fused_allreduce_rmsnorm(
            multicast_buffer_ptr=0,
            buffer_ptrs_dev=0,
            unicast_ptr=0,
            buffer_M=x.shape[0],
            buffer_flags_mnnvl=torch.empty(0, dtype=torch.int32),
            nranks=2,
            rank=0,
            epsilon=1e-6,
            launch_with_pdl=False,
            **args,
        )
