# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host regressions for collective setup validation, without allocating weights."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch


@pytest.fixture
def setup_inputs(monkeypatch):
    """Supply CUDA tensor metadata and stop setup before any device allocation."""
    from flashinfer.experimental.cake_w4a8_megamoe_ep16 import backend

    def tensor(shape, dtype):
        t = MagicMock(spec=torch.Tensor)
        t.shape, t.ndim, t.dtype = shape, len(shape), dtype
        t.device, t.is_cuda = torch.device("cuda", 0), True
        t.is_contiguous.return_value = True
        return t

    weights = backend.CakeW4A8MegaMoeEp16Weights(
        tensor((327680, 1536), torch.uint8),
        tensor((98304, 2560), torch.uint8),
        tensor((768, 10240), torch.uint32),
        tensor((1280, 3072), torch.uint32),
    )
    ids = tensor((16, 8), torch.int64)
    ids.__ge__.return_value = torch.ones((16, 8), dtype=torch.bool)
    ids.__lt__.return_value = torch.ones((16, 8), dtype=torch.bool)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: "nccl")
    prop = SimpleNamespace(major=10, minor=3, multi_processor_count=152)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: prop)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    load = Mock(side_effect=AssertionError("must reject before JIT or allocation"))
    monkeypatch.setattr(backend, "load_module", load)
    return backend, weights, ids, prop, load


@pytest.mark.parametrize(
    "failure,match",
    [
        ("weights_type", "prepared"),
        ("weight_type", "w13 must be a tensor"),
        ("weight_shape", "w2 must have shape"),
        ("weight_cpu", "CUDA storage"),
        ("weight_device", "must be on"),
        ("scale_dtype", "dtype"),
        ("architecture", "SM103a"),
        ("current_device", "current CUDA device"),
        ("ids_type", "topk_ids must be a tensor"),
        ("ids_ndim", "token rows"),
        ("ids_rows", "token rows"),
        ("ids_width", "shape"),
        ("ids_dtype", "dtype"),
        ("ids_cpu", "CUDA storage"),
        ("ids_device", "must be on"),
        ("ids_contiguous", "contiguous"),
    ],
)
def test_local_setup_error_is_shared(setup_inputs, monkeypatch, failure, match):
    """A locally invalid rank still participates before all ranks reject setup."""
    backend, weights, ids, prop, load = setup_inputs
    if failure == "weights_type":
        weights = None
    elif failure == "weight_type":
        weights = replace(weights, w13=None)
    elif failure == "weight_shape":
        weights.w2.shape = (1, 1)
    elif failure == "weight_cpu":
        weights.w13.is_cuda = False
        weights.w13.device = torch.device("cpu")
    elif failure == "weight_device":
        weights.w2.device = torch.device("cuda", 1)
    elif failure == "scale_dtype":
        weights.w13_scale.dtype = torch.float32
    elif failure == "architecture":
        prop.minor = 0
    elif failure == "current_device":
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    elif failure == "ids_type":
        ids = None
    elif failure == "ids_ndim":
        ids.ndim = 1
    elif failure == "ids_rows":
        ids.shape = (385, 8)
    elif failure == "ids_width":
        ids.shape = (16, 7)
    elif failure == "ids_dtype":
        ids.dtype = torch.int32
    elif failure == "ids_cpu":
        ids.is_cuda = False
    elif failure == "ids_device":
        ids.device = torch.device("cuda", 1)
    elif failure == "ids_contiguous":
        ids.is_contiguous.return_value = False

    group = object()

    def gather(output, local, *, group):
        output[:] = [(16, True, None)] * 16
        output[5] = local

    collective = Mock(side_effect=gather)
    monkeypatch.setattr(backend.dist, "all_gather_object", collective)
    with pytest.raises(ValueError, match=rf"ranks \[\(5, .*{match}"):
        backend.Session(weights, ids, process_group=group)
    collective.assert_called_once()
    assert collective.call_args.kwargs["group"] is group
    load.assert_not_called()


@pytest.mark.parametrize(
    "peer,match",
    [
        ((-1, False, "w13 must be a tensor"), "ranks.*5.*w13"),
        ((17, True, None), "equal token counts"),
        ((16, False, None), "expert IDs"),
    ],
)
def test_valid_rank_rejects_peer_failure(setup_inputs, monkeypatch, peer, match):
    """A valid local input cannot proceed when a peer reports invalid setup."""
    backend, weights, ids, _, load = setup_inputs

    def gather(output, local, *, group):
        assert local == (16, True, None)
        output[:] = [local] * 16
        output[5] = peer

    monkeypatch.setattr(backend.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match=match):
        backend.Session(weights, ids)
    ids.clone.assert_not_called()
    load.assert_not_called()


def test_valid_setup_reaches_routing_clone(setup_inputs, monkeypatch):
    """Successful collective validation preserves the routing setup boundary."""
    backend, weights, ids, _, _ = setup_inputs

    def gather(output, local, *, group):
        assert local == (16, True, None)
        output[:] = [local] * 16

    monkeypatch.setattr(backend.dist, "all_gather_object", gather)
    ids.clone.side_effect = RuntimeError("reached routing clone")
    with pytest.raises(RuntimeError, match="reached routing clone"):
        backend.Session(weights, ids)
    ids.clone.assert_called_once_with()
