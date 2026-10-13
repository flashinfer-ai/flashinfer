# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""RDMA tracing must not fabricate a single-rank collective replay."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import RdmaMoeAlltoAll


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("method", [RdmaMoeAlltoAll.dispatch, RdmaMoeAlltoAll.combine])
def test_distributed_replay_is_explicitly_unsupported(tmp_path, method, world_size):
    # No communicator or GPU tensor is needed to report unsupported replay.
    # These sentinels also prevent accidental inspection of collective inputs.
    kwargs = {"self": SimpleNamespace(num_ranks=world_size), "x": object()}
    if method is RdmaMoeAlltoAll.dispatch:
        kwargs["topk_idx"] = object()
    else:
        kwargs["handle"] = object()
    was_initialized = torch.cuda.is_initialized()

    assert method.fi_trace(save_dir=tmp_path, **kwargs) == {}
    assert list(tmp_path.iterdir()) == []
    assert torch.cuda.is_initialized() == was_initialized
