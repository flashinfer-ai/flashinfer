# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Staging follows its tensors when the caller switches CUDA devices."""

import pytest
import torch

from flashinfer.moe_ep.core import bf16_stage


@pytest.mark.gpu_2
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_staging_uses_tensor_device(ids_dtype):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTeDSL is not available")
    device, other = 0, 1
    previous = torch.cuda.current_device()
    graph = None
    bf16_stage._STAGERS.clear()
    if hasattr(bf16_stage._PREPARED, "stager"):
        del bf16_stage._PREPARED.stager
    try:
        with torch.cuda.device(device):
            x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
            ids = torch.arange(12, device="cuda", dtype=ids_dtype).reshape(3, 4)
            scores = torch.rand(3, 4, device="cuda")
            out = torch.empty(8, 256, device="cuda", dtype=torch.bfloat16)
            out_ids = torch.empty(8, 4, device="cuda", dtype=ids_dtype)
            out_scores = torch.empty(8, 4, device="cuda")

        def stage():
            assert bf16_stage.fused_bf16_stage(x, ids, scores, out, out_ids, out_scores)
            assert torch.cuda.current_device() == other

        def check():
            torch.testing.assert_close(out[:3], x, rtol=0, atol=0)
            torch.testing.assert_close(out_ids[:3], ids, rtol=0, atol=0)
            torch.testing.assert_close(out_scores[:3], scores, rtol=0, atol=0)
            assert (out_ids[3:] == -1).all()

        torch.cuda.set_device(other)
        stage()  # Compile and bind on the tensor device.
        check()
        x.add_(1)
        stage()  # Reuse the prepared binding after the device switch.
        check()
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torch.cuda.set_device(other)
                stage()
                torch.cuda.set_device(device)
            x.mul_(2)
            graph.replay()
            torch.cuda.synchronize()
            check()
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.set_device(previous)
