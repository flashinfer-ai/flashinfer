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

"""Run with torchrun on 16 mutually NVLink-accessible SM103a/152-SM GPUs."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@pytest.fixture(scope="module")
def environment():
    if int(os.environ.get("WORLD_SIZE", "1")) != 16 or not torch.cuda.is_available():
        pytest.skip("requires a torchrun job with 16 SM103a GPUs in one NVL72")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (prop.major, prop.minor, prop.multi_processor_count) != (10, 3, 152):
        pytest.skip("requires SM103a with 152 physical SMs")
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=1200),
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )
    import deep_gemm
    from flashinfer.moe_ep.weights import MoEWeightPack
    from flashinfer.moe_ep.cake_w4a8_megamoe_ep16 import (
        preprocess_cake_w4a8_megamoe_ep16_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.weights import (
        preprocess_mega_weights,
    )

    generator = torch.Generator(device="cuda").manual_seed(9137 + dist.get_rank())
    weights = MoEWeightPack(
        torch.randint(
            0,
            256,
            (32, 10240, 1536),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        ),
        torch.randint(
            0,
            256,
            (32, 3072, 2560),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        ),
        torch.full((32, 10240, 96), 119, dtype=torch.uint8, device="cuda"),
        torch.full((32, 3072, 160), 119, dtype=torch.uint8, device="cuda"),
    )
    candidate_weights = preprocess_cake_w4a8_megamoe_ep16_weights(weights)
    native_weights = preprocess_mega_weights(
        weights, hidden_size=3072, intermediate_size=5120
    )
    symm_mem.set_backend("NVSHMEM")
    workspace = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, 512, 384, 8, 3072, 5120
    )
    yield candidate_weights, native_weights, workspace
    torch.cuda.synchronize()
    dist.barrier()
    if owned:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "tokens,hot",
    [
        (16, False),
        (32, False),
        (64, False),
        (16, True),
        (32, True),
        (64, True),
        (0, False),
        (17, True),
        (384, True),
    ],
)
def test_forward_and_replay(environment, tokens, hot):
    import deep_gemm
    from flashinfer.moe_ep.cake_w4a8_megamoe_ep16 import CakeW4A8MegaMoeEp16
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.staging import (
        stage_mega_moe_inputs,
    )

    weights, native, workspace = environment
    generator = torch.Generator(device="cuda").manual_seed(
        3137 + tokens + dist.get_rank()
    )
    ids = torch.randint(
        0, 512, (tokens, 8), dtype=torch.int64, device="cuda", generator=generator
    )
    if hot:
        ids[:, :4] = 0
    if tokens in (17, 384):
        ids.zero_()
    x = torch.randn(
        tokens, 3072, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    rw = torch.randn(
        tokens, 8, dtype=torch.float32, device="cuda", generator=generator
    ).softmax(-1)
    guarded = torch.full((tokens + 2, 3072), -123, dtype=torch.bfloat16, device="cuda")
    out = guarded[1:-1]
    session = CakeW4A8MegaMoeEp16(weights, ids)
    expected = torch.empty_like(x)
    if tokens:
        workspace.topk_idx.fill_(-1)
        stage_mega_moe_inputs(
            x,
            rw,
            ids,
            workspace.x[:tokens],
            workspace.x_sf[:tokens],
            workspace.topk_idx[:tokens],
            workspace.topk_weights[:tokens],
        )
        torch.cuda.synchronize()
        dist.barrier()
        deep_gemm.fp8_fp4_mega_moe(
            expected,
            native[0],
            native[1],
            workspace,
            activation_clamp=None,
            fast_math=True,
        )
    torch.cuda.synchronize()
    dist.barrier()
    saved = [t.clone() for t in (x, ids, rw)]
    outputs = []
    for _ in range(32):
        assert session.forward(x, rw, out=out) is out
        outputs.append(out.clone())
    torch.cuda.synchronize()
    dist.barrier()
    for value in outputs:
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected, atol=0.15, rtol=0.05)
        assert torch.equal(value, outputs[0])
    relative_l2 = (
        outputs[0].float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-6)
    assert relative_l2.item() < 0.02
    assert (guarded[[0, -1]] == -123).all()
    assert all(torch.equal(a, b) for a, b in zip((x, ids, rw), saved, strict=True))
    if tokens:
        with pytest.raises(ValueError, match="overlap"):
            session.forward(x, rw, out=x)
        with pytest.raises(ValueError, match="dtype"):
            session.forward(x.float(), rw, out=out)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), pytest.raises(ValueError, match="stream"):
        session.forward(x, rw, out=out)
