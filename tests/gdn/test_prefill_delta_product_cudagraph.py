"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CUDA graph capture/replay for chunk_gated_delta_product.

Two failure modes this catches that the eager tests cannot:

  * A host sync inside the wrapper -- ``.item()``, ``.tolist()``, ``int(t)``,
    or any data-dependent control flow on a device tensor. Capture raises.
  * Values baked in at capture time. The expansion derives ``cu_seqlens * n_h``,
    ``g_flat`` and ``q_flat`` from device tensors; if any of that is computed on
    the host, replay silently reuses the captured numbers. So we replay with
    DIFFERENT input contents and check the output tracks them.

Note the graph-safe calling convention: pass ``output=``/``output_state=`` so
results land in caller-owned buffers. Letting the wrapper allocate works under
capture (the allocation joins the graph pool) but the returned tensor is the
same storage on every replay, which is a sharper edge than it looks.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.gdn_product import chunk_gated_delta_product

from .reference_delta_product import delta_product
from .test_prefill_delta_product import _gen_product_inputs, _skip_if_unsupported


@pytest.mark.parametrize(
    "num_householder", [1, 2, 3], ids=lambda nh: f"num_householder={nh}"
)
@pytest.mark.parametrize(
    "num_heads",
    [(8, 8, 8), (8, 8, 16)],
    ids=lambda qkv: "num_heads={0}/{1}/{2}".format(*qkv),
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cudagraph_capture_and_replay(qkv_factory, num_householder, num_heads, dtype):
    _skip_if_unsupported()

    num_q_heads, num_k_heads, num_v_heads = num_heads
    num_o_heads = num_sab_heads = max(num_q_heads, num_v_heads)
    seq_lens = [64, 128, 256]
    head_size, n_h = 128, num_householder
    num_seqs, total_seqlen = len(seq_lens), sum(seq_lens)
    device = torch.device("cuda")
    dtype = getattr(torch, dtype)

    def gen(seed):
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return _gen_product_inputs(
            seq_lens,
            n_h,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            head_size,
            dtype,
            qkv_factory,
            device,
        )

    # Static buffers -- graphs fix both shapes AND addresses, so every replay
    # must read from and write to exactly these tensors.
    q, k, v, alpha, beta, cu_seqlens = gen(0)
    our_o = torch.empty(
        (total_seqlen, num_o_heads, head_size), dtype=dtype, device=device
    )
    our_state = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    def run_op():
        chunk_gated_delta_product(
            q,
            k,
            v,
            alpha,
            beta,
            1.0,  # scale
            None,  # initial_state
            True,  # output_final_state
            cu_seqlens,
            True,  # use_qk_l2norm_in_kernel
            output=our_o,
            output_state=our_state,
        )

    # Warm up on a side stream. This is not politeness: flashinfer JIT-compiles
    # and autotunes on first call for a given shape, and neither may happen
    # inside a capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run_op()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_op()
    torch.cuda.synchronize()

    if dtype == torch.bfloat16:
        atol_o, rtol_o, atol_kv, rtol_kv = 1e-2, 1e-2, 5e-3, 1e-3
    else:
        atol_o, rtol_o, atol_kv, rtol_kv = 2e-3, 1e-3, 1e-3, 1e-4

    # Replay with fresh contents each round. If anything was computed on the
    # host at capture time, these will not track.
    for step in range(2):
        new_q, new_k, new_v, new_alpha, new_beta, _ = gen(1000 + step)
        q.copy_(new_q)
        k.copy_(new_k)
        v.copy_(new_v)
        alpha.copy_(new_alpha)
        beta.copy_(new_beta)

        our_o.fill_(float("nan"))
        our_state.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        assert not our_o.isnan().any(), (
            f"step {step}: output not fully written on replay -- a strided "
            f"copy-back is likely missing rows"
        )
        assert not our_state.isnan().any(), f"step {step}: state not written"

        ref_o, ref_state = delta_product(
            new_q.float(),
            new_k.float(),
            new_v.float(),
            seq_lens,
            alpha=new_alpha,
            beta=new_beta,
            scale_factor=1.0,
        )
        torch.testing.assert_close(
            our_o,
            ref_o.to(dtype),
            atol=atol_o,
            rtol=rtol_o,
            msg=lambda m: f"step {step} output mismatch after replay\n{m}",
        )
        torch.testing.assert_close(
            our_state.transpose(-1, -2),
            ref_state,
            atol=atol_kv,
            rtol=rtol_kv,
            msg=lambda m: f"step {step} state mismatch after replay\n{m}",
        )


@pytest.mark.parametrize(
    "num_householder", [2, 3], ids=lambda nh: f"num_householder={nh}"
)
def test_cudagraph_replay_is_not_frozen(qkv_factory, num_householder):
    """A graph that ignored its inputs would still pass a single-replay check.

    Replay twice with different data and assert the two outputs DIFFER -- this
    is the cheap guard against a capture that merely re-emits stale results.
    """
    _skip_if_unsupported()

    seq_lens = [128]
    head_size, n_h = 128, num_householder
    num_q_heads = num_k_heads = num_v_heads = 8
    total_seqlen = sum(seq_lens)
    device = torch.device("cuda")
    dtype = torch.float16

    def gen(seed):
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return _gen_product_inputs(
            seq_lens,
            n_h,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            head_size,
            dtype,
            qkv_factory,
            device,
        )

    q, k, v, alpha, beta, cu_seqlens = gen(0)
    our_o = torch.empty(
        (total_seqlen, num_v_heads, head_size), dtype=dtype, device=device
    )
    our_state = torch.empty(
        (len(seq_lens), num_v_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    def run_op():
        chunk_gated_delta_product(
            q,
            k,
            v,
            alpha,
            beta,
            1.0,
            None,
            True,
            cu_seqlens,
            True,
            output=our_o,
            output_state=our_state,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run_op()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_op()
    torch.cuda.synchronize()

    outs = []
    for seed in (1000, 2000):
        new_q, new_k, new_v, new_alpha, new_beta, _ = gen(seed)
        q.copy_(new_q)
        k.copy_(new_k)
        v.copy_(new_v)
        alpha.copy_(new_alpha)
        beta.copy_(new_beta)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(our_o.clone())

    assert not torch.allclose(outs[0], outs[1]), (
        "replay produced identical output for different inputs -- the graph is "
        "not reading its input buffers"
    )
