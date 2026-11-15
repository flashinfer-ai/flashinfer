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
"""

# Run the experimental fused MNNVL all-reduce + mHC post-map / pre-norm partials on 8 ranks of one
# MNNVL domain (SM100 / SM103) and compare it with the stock all-reduce plus a PyTorch epilogue.
# Two 4-GPU GB200 / GB300 trays:
#   torchrun --nnodes 2 --nproc-per-node 4 --node-rank <n> --master-addr <host> \
#       examples/experimental/allreduce_mhc_post_prenorm.py

import os

import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.comm.trtllm_mnnvl_ar import (
    MNNVLAllReduceFusionWorkspace,
    trtllm_mnnvl_allreduce,
)

WORLD_SIZE, HIDDEN, HC, NUM_SPLITS, NUM_TOKENS = 8, 4096, 4, 8, 4
MIX = HC * (2 + HC)


def reference(x, residual, post_layer_mix, comb_res_mix, fn):
    """The epilogue on the reduced ``x``: fp32 post-map, partials accumulated in fp64."""
    new_r = post_layer_mix.unsqueeze(-1) * x.float().unsqueeze(1)
    for k in range(HC):
        new_r = new_r + comb_res_mix[:, k, :, None] * residual[:, k, None, :].float()
    r = new_r.double().view(NUM_TOKENS, HC, NUM_SPLITS, HIDDEN // NUM_SPLITS)
    f = fn.double().view(MIX, HC, NUM_SPLITS, HIDDEN // NUM_SPLITS)
    yp = torch.einsum("mjsh,njsh->smn", r, f).float()
    rp = r.square().sum(dim=(1, 3)).transpose(0, 1).float()
    return yp, rp, new_r.to(torch.bfloat16)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size == WORLD_SIZE, "launch 8 ranks"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=world_size,
    )
    workspace = MNNVLAllReduceFusionWorkspace(
        mapping,
        max_num_tokens=32,
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        comm_backend=TorchDistBackend(),
    )
    # Build the module outside CUDA-graph capture (here: before the first call).
    comm.allreduce_mhc_post_prenorm_prepare()

    # This rank's partial sublayer output, and the replicated mHC state of the boundary.
    g = torch.Generator(device="cuda").manual_seed(rank)
    partial = (0.3 * torch.randn(NUM_TOKENS, HIDDEN, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    g.manual_seed(1234)
    residual = torch.randn(NUM_TOKENS, HC, HIDDEN, device="cuda", generator=g).to(
        torch.bfloat16
    )
    post_layer_mix = 2.0 * torch.rand(NUM_TOKENS, HC, device="cuda", generator=g)
    comb_res_mix = torch.rand(NUM_TOKENS, HC, HC, device="cuda", generator=g)
    comb_res_mix = comb_res_mix / comb_res_mix.sum(dim=-1, keepdim=True)
    fn = (
        torch.randn(MIX, HC * HIDDEN, device="cuda", generator=g) / (HC * HIDDEN) ** 0.5
    )

    fused = comm.allreduce_mhc_post_prenorm_supported(
        workspace, NUM_TOKENS, HIDDEN, HC, torch.bfloat16
    )
    yp, rp, residual_out = comm.allreduce_mhc_post_prenorm(
        partial, residual, post_layer_mix, comb_res_mix, fn, workspace
    )

    x = trtllm_mnnvl_allreduce(partial, workspace, launch_with_pdl=True)
    yp_ref, rp_ref, residual_ref = reference(
        x, residual, post_layer_mix, comb_res_mix, fn
    )
    torch.cuda.synchronize()
    # yp / rp differ from the fp64 reference by fp32 summation order only; residual_out rounds
    # the same fp32 post-map (the kernel's fused multiply-adds round once).
    torch.testing.assert_close(yp, yp_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rp, rp_ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(residual_out, residual_ref, atol=1e-6, rtol=2.0**-7)
    exact = (residual_out == residual_ref).float().mean().item()
    if rank == 0:
        print(
            f"fused kernel: {fused}; yp {tuple(yp.shape)}, rp {tuple(rp.shape)}, "
            f"residual_out {tuple(residual_out.shape)}; matches the reference "
            f"(residual_out bit-exact: {100 * exact:.2f}%)"
        )

    dist.barrier()
    workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
