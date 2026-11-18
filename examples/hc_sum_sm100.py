# SPDX-License-Identifier: Apache-2.0
"""Standalone HC, including changed-input CUDA Graph replay (SM100/SM103)."""

import torch

from flashinfer.hyperconnection import prepare_hc_sum_weights


@torch.no_grad()
def main():
    torch.manual_seed(0)
    rows, hidden, lowrank = 25, 512, 64
    options = dict(device="cpu", dtype=torch.bfloat16)
    down = torch.randn(lowrank, 4 * hidden, **options) * 0.01
    up = torch.randn(4 * hidden, lowrank, **options) * 0.01
    inject = torch.randn(4, 4 * hidden, **options) * 0.01
    norm = torch.randn(4, hidden, **options) * 0.01
    residual = torch.randn(rows, 4 * hidden, device="cuda", dtype=torch.bfloat16)
    plan = prepare_hc_sum_weights(down, up, inject, norm, device=residual.device)
    workspace = plan.allocate_workspace(rows)
    sums = torch.empty(rows, 4, device="cuda", dtype=torch.float32)

    def run():
        plan.bootstrap(residual, out_sum=sums)
        mixed, _ = plan.mix(residual, sums, workspace)
        # Identity core only for this standalone example.
        return plan.apply(mixed, residual, workspace)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    residual.normal_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(workspace.residual).all()
    assert torch.isfinite(workspace.sum_sq).all()
    print(
        "HC graph replay:",
        tuple(workspace.residual.shape),
        tuple(workspace.sum_sq.shape),
    )


if __name__ == "__main__":
    main()
