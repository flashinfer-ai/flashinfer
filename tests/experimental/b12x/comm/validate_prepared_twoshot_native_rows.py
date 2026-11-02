#!/usr/bin/env python3
"""Check prepared BF16 all-reduce shape invariance and mutated graph replay.

Run with torchrun on four visible GPUs. Each rank contributes distinct integer
values exactly representable in BF16; every output must equal the FP32 sum.
"""

import os

import torch
import torch.distributed as dist

from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie import _twoshot_preparation as preparation
from b12x.preparation import PreparationSession
from b12x.preparation.types import require_prepared


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(backend="gloo")
    runtime = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_rows=32,
        row_elems=4096,
    )
    try:
        for shape in ((128, 1024), (32, 4096), (16, 8192), (131072,)):
            source = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)
            output = torch.full_like(source, float("nan"))
            query = preparation.query_from_runtime(
                runtime,
                surface="PCIeTwoShotBF16.all_reduce",
                call={"inp": source, "out": output, "threads": 512, "block_limit": 64},
            )
            plan = preparation.plan(query, runtime=runtime)
            request = plan.request(
                name="bf16_shape",
                prepare_call=lambda state: preparation.prepared_call(
                    state, payload=source, out=output
                ),
            )
            with PreparationSession(
                device=device, autotune=False, compile_workers=1
            ) as session:
                session.prepare((request,))
                state = require_prepared(plan, "comm.pcie", device)
                # Exercise the prepared entrypoint used by startup priming,
                # not the public method that first reshapes its input.
                state.run(source, None, output, threads=512, block_limit=64)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(output, 10), rtol=0, atol=0
                )
                graph = torch.cuda.CUDAGraph()
                with (
                    session.capture(),
                    runtime.capture(plan=plan),
                    torch.cuda.graph(graph),
                ):
                    state.run(source, None, output, threads=512, block_limit=64)
                torch.cuda.synchronize()
                for step in range(1, 4):
                    source.fill_(rank + 1 + step)
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        output, torch.full_like(output, 10 + 4 * step), rtol=0, atol=0
                    )
                graph.reset()
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS shape={shape}: eager and three mutated graph replays",
                    flush=True,
                )
    finally:
        dist.barrier()
        runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
