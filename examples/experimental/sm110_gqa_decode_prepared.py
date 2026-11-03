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

# Prepared FP16 GQA decode on exact SM110a, CUDA 13 or newer.

import argparse

import torch

from flashinfer import (
    launch_sm110_gqa_decode_prepared,
    prepare_sm110_gqa_decode,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()

    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device="cuda") * 0.25
    kv = (
        torch.randn(
            args.batch, 2, 8, args.capacity, 128, dtype=torch.float16, device="cuda"
        )
        * 0.25
    )
    lengths = torch.full((args.batch,), args.capacity, dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)
    prepared = prepare_sm110_gqa_decode(
        {"Q": q, "KV": kv, "O": output, "sequence_lengths": lengths}
    )

    # Compile, initialize workspace and warm up before capture. The side stream
    # waits for input initialization, and the capture stream waits for warmup.
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.current_stream().wait_stream(warmup)

    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_sm110_gqa_decode_prepared(prepared)
        # Tensor addresses stay fixed, while device length values can change.
        lengths.fill_(max(1, args.capacity - 1))
        graph.replay()
    else:
        launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()
    print(f"output shape={tuple(output.shape)}, dtype={output.dtype}")


if __name__ == "__main__":
    main()
