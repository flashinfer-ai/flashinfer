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

"""Dense D576 guard: run each revision on the same saved cold-L2 graph fixture.

Run with --package-root pointing first to a baseline checkout and then to the
candidate. Reuse --fixture; compare its recorded hashes and output tensors.
No installation changes or mutations of either checkout are needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=12)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root.resolve()))

    import torch
    from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper
    from sparse_mla_bench_utils import ColdL2GraphBenchmark

    if not args.fixture.exists():
        generator = torch.Generator().manual_seed(2026)
        fixtures = []
        for batch in (1, 8):
            for heads in (16, 64, 128):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    fixtures.append(
                        dict(
                            q=torch.randn(batch, 1, heads, 576, generator=generator).to(
                                dtype
                            ),
                            kv=torch.randn(
                                batch * 128, 32, 576, generator=generator
                            ).to(dtype),
                            table=torch.arange(batch * 128, dtype=torch.int32).view(
                                batch, 128
                            ),
                            lengths=4096 - torch.arange(batch, dtype=torch.int32),
                        )
                    )
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixtures, args.fixture)
    fixtures = torch.load(args.fixture, weights_only=True)
    properties = torch.cuda.get_device_properties(0)
    report = dict(
        revision=subprocess.check_output(
            ["git", "-C", str(args.package_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        gpu=properties.name,
        protocol="identical-input/cold-L2/CUDA-Graph",
        eviction_method="read_only_reduction",
        eviction_bytes=4 * properties.L2_cache_size,
        samples_per_replay=4,
        replays=args.replays,
        cases=[],
    )
    outputs = []
    for fixture in fixtures:
        digest = hashlib.sha256()
        for name, tensor in fixture.items():
            digest.update(f"{name}:{tensor.dtype}:{tensor.shape}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        q, kv, table, lengths = (v.cuda() for v in fixture.values())
        batch, _, heads, _ = q.shape
        wrapper = BatchMLADecodePagedTSWrapper()
        wrapper.plan(
            q.device,
            batch,
            heads,
            512,
            64,
            32,
            4096,
            max_seq_len_q=1,
            packed_query=False,
            q_data_type=q.dtype,
            kv_data_type=q.dtype,
            o_data_type=torch.bfloat16,
        )
        out = torch.empty(batch, 1, heads, 512, device=q.device, dtype=torch.bfloat16)
        wrapper.run(q, kv, table, lengths, bmm1_scale=576**-0.5, out=out)
        runner = ColdL2GraphBenchmark(
            lambda: wrapper.run(
                q,
                kv,
                table,
                lengths,
                bmm1_scale=576**-0.5,
                out=out,
                validate=False,
            ),
            device=q.device,
            samples_per_replay=4,
        )
        runner.sample()
        times = [t for _ in range(args.replays) for t in runner.sample()]
        outputs.append(out.cpu())
        case = dict(
            batch=batch,
            heads=heads,
            dtype=str(q.dtype),
            fixture_id=digest.hexdigest(),
            median_us=statistics.median(times),
            times_us=times,
        )
        report["cases"].append(case)
        print(batch, heads, q.dtype, case["median_us"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(outputs, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
