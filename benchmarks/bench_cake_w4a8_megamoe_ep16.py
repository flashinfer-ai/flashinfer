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

"""EP16 eager full-forward comparison; launch with torchrun on one NVL72."""

import argparse
import json
import math
import os
import statistics
import warnings
from datetime import timedelta
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="one JSON file per rank is saved next to this prefix",
    )
    args = parser.parse_args()
    from cupti import cupti  # noqa: F401

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >=13 is required; timing fallback is not accepted")
    warnings.filterwarnings("error", message=".*Falling back to CUDA.*")
    owned = not dist.is_initialized()
    if owned:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=1200),
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )
    rank = dist.get_rank()
    if dist.get_world_size() != 16:
        raise RuntimeError("requires exactly 16 ranks")
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
    )
    from flashinfer.moe_ep.cake_w4a8_megamoe_ep16 import (
        CakeW4A8MegaMoeEp16,
        preprocess_cake_w4a8_megamoe_ep16_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.fp8_fp4_bf16_deepgemm.weights import (
        _quantize_grouped_fp4,
    )
    from flashinfer.testing.utils import bench_gpu_time_with_cupti

    generator = torch.Generator(device="cuda").manual_seed(2026103401 + rank)
    w13 = torch.randn(
        32, 10240, 3072, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    w13.mul_(3072**-0.5)
    w2 = torch.randn(
        32, 3072, 5120, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    w2.mul_(5120**-0.5)
    q13, s13 = _quantize_grouped_fp4(w13)
    q2, s2 = _quantize_grouped_fp4(w2)
    wp = MoEWeightPack(q13, q2, s13, s2)
    del w13, w2
    weights = preprocess_cake_w4a8_megamoe_ep16_weights(wp)
    rows = []
    for case, (hot, tokens) in enumerate(
        (h, n) for h in (False, True) for n in (16, 32, 64)
    ):
        offset = (16, 32, 64).index(tokens)
        input_seed = 2026090401 + offset
        route_seed = (2026110451 if hot else 2026110401) + offset
        generator = torch.Generator(device="cpu").manual_seed(route_seed + rank)
        if hot:
            ids = torch.cat(
                (
                    torch.randint(128, (tokens, 4), generator=generator),
                    128 + torch.randint(384, (tokens, 4), generator=generator),
                ),
                1,
            )
            for token in range(tokens):
                ids[token] = ids[token, torch.randperm(8, generator=generator)]
        else:
            ids = torch.randint(512, (tokens, 8), generator=generator)
        ids = ids.cuda()
        x = torch.randn(
            tokens,
            3072,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(input_seed + rank),
        )
        rw = torch.randn(
            tokens,
            8,
            dtype=torch.float32,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(route_seed + rank),
        ).softmax(-1)
        out = torch.empty_like(x)
        session = CakeW4A8MegaMoeEp16(weights, ids)
        layer = MoEEpMegaLayer(
            bootstrap=BootstrapConfig(
                world_size=16, rank=rank, device=torch.cuda.current_device()
            ),
            fleet_params=FleetParams(
                num_experts=512, max_tokens_per_rank=tokens, token_hidden_size=3072
            ),
            weights=wp,
            backend=MegaConfig(
                megakernel=Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
                    intermediate_size=5120,
                    top_k=8,
                    activation_clamp=None,
                    fast_math=True,
                ),
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        inputs = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=rw)
        callbacks = {
            "native": partial(layer.forward, inputs),
            "candidate": partial(session.forward, x, rw, out=out),
        }
        expected = callbacks["native"]()
        callbacks["candidate"]()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(out, expected, atol=0.15, rtol=0.05)
        groups = []
        for group in range(3):
            order = (
                ("native", "candidate")
                if (case + group) % 2 == 0
                else ("candidate", "native")
            )
            record = {"order": order}
            for name in order:
                times = bench_gpu_time_with_cupti(
                    callbacks[name],
                    dry_run_time_ms=100,
                    repeat_time_ms=1000,
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                    aggregate_op=max,
                )
                record[name] = {
                    "times_ms": times,
                    "median_ms": statistics.median(times),
                }
            groups.append(record)
        callbacks["candidate"]()
        expected = callbacks["native"]()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(out, expected, atol=0.15, rtol=0.05)
        native = statistics.median(g["native"]["median_ms"] for g in groups)
        candidate = statistics.median(g["candidate"]["median_ms"] for g in groups)
        row = dict(
            global_tokens=tokens * 16,
            routing="hotset50" if hot else "balanced",
            native_ms=native,
            candidate_ms=candidate,
            speedup=native / candidate,
            groups=groups,
        )
        rows.append(row)
        if rank == 0:
            print({k: v for k, v in row.items() if k != "groups"}, flush=True)
        del callbacks, session, layer
    report = dict(
        gpu=torch.cuda.get_device_name(),
        world_size=16,
        mode="eager",
        timing="CUPTI full GPU span, per-iteration max rank",
        rows=rows,
        geomean_speedup=math.exp(statistics.mean(math.log(r["speedup"]) for r in rows)),
    )
    output = args.output.with_name(args.output.name + f"-rank-{rank}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if owned:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
