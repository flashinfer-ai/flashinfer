# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrate and inspect the SM120 NVFP4 sparse-MLA phase planner."""

from __future__ import annotations

import argparse
import dataclasses
import json

import torch

from flashinfer.mla._sparse_mla_nvfp4_sm120_plan import (
    _CROSSOVER_PROBED_T,
    calibrate_nvfp4_sparse_mla_sm120,
    plan_nvfp4_sparse_mla_sm120,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--extra-topk", type=int, default=512)
    parser.add_argument("--extra-page-size", type=int, default=2)
    parser.add_argument("--no-topk-length", action="store_true")
    parser.add_argument("--no-extra-topk-length", action="store_true")
    parser.add_argument("--attn-sink", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    has_topk_length = not args.no_topk_length
    has_extra_topk_length = args.extra_topk > 0 and not args.no_extra_topk_length
    report = calibrate_nvfp4_sparse_mla_sm120(
        device,
        num_heads=args.heads,
        topk=args.topk,
        extra_topk=args.extra_topk,
        extra_page_size=(args.extra_page_size if args.extra_topk else 0),
        has_topk_length=has_topk_length,
        has_extra_topk_length=has_extra_topk_length,
        has_attn_sink=args.attn_sink,
        force=args.force,
    )
    plans = {}
    for num_tokens in _CROSSOVER_PROBED_T:
        planned = plan_nvfp4_sparse_mla_sm120(
            num_tokens,
            args.heads,
            args.topk,
            64,
            device,
            extra_topk=args.extra_topk,
            extra_page_size=(args.extra_page_size if args.extra_topk else 0),
            has_topk_length=has_topk_length,
            has_extra_topk_length=has_extra_topk_length,
            has_attn_sink=args.attn_sink,
        )
        plans[num_tokens] = (
            None
            if planned is None
            else {"variant": planned.variant.value, "cpb": planned.cpb}
        )
    payload = dataclasses.asdict(report)
    payload["plans"] = plans
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
