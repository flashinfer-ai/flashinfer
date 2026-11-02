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
"""Run the experimental full/selective lifecycle and emit its trace fixtures."""

import argparse
import json
from pathlib import Path

import torch

from flashinfer import FP8PagedMQATopKWrapper, fi_trace
from flashinfer.experimental.attn_scores.trace import (
    fp8_paged_mqa_compressed_topk_trace,
    fp8_paged_mqa_selective_topk_trace,
    fp8_paged_mqa_topk_trace,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for template in (
        fp8_paged_mqa_topk_trace,
        fp8_paged_mqa_selective_topk_trace,
        fp8_paged_mqa_compressed_topk_trace,
    ):
        inputs = template.init(
            total_q=16,
            batch_size=1,
            num_heads=8,
            head_dim=128,
            page_size=64,
            max_kv_len=8192,
            top_k=512,
            device="cuda",
        )
        wrapper = FP8PagedMQATopKWrapper(**inputs["constructor"])
        wrapper.plan(**inputs["plan"])
        indices = wrapper.run(**inputs["run"])
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert indices.shape == (16, 512)
        definition = fi_trace(wrapper.run, **inputs["run"])
        print(definition["name"])
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / (definition["name"] + ".json")).write_text(
                json.dumps(definition, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
