"""
Copyright (c) 2024 by FlashInfer team.

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

import argparse
from itertools import product
from pathlib import Path
from typing import List

from . import (
    generate_batch_paged_prefill_sm90_inst,
    generate_batch_ragged_prefill_sm90_inst,
    generate_single_prefill_sm90_inst,
)


def get_sm90_instantiation_cu(args: argparse.Namespace) -> List[str]:
    def write_if_different(path: Path, content: str) -> None:
        if path.exists() and path.read_text() == content:
            return
        path.write_text(content)

    path: Path = args.path
    head_dims: List[int] = args.head_dims
    pos_encoding_modes: List[int] = args.pos_encoding_modes
    use_fp16_qk_reductions: List[int] = args.use_fp16_qk_reductions
    mask_modes: List[int] = args.mask_modes
    enable_f16: bool = args.enable_f16
    enable_bf16: bool = args.enable_bf16

    path.mkdir(parents=True, exist_ok=True)

    idtypes = ["i32"]
    prefill_dtypes = []
    decode_dtypes = []
    fp16_dtypes = []
    if enable_f16:
        prefill_dtypes.append("f16")
        decode_dtypes.append("f16")
        fp16_dtypes.append("f16")
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
        fp16_dtypes.append("bf16")

    # single prefill files
    single_prefill_sm90_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        use_fp16_qk_reduction,
        mask_mode,
    ) in product(
        head_dims,
        pos_encoding_modes,
        use_fp16_qk_reductions,
        mask_modes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)):
            fname = f"single_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{use_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_sm90.cu"
            content = generate_single_prefill_sm90_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                use_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
            )
            for use_sliding_window in [True, False]:
                for use_logits_soft_cap in [True, False]:
                    if (
                        mask_mode == 0
                    ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                        single_prefill_sm90_uris.append(
                            f"single_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}_"
                            f"f16qk_{bool(use_fp16_qk_reduction)}_sm90"
                        )
            write_if_different(path / fname, content)

    # batch prefill files
    batch_prefill_sm90_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        use_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in product(
        head_dims,
        pos_encoding_modes,
        use_fp16_qk_reductions,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)):
            fname = f"batch_paged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{use_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}_sm90.cu"
            content = generate_batch_paged_prefill_sm90_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                use_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            fname = f"batch_ragged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{use_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}_sm90.cu"
            content = generate_batch_ragged_prefill_sm90_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                use_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            for sliding_window in [True, False]:
                for logits_soft_cap in [True, False]:
                    if (
                        mask_mode == 0
                    ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                        batch_prefill_sm90_uris.append(
                            f"batch_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{sliding_window}_"
                            f"use_logits_cap_{logits_soft_cap}_"
                            f"f16qk_{bool(use_fp16_qk_reduction)}_sm90"
                        )

    return single_prefill_sm90_uris + batch_prefill_sm90_uris


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate cuda files")
    parser.add_argument(
        "--path", type=Path, required=True, help="Path to the dispatch inc file"
    )
    parser.add_argument(
        "--head_dims", type=int, required=True, nargs="+", help="Head dimensions"
    )
    parser.add_argument(
        "--pos_encoding_modes",
        type=int,
        required=True,
        nargs="+",
        help="Position encoding modes",
    )
    parser.add_argument(
        "--use_fp16_qk_reductions",
        type=lambda x: x if isinstance(x, int) else int(x.lower() == "true"),
        required=True,
        nargs="+",
        help="Allow fp16 qk reductions",
    )
    parser.add_argument(
        "--mask_modes",
        type=int,
        required=True,
        nargs="+",
        help="Mask modes",
    )
    parser.add_argument(
        "--enable_f16",
        type=lambda x: x if isinstance(x, int) else x.lower() == "true",
        required=True,
        nargs="+",
        help="Enable f16",
    )
    parser.add_argument(
        "--enable_bf16",
        type=lambda x: x if isinstance(x, int) else x.lower() == "true",
        required=True,
        nargs="+",
        help="Enable bf16",
    )
    args = parser.parse_args()
    get_sm90_instantiation_cu(args)
