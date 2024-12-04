import argparse
import sys
from pathlib import Path
from itertools import product
from typing import List

if __package__:
    from . import generate_batch_paged_decode_inst
    from . import generate_batch_paged_prefill_inst
    from . import generate_batch_ragged_prefill_inst
    from . import generate_dispatch_inc
    from . import generate_single_decode_inst
    from . import generate_single_prefill_inst
else:
    sys.path.append(str(Path(__file__).resolve()))
    import generate_batch_paged_decode_inst
    import generate_batch_paged_prefill_inst
    import generate_batch_ragged_prefill_inst
    import generate_dispatch_inc
    import generate_single_decode_inst
    import generate_single_prefill_inst


def get_instantiation_cu(args: argparse.Namespace) -> List[str]:
    def write_if_different(path: Path, content: str) -> None:
        if path.exists() and path.read_text() == content:
            return
        path.write_text(content)

    path: Path = args.path
    head_dims: List[int] = args.head_dims
    pos_encoding_modes: List[int] = args.pos_encoding_modes
    allow_fp16_qk_reductions: List[int] = args.allow_fp16_qk_reductions
    mask_modes: List[int] = args.mask_modes
    enable_bf16: bool = args.enable_bf16
    enable_fp8: bool = args.enable_fp8

    path.mkdir(parents=True, exist_ok=True)

    # dispatch.inc
    write_if_different(
        path / "dispatch.inc",
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=head_dims,
                pos_encoding_modes=pos_encoding_modes,
                allow_fp16_qk_reductions=allow_fp16_qk_reductions,
                mask_modes=mask_modes,
            )
        ),
    )

    idtypes = ["i32"]
    prefill_dtypes = ["f16"]
    decode_dtypes = ["f16"]
    fp16_dtypes = ["f16"]
    fp8_dtypes = ["e4m3", "e5m2"]
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
        fp16_dtypes.append("bf16")
    if enable_fp8:
        decode_dtypes.extend(fp8_dtypes)

    single_decode_uris = []
    # single decode files
    for head_dim, pos_encoding_mode in product(head_dims, pos_encoding_modes):
        for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
            product(fp16_dtypes, fp8_dtypes)
        ):
            dtype_out = dtype_q
            fname = f"single_decode_head_{head_dim}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}.cu"
            content = generate_single_decode_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                dtype_q,
                dtype_kv,
                dtype_out,
            )
            for use_sliding_window in [True, False]:
                for use_logits_soft_cap in [True, False]:
                    single_decode_uris.append(
                        f"single_decode_with_kv_cache_dtype_q_{dtype_q}_"
                        f"dtype_kv_{dtype_kv}_"
                        f"dtype_o_{dtype_out}_"
                        f"head_dim_{head_dim}_"
                        f"posenc_{pos_encoding_mode}_"
                        f"use_swa_{use_sliding_window}_"
                        f"use_logits_cap_{use_logits_soft_cap}"
                    )
            write_if_different(path / fname, content)

    # batch decode files
    batch_decode_uris = []
    for (
        head_dim,
        pos_encoding_mode,
    ) in product(
        head_dims,
        pos_encoding_modes,
    ):
        for idtype in idtypes:
            for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
                product(fp16_dtypes, fp8_dtypes)
            ):
                dtype_out = dtype_q
                fname = f"batch_paged_decode_head_{head_dim}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}_idtype_{idtype}.cu"
                content = generate_batch_paged_decode_inst.get_cu_file_str(
                    head_dim,
                    pos_encoding_mode,
                    dtype_q,
                    dtype_kv,
                    dtype_out,
                    idtype,
                )
                for use_sliding_window in [True, False]:
                    for use_logits_soft_cap in [True, False]:
                        batch_decode_uris.append(
                            f"batch_decode_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_out}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}"
                        )
                write_if_different(path / fname, content)

    # single prefill files
    single_prefill_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
    ) in product(
        head_dims,
        pos_encoding_modes,
        allow_fp16_qk_reductions,
        mask_modes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"single_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}.cu"
            content = generate_single_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
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
                        single_prefill_uris.append(
                            f"single_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}_"
                            f"f16qk_{bool(allow_fp16_qk_reduction)}"
                        )
            write_if_different(path / fname, content)

    # batch prefill files
    batch_prefill_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in product(
        head_dims,
        pos_encoding_modes,
        allow_fp16_qk_reductions,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"batch_paged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            content = generate_batch_paged_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            fname = f"batch_ragged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            content = generate_batch_ragged_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
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
                        batch_prefill_uris.append(
                            f"batch_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{sliding_window}_"
                            f"use_logits_cap_{logits_soft_cap}_"
                            f"f16qk_{bool(allow_fp16_qk_reduction)}"
                        )

    return (
        single_decode_uris
        + batch_decode_uris
        + single_prefill_uris
        + batch_prefill_uris
    )


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
        "--allow_fp16_qk_reductions",
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
        "--enable_bf16",
        type=lambda x: x if isinstance(x, int) else x.lower() == "true",
        required=True,
        nargs="+",
        help="Enable bf16",
    )
    parser.add_argument(
        "--enable_fp8",
        type=lambda x: x if isinstance(x, int) else x.lower() == "true",
        default=True,
        nargs="+",
        help="Enable fp8",
    )
    args = parser.parse_args()
    get_instantiation_cu(args)