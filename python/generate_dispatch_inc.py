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
from pathlib import Path
from literal_map import (
    pos_encoding_mode_literal,
    bool_literal,
    mask_mode_literal,
    logits_hook_literal,
)


def get_dispatch_inc_str(args: argparse.Namespace) -> str:
    # head dims
    dispatch_head_dims_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(_)
            for _ in args.head_dims
        ]
    )
    dispatch_head_dims_str = f"""#define _DISPATCH_CASES_head_dim(case_var, ...)         \\
{dispatch_head_dims_entries}
// EOL
"""
    # logits post hooks
    dispatch_logits_post_hooks_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(
                logits_hook_literal[_]
            )
            for _ in args.logits_post_hooks
        ]
    )
    dispatch_logits_post_hooks_str = f"""#define _DISPATCH_CASES_logits_post_hook(case_var, ...)         \\
{dispatch_logits_post_hooks_entries}
// EOL
"""
    # positional encoding modes
    dispatch_pos_encoding_modes_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(
                pos_encoding_mode_literal[_]
            )
            for _ in args.pos_encoding_modes
        ]
    )
    dispatch_pos_encoding_modes_str = f"""#define _DISPATCH_CASES_pos_encoding_mode(case_var, ...)         \\
{dispatch_pos_encoding_modes_entries}
// EOL
"""
    # allow fp16 qk reductions
    dispatch_allow_fp16_qk_reduction_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(bool_literal[_])
            for _ in args.allow_fp16_qk_reductions
        ]
    )
    dispatch_allow_fp16_qk_reductions_str = f"""#define _DISPATCH_CASES_allow_fp16_qk_reduction(case_var, ...)         \\
{dispatch_allow_fp16_qk_reduction_entries}
// EOL
"""
    # mask_mode
    dispatch_mask_mode_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(
                mask_mode_literal[_]
            )
            for _ in args.mask_modes
        ]
    )
    dispatch_mask_mode_str = f"""#define _DISPATCH_CASES_mask_mode(case_var, ...)         \\
{dispatch_mask_mode_entries}
// EOL
"""

    return "\n".join(
        [
            dispatch_head_dims_str,
            dispatch_logits_post_hooks_str,
            dispatch_pos_encoding_modes_str,
            dispatch_allow_fp16_qk_reductions_str,
            dispatch_mask_mode_str,
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate dispatch inc file")
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the dispatch inc file"
    )
    parser.add_argument(
        "--head_dims", type=int, required=True, nargs="+", help="Head dimensions"
    )
    parser.add_argument(
        "--logits_post_hooks",
        type=int,
        required=True,
        nargs="+",
        help="Logit post hooks",
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
        type=lambda x: x if isinstance(x, int) else x.lower() == "true",
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
    args = parser.parse_args()
    print(args)
    with open(Path(args.path), "w") as f:
        f.write(get_dispatch_inc_str(args))
