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
from literal_map import kv_layout_literal, pos_encoding_mode_literal, bool_literal


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
    # group sizes
    dispatch_group_sizes_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(_)
            for _ in args.group_sizes
        ]
    )
    dispatch_group_sizes_str = f"""#define _DISPATCH_CASES_group_size(case_var, ...)         \\
{dispatch_group_sizes_entries}
// EOL
"""
    # page sizes
    dispatch_page_sizes_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(_)
            for _ in args.page_sizes
        ]
    )
    dispatch_page_sizes_str = f"""#define _DISPATCH_CASES_page_size(case_var, ...)         \\
{dispatch_page_sizes_entries}
// EOL
"""
    # kv layouts
    dispatch_kv_layouts_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(
                kv_layout_literal[_]
            )
            for _ in args.kv_layouts
        ]
    )
    dispatch_kv_layouts_str = f"""#define _DISPATCH_CASES_kv_layout(case_var, ...)         \\
{dispatch_kv_layouts_entries}
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
    # causal
    dispatch_causal_entries = "\n".join(
        [
            "  _DISPATCH_CASE({}, case_var, __VA_ARGS__) \\".format(bool_literal[_])
            for _ in args.causals
        ]
    )
    dispatch_causal_str = f"""#define _DISPATCH_CASES_causal(case_var, ...)         \\
{dispatch_causal_entries}
// EOL
"""

    return "\n".join(
        [
            dispatch_head_dims_str,
            dispatch_group_sizes_str,
            dispatch_page_sizes_str,
            dispatch_kv_layouts_str,
            dispatch_pos_encoding_modes_str,
            dispatch_allow_fp16_qk_reductions_str,
            dispatch_causal_str,
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
        "--page_sizes",
        type=int,
        required=True,
        nargs="+",
        help="Prefill attention page sizes",
    )
    parser.add_argument(
        "--group_sizes", type=int, required=True, nargs="+", help="Group sizes"
    )
    parser.add_argument(
        "--kv_layouts", type=int, required=True, nargs="+", help="KV layouts"
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
        "--causals",
        type=lambda x: x if isinstance(x, int) else x.lower() == "true",
        required=True,
        nargs="+",
        help="Causals",
    )
    args = parser.parse_args()
    print(args)
    with open(Path(args.path), "w") as f:
        f.write(get_dispatch_inc_str(args))
