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

"""Launch path of the generated Cake ``concat_mla_kv_quant_fp8`` programs.

The public dispatcher (:func:`flashinfer.mla_kv_pack.concat_mla_kv_quant_fp8`)
validates the call, plans the head group and the grid, resolves the exact
compute-capability build through :mod:`flashinfer.jit.cake_concat_mla_kv_quant_fp8` and then
calls :func:`launch` with the loaded module and its route record.  Everything
here is allocation-free and CUDA-graph capturable.
"""

from typing import Any, Tuple

import torch
import tvm_ffi


def launch_args(record: dict, values: dict, grid: Tuple[int, int, int]) -> tuple:
    """Positional arguments of the generated binding in its ``arg_plan`` order."""
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    args = []
    for raw_kind, raw_name in record["arg_plan"]:
        kind, name = str(raw_kind), str(raw_name)
        if kind in ("buffer", "parameter"):
            if name not in values:
                raise RuntimeError(
                    f"generated program requires unknown argument {name!r}"
                )
            args.append(values[name])
        elif kind == "grid" and name in grid_values:
            args.append(grid_values[name])
        else:
            # The pack program uses plain global pointers: no TMA descriptors
            # and no descriptor workspace.
            raise RuntimeError(
                f"generated program has unsupported argument {(kind, name)!r}"
            )
    return tuple(args)


def launch(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_group: int,
    warps_per_token: int,
    grid: int,
    module: Any,
    record: dict,
) -> None:
    """One generated kernel launch on the current PyTorch stream."""
    num_tokens, num_heads = int(kv_nope.shape[0]), int(kv_nope.shape[1])
    values = {
        "kv_nope": kv_nope,
        "k_pe": k_pe,
        "key": key,
        "value": value,
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "head_pairs": (num_heads + 1) // 2,
        "warps_per_token": int(warps_per_token),
    }
    args = launch_args(record, values, (int(grid), 1, 1))
    with tvm_ffi.use_torch_stream():
        getattr(module, str(record["ffi_entry"]))(*args)
