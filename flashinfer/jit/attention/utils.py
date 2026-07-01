"""
Copyright (c) 2025 by FlashInfer team.

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

from typing import List


_SF_STRIDE_TENSORS = ("maybe_k_cache_sf", "maybe_v_cache_sf")


def get_sf_stride_tensor_names(additional_tensor_names: List[str]) -> List[str]:
    return [name for name in additional_tensor_names if name in _SF_STRIDE_TENSORS]


def generate_sf_stride_setter_lines(
    sf_stride_vars: List[str], prefix: str = "params."
) -> List[str]:
    lines: List[str] = []
    for var in sf_stride_vars:
        tensor_ref = f"{var}_tensor"
        lines.extend(
            [
                f"if ({var}) {{ const auto& {tensor_ref} = {var}.value(); "
                f"{prefix}{var}_stride_page = {tensor_ref}.ndim() == 4 ? "
                f"static_cast<uint32_t>({tensor_ref}.stride(0)) : 0; "
                f"if (kv_layout == QKVLayout::kNHD) {{ "
                f"{prefix}{var}_stride_n = static_cast<uint32_t>({tensor_ref}.stride({tensor_ref}.ndim() == 4 ? 1 : 0)); "
                f"{prefix}{var}_stride_h = static_cast<uint32_t>({tensor_ref}.stride({tensor_ref}.ndim() == 4 ? 2 : 1)); "
                f"}} else {{ "
                f"{prefix}{var}_stride_n = static_cast<uint32_t>({tensor_ref}.stride({tensor_ref}.ndim() == 4 ? 2 : 1)); "
                f"{prefix}{var}_stride_h = static_cast<uint32_t>({tensor_ref}.stride({tensor_ref}.ndim() == 4 ? 1 : 0)); "
                f"}} }} else {{ {prefix}{var}_stride_page = 0; "
                f"{prefix}{var}_stride_n = 0; {prefix}{var}_stride_h = 0; }}",
            ]
        )
    return lines


def generate_additional_params(
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    is_sm90_template: bool = False,
):
    sf_stride_vars = get_sf_stride_tensor_names(additional_tensor_names)
    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_tensor_dtypes,
                additional_tensor_names,
                strict=True,
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(
                additional_scalar_dtypes, additional_scalar_names, strict=True
            )
        ]
        + [
            f"uint32_t {var}_stride_page;\n"
            f"uint32_t {var}_stride_h;\n"
            f"uint32_t {var}_stride_n;\n"
            for var in sf_stride_vars
        ]
    )
    additional_func_params = "".join(
        [
            (
                f", Optional<ffi::Tensor> {var}"
                if var.startswith("maybe")
                else f", ffi::Tensor {var}"
            )
            for var in additional_tensor_names
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_scalar_dtypes, additional_scalar_names, strict=True
            )
        ]
    )
    if is_sm90_template:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.additional_params.{var} = {var} ? static_cast<{dtype}*>({var}.value().data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.additional_params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(
                    additional_tensor_dtypes, additional_tensor_names, strict=True
                )
            ]
            + [
                f"params.additional_params.{var} = {var};"
                for var in additional_scalar_names
            ]
            + generate_sf_stride_setter_lines(
                sf_stride_vars, prefix="params.additional_params."
            )
        )
    else:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.{var} = {var} ? static_cast<{dtype}*>({var}.value().data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(
                    additional_tensor_dtypes, additional_tensor_names, strict=True
                )
            ]
            + [f"params.{var} = {var};" for var in additional_scalar_names]
            + generate_sf_stride_setter_lines(sf_stride_vars)
        )
    return (additional_params_decl, additional_func_params, additional_params_setter)
