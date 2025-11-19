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


def generate_additional_params(
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    is_sm90_template: bool = False,
):
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
        )
    return (additional_params_decl, additional_func_params, additional_params_setter)
