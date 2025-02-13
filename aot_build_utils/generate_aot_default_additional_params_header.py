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


def generate_macro_entry(
    macro_prefix,
    additional_tensor_names,
    additional_tensor_dtypes,
    additional_scalar_names,
    additional_scalar_dtypes,
    is_sm90_template: bool = False,
) -> str:
    # NOTE(Zihao): mostly copy-paste from generate_additional_params in flashinfer.jit.attention.py
    additional_func_params = "".join(
        [
            (
                f", std::optional<at::Tensor> {var}"
                if var.startswith("maybe")
                else f", at::Tensor {var}"
            )
            for var in additional_tensor_names
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(additional_scalar_dtypes, additional_scalar_names)
        ]
    )
    if is_sm90_template:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.additional_params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.additional_params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(additional_tensor_dtypes, additional_tensor_names)
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
                    f"params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(additional_tensor_dtypes, additional_tensor_names)
            ]
            + [f"params.{var} = {var};" for var in additional_scalar_names]
        )
    return f"""#define {macro_prefix}_ADDITIONAL_FUNC_PARAMS {additional_func_params}

#define {macro_prefix}_ADDITIONAL_PARAMS_SETTER {additional_params_setter}

"""


def get_aot_default_additional_params_header_str() -> str:
    ret = ""

    ret += generate_macro_entry(
        "SINGLE_DECODE",
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["float", "float", "float", "float"],  # additional_scalar_dtypes
    )

    ret += generate_macro_entry(
        "SINGLE_PREFILL",
        ["maybe_custom_mask", "maybe_alibi_slopes"],
        ["uint8_t", "float"],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],
        ["float", "float", "float", "float"],
    )

    ret += generate_macro_entry(
        "SINGLE_PREFILL_SM90",
        [],
        [],
        ["logits_soft_cap", "sm_scale"],
        ["float", "float"],
        is_sm90_template=True,
    )

    ret += generate_macro_entry(
        "BATCH_DECODE",
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["float", "float", "float", "float"],  # additional_scalar_dtypes
    )

    ret += generate_macro_entry(
        "BATCH_PREFILL",
        ["maybe_custom_mask", "maybe_mask_indptr", "maybe_alibi_slopes"],
        ["uint8_t", "int32_t", "float"],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],
        ["float", "float", "float", "float"],
    )

    ret += generate_macro_entry(
        "BATCH_PREFILL_SM90",
        [],
        [],
        ["logits_soft_cap", "sm_scale"],
        ["float", "float"],
        is_sm90_template=True,
    )

    return ret
