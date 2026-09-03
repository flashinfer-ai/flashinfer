"""CPU gates for the isolated TRT-LLM MoE all-reduce source bundle."""

from __future__ import annotations

import json
import re

from flashinfer.jit import cake_trtllm_moe_allreduce as backend


_FFI_PARAMETER_NAMES = (
    "world_size",
    "world_rank",
    "token_num",
    "hidden_dim",
    "workspace_ptrs",
    "launch_with_pdl",
    "residual_in",
    "rms_gamma",
    "rms_eps",
    "scale_factor",
    "active_experts",
    "expert_scales",
    "active_expert_tokens",
    "token_input",
    "moe_allreduce_out",
    "residual_out",
    "norm_out",
    "weight_bias",
)

_KERNEL_ARGUMENT_NAMES = (
    "p_active",
    "p_scales",
    "p_token",
    "p_residual",
    "p_gamma",
    "p_moe_out",
    "p_residual_out",
    "p_norm_out",
    "p_quant_out",
    "p_scale_out",
    "p_workspace",
    "rank32",
    "tokens32",
    "experts32",
    "eps32",
    "weight_bias32",
    "scale_factor32",
    "unused_layout",
)


def test_source_bundle_has_exact_twelve_symbol_inventory() -> None:
    source_path, source = backend._load_source_bundle()
    manifest = json.loads((source_path.parent / "manifest.json").read_text())
    symbols = tuple(
        match.decode()
        for match in re.findall(
            rb"(?m)^kernel_cake_trtllm_moe_reduction_[A-Za-z0-9_]+(?=\()",
            source,
        )
    )

    assert len(symbols) == 12
    assert symbols == backend._KERNEL_SYMBOLS
    assert manifest["kernel_symbols"] == list(backend._KERNEL_SYMBOLS)
    assert manifest["architectures"] == ["sm_100a", "sm_103a"]
    assert manifest["constraints"]["world_sizes"] == [2, 4, 8]
    assert manifest["constraints"]["max_lamport_comm_size_bytes"] == 2145386496
    assert "max_tokens" not in manifest["constraints"]
    assert not re.search(rb"kernel_cake_trtllm_moe_(?!reduction_)", source)


def test_host_exposes_only_reduction_and_has_exact_18_parameter_ffi() -> None:
    source = backend._HOST_SOURCE
    signature = re.search(r"void RunReduction\((.*?)\) \{", source, re.DOTALL)
    assert signature is not None
    parameter_names = tuple(
        re.findall(
            r"(?:int64_t|bool|double|TensorView|Optional<TensorView>|"
            r"Optional<double>)\s+(\w+)",
            signature.group(1),
        )
    )
    exported_functions = re.findall(
        r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\((\w+)", source
    )

    assert parameter_names == _FFI_PARAMETER_NAMES
    assert len(parameter_names) == 18
    assert exported_functions == ["run_reduction"]
    assert source.count("CAKE_MOE_AR_LAUNCH_CASE(") == 13
    assert "dtype_index * 6 + world_index * 2 + output_index" in source


def test_embedded_kernel_launch_has_exact_18_argument_pointer_table() -> None:
    source = backend._HOST_SOURCE
    function_start = source.index("void RunReduction(")
    args_start = source.index("void* args[] = {", function_start)
    args_end = source.index("};", args_start)
    argument_names = tuple(re.findall(r"&(\w+)", source[args_start:args_end]))

    assert argument_names == _KERNEL_ARGUMENT_NAMES
    assert len(argument_names) == 18
