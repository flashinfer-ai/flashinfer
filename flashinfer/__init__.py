"""
Copyright (c) 2023 by FlashInfer team.

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

# Load environment variables from flashinfer.json if it exists
import json
import os
import pathlib


def _load_env_from_json():
    """Load environment variables from flashinfer.json file."""
    # Check for cached config first
    cached_config_path = pathlib.Path.home() / ".config" / "flashinfer.json"

    if cached_config_path.exists():
        try:
            with open(cached_config_path, "r") as f:
                config = json.load(f)

            # Load from cached config
            _load_from_config(config)
            return
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load cached config from {cached_config_path}: {e}")

    # Try to find flashinfer.json in standard locations
    config_paths = [
        pathlib.Path.cwd() / "flashinfer.json",  # Current working directory
        pathlib.Path(__file__).parent.parent / "flashinfer.json",  # Package root
        pathlib.Path.home() / ".flashinfer" / "flashinfer.json",  # User home directory
        pathlib.Path("/etc/flashinfer/flashinfer.json"),  # System-wide configuration
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Load and process the configuration
                _load_from_config(config)

                # Save the filled configuration to cache
                _save_filled_config(config, cached_config_path)
                break
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load flashinfer.json from {config_path}: {e}")


def _load_from_config(config):
    """Load environment variables from a config dictionary."""
    if "environment_variables" in config:
        env_vars = config["environment_variables"]
        # Check if it's categorized (nested) or flat structure
        for key, value in env_vars.items():
            if isinstance(value, dict) and "default" in value:
                # Direct variable definition (flat structure)
                _set_env_var(key, value)
            elif isinstance(value, dict):
                # Category containing variables (nested structure)
                for var_name, var_info in value.items():
                    _set_env_var(var_name, var_info)


def _save_filled_config(config, cached_config_path):
    """Save the configuration with all environment variables filled with actual values."""
    try:
        # Create a deep copy to avoid modifying the original
        filled_config = json.loads(json.dumps(config))

        # Update all variables with their actual values from environment
        if "environment_variables" in filled_config:
            env_vars = filled_config["environment_variables"]
            for key, value in env_vars.items():
                if isinstance(value, dict) and "default" in value:
                    # Direct variable definition (flat structure)
                    if key in os.environ:
                        value["default"] = os.environ[key]
                        value["resolved"] = True
                elif isinstance(value, dict):
                    # Category containing variables (nested structure)
                    for var_name, var_info in value.items():
                        if var_name in os.environ:
                            var_info["default"] = os.environ[var_name]
                            var_info["resolved"] = True

        # Ensure the .config directory exists
        cached_config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the filled configuration
        with open(cached_config_path, "w") as f:
            json.dump(filled_config, f, indent=2)

    except Exception as e:
        print(f"Warning: Could not save filled config to {cached_config_path}: {e}")


def _set_env_var(var_name, var_info):
    """Set a single environment variable if not already set."""
    # Only set if not already set in environment
    if var_name not in os.environ:
        default_value = var_info.get("default")
        if default_value is not None:
            # Handle special cases
            if isinstance(default_value, str):
                if default_value == "~":
                    default_value = str(pathlib.Path.home())
                elif "$cuda_home" in default_value:
                    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
                    default_value = default_value.replace("$cuda_home", cuda_home)

            os.environ[var_name] = str(default_value)


# Load environment variables from flashinfer.json on import
_load_env_from_json()

try:
    from ._build_meta import __version__ as __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"

from . import jit as jit
from .activation import gelu_and_mul as gelu_and_mul
from .activation import gelu_tanh_and_mul as gelu_tanh_and_mul
from .activation import silu_and_mul as silu_and_mul
from .attention import BatchAttention as BatchAttention
from .autotuner import autotune as autotune
from .cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
)
from .cascade import merge_state as merge_state
from .cascade import merge_state_in_place as merge_state_in_place
from .cascade import merge_states as merge_states
from .decode import (
    BatchDecodeMlaWithPagedKVCacheWrapper as BatchDecodeMlaWithPagedKVCacheWrapper,
)
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper as BatchDecodeWithPagedKVCacheWrapper,
)
from .decode import (
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
)
from .decode import cudnn_batch_decode_with_kv_cache as cudnn_batch_decode_with_kv_cache
from .decode import single_decode_with_kv_cache as single_decode_with_kv_cache
from .fp4_quantization import (
    SfLayout,
    block_scale_interleave,
    nvfp4_block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp4_dequantize_host,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_quantize,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from .fp8_quantization import mxfp8_dequantize_host, mxfp8_quantize
from .fused_moe import (
    RoutingMethodType,
    GatedActType,
    cutlass_fused_moe,
    reorder_rows_for_gated_act_gemm,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
)
from .gemm import SegmentGEMMWrapper as SegmentGEMMWrapper
from .gemm import bmm_fp8 as bmm_fp8
from .gemm import mm_fp4 as mm_fp4
from .mla import BatchMLAPagedAttentionWrapper as BatchMLAPagedAttentionWrapper
from .norm import fused_add_rmsnorm as fused_add_rmsnorm
from .norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
from .norm import gemma_rmsnorm as gemma_rmsnorm
from .norm import rmsnorm as rmsnorm
from .page import append_paged_kv_cache as append_paged_kv_cache
from .page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
from .page import get_batch_indices_positions as get_batch_indices_positions
from .page import get_seq_lens as get_seq_lens
from .pod import PODWithPagedKVCacheWrapper as PODWithPagedKVCacheWrapper
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper as BatchPrefillWithPagedKVCacheWrapper,
)
from .prefill import (
    BatchPrefillWithRaggedKVCacheWrapper as BatchPrefillWithRaggedKVCacheWrapper,
)
from .prefill import single_prefill_with_kv_cache as single_prefill_with_kv_cache
from .prefill import (
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from .quantization import packbits as packbits
from .quantization import segment_packbits as segment_packbits
from .rope import apply_llama31_rope as apply_llama31_rope
from .rope import apply_llama31_rope_inplace as apply_llama31_rope_inplace
from .rope import apply_llama31_rope_pos_ids as apply_llama31_rope_pos_ids
from .rope import (
    apply_llama31_rope_pos_ids_inplace as apply_llama31_rope_pos_ids_inplace,
)
from .rope import apply_rope as apply_rope
from .rope import apply_rope_inplace as apply_rope_inplace
from .rope import apply_rope_pos_ids as apply_rope_pos_ids
from .rope import apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace
from .rope import apply_rope_with_cos_sin_cache as apply_rope_with_cos_sin_cache
from .rope import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace,
)
from .sampling import chain_speculative_sampling as chain_speculative_sampling
from .sampling import min_p_sampling_from_probs as min_p_sampling_from_probs
from .sampling import sampling_from_logits as sampling_from_logits
from .sampling import sampling_from_probs as sampling_from_probs
from .sampling import softmax as softmax
from .sampling import top_k_mask_logits as top_k_mask_logits
from .sampling import top_k_renorm_probs as top_k_renorm_probs
from .sampling import top_k_sampling_from_probs as top_k_sampling_from_probs
from .sampling import (
    top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
)
from .sampling import top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs
from .sampling import top_p_renorm_probs as top_p_renorm_probs
from .sampling import top_p_sampling_from_probs as top_p_sampling_from_probs
from .sparse import BlockSparseAttentionWrapper as BlockSparseAttentionWrapper
from .sparse import (
    VariableBlockSparseAttentionWrapper as VariableBlockSparseAttentionWrapper,
)
from .utils import next_positive_power_of_2 as next_positive_power_of_2
