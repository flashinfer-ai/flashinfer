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

import copy
import json
import os
import pathlib
import sys
from typing import Any, Dict, Optional


class EnvConfig:
    """FlashInfer environment variable configuration."""

    def __init__(self):
        self._env_dict = {}

    def __getattr__(self, name: str) -> Any:
        """Enable attribute-style access to environment variables."""
        if name.startswith('_'):
            # Private attributes use normal Python attribute access
            return object.__getattribute__(self, name)
        if name in self._env_dict:
            return self._env_dict[name]
        # Return None for optional variables that weren't set
        return None

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._env_dict.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get an environment variable value with optional default fallback."""
        return self._env_dict.get(key, default)

    def set(self, key: str, value: Any):
        """Set a single environment variable value."""
        self._env_dict[key] = value

    def update(self, config_dict: Dict[str, Any]):
        """Batch update multiple environment variables from a dictionary."""
        self._env_dict.update(config_dict)


# Singleton instance holding all environment variables
_env_config = EnvConfig()


# Default configuration template with all FlashInfer environment variables
DEFAULT_CONFIG = {
    "environment_variables": {
        "workspace": {
            "FLASHINFER_WORKSPACE_BASE": {
                "description": "Base directory for FlashInfer workspace.",
                "default": "~",
                "type": "path"
            }
        },
        "cubin": {
            "FLASHINFER_CUBIN_DIR": {
                "description": "Directory for compiled CUDA binaries. Defaults to <CACHE_DIR>/cubins.",
                "default": None,
                "type": "path"
            },
            "FLASHINFER_CUBIN_CHECKSUM_DISABLED": {
                "description": "Disable checksum verification (\"1\" = disabled, \"0\" = enabled).",
                "default": "0",
                "type": "string_boolean"
            },
            "FLASHINFER_CUBINS_REPOSITORY": {
                "description": "Repository URL for downloading prebuilt cubins.",
                "default": None,
                "type": "url"
            }
        },
        "autotuner": {
            "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE": {
                "description": "Load autotuner config from file (\"1\" = enabled, \"0\" = disabled).",
                "default": "0",
                "type": "string_boolean"
            }
        },
        "logging": {
            "FLASHINFER_JIT_VERBOSE": {
                "description": "Enable verbose JIT compilation logging (\"1\" = enabled, \"0\" = disabled).",
                "default": "0",
                "type": "string_boolean"
            }
        },
        "build": {
            "FLASHINFER_EXTRA_LDFLAGS": {
                "description": "Extra linker flags for compilation (space-separated).",
                "default": None,
                "type": "string"
            },
            "FLASHINFER_BUILDING_DOCS": {
                "description": "Documentation building mode (\"1\" = docs build, \"0\" = normal).",
                "default": "0",
                "type": "string_boolean"
            }
        },
        "nvshmem": {
            "NVSHMEM_INCLUDE_PATH": {
                "description": "Include paths for NVSHMEM headers.",
                "default": None,
                "type": "path_list",
                "separator": ":"
            },
            "NVSHMEM_LIBRARY_PATH": {
                "description": "Library paths for NVSHMEM.",
                "default": None,
                "type": "path_list",
                "separator": ":"
            },
            "NVSHMEM_LDFLAGS": {
                "description": "Additional linker flags for NVSHMEM (space-separated).",
                "default": "",
                "type": "string"
            }
        },
        "toolchain": {
            "CC": {
                "description": "C compiler executable.",
                "default": None,
                "type": "string"
            },
            "CXX": {
                "description": "C++ compiler executable.",
                "default": "c++",
                "type": "string"
            },
            "PYTORCH_NVCC": {
                "description": "Path to NVCC for PyTorch CUDA extensions.",
                "default": "$cuda_home/bin/nvcc",
                "type": "path"
            }
        },
        "cuda": {
            "CUDA_LIB_PATH": {
                "description": "Path to CUDA library directory.",
                "default": "/usr/local/cuda/targets/x86_64-linux/lib/",
                "type": "path"
            },
            "TORCH_CUDA_ARCH_LIST": {
                "description": "CUDA architectures for compilation (e.g., \"7.5,8.0,8.6\").",
                "default": None,
                "type": "string"
            }
        },
        "trtllm": {
            "TRTLLM_FORCE_MNNVL_AR": {
                "description": "Force TensorRT-LLM MNNVL AR mode (\"1\" = enabled, \"0\" = disabled).",
                "default": "0",
                "type": "string_boolean"
            }
        }
    },
    "type_definitions": {
        "string": "UTF-8 string value.",
        "url": "String containing a valid URL.",
        "path": "Filesystem path (expanded for ~ and env vars).",
        "path_list": "List of filesystem paths joined by the specified separator.",
        "string_boolean": "String '0' (false) or '1' (true)."
    }
}


def _extract_env_values(config: Dict) -> Dict[str, Any]:
    """
    Extract environment variable values.
    Environment variables from os.environ take precedence over config file values.
    """
    env_dict = {}

    if "environment_variables" in config:
        env_vars = config["environment_variables"]
        for category in env_vars.values():
            if isinstance(category, dict):
                for var_name, var_info in category.items():
                    if isinstance(var_info, dict):
                        # Environment variable takes precedence
                        value = os.environ.get(var_name)

                        # Fall back to cached or default values
                        if value is None:
                            # Use cached effective_value if available
                            value = var_info.get("effective_value")
                            if value is None:
                                value = var_info.get("default")

                        # Expand any template variables (e.g., ~, $cuda_home)
                        if value is not None:
                            if isinstance(value, str):
                                if value == "~":
                                    value = str(pathlib.Path.home())
                                elif "$cuda_home" in value:
                                    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
                                    value = value.replace("$cuda_home", cuda_home)
                            env_dict[var_name] = value

    return env_dict


def _save_config(config: Dict, env_dict: Dict[str, Any], config_path: pathlib.Path):
    """
    Persist configuration with resolved values for caching.
    This preserves user-edited "default" values and adds a separate "effective_value" field.
    """
    try:
        filled_config = copy.deepcopy(config)

        # Add effective values to each variable's configuration
        if "environment_variables" in filled_config:
            env_vars = filled_config["environment_variables"]
            for category in env_vars.values():
                if isinstance(category, dict):
                    for var_name, var_info in category.items():
                        if isinstance(var_info, dict) and var_name in env_dict:
                            effective_value = env_dict[var_name]
                            # Preserve original default, add effective value separately
                            var_info["effective_value"] = effective_value

        # Persist to disk
        with open(config_path, "w") as f:
            json.dump(filled_config, f, indent=2)

    except (IOError, TypeError, OSError) as e:
        print(f"Warning: Could not save config to {config_path}: {e}", file=sys.stderr)


def initialize_env_config():
    """Load environment variables from ~/.config/flashinfer.json."""
    global _env_config

    cached_config_path = pathlib.Path.home() / ".config" / "flashinfer.json"

    # Initialize with default config on first run
    if not cached_config_path.exists():
        try:
            cached_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_config_path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
        except (IOError, OSError) as e:
            print(f"Warning: Could not create default config at {cached_config_path}: {e}", file=sys.stderr)

    # Attempt to load existing configuration
    config = None
    try:
        with open(cached_config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load config from {cached_config_path}: {e}", file=sys.stderr)
        config = DEFAULT_CONFIG

    if config is None:
        config = DEFAULT_CONFIG

    # Extract and resolve all environment variable values
    env_dict = _extract_env_values(config)

    # Populate the global configuration singleton
    _env_config.update(env_dict)

    if config is not DEFAULT_CONFIG:
        _save_config(config, env_dict, cached_config_path)


# Module-level attribute access
def __getattr__(name: str) -> Any:
    """Enable direct module attribute access to environment variables"""
    return getattr(_env_config, name)
