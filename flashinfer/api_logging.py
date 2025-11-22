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

import enum
import functools
import inspect
import logging
import os
import sys
from typing import Any, Callable
import contextlib
import torch


# Helper function to substitute %i with process ID in file paths
def _substitute_process_id(path: str) -> str:
    """
    Replace %i with the current process ID in a path.

    This is useful for multi-process/multi-GPU environments where each process
    needs its own log file.
    """
    if "%i" in path:
        return path.replace("%i", str(os.getpid()))
    return path


# Read environment variables once at module load time
_API_LOG_LEVEL = int(os.environ.get("FLASHINFER_LOGLEVEL", "0"))
_API_LOG_DEST = _substitute_process_id(os.environ.get("FLASHINFER_LOGDEST", "stdout"))

# Create logger using Python's logging library
_logger = logging.getLogger("flashinfer.api")


def _setup_logger():
    """Set up the logger based on environment variables."""
    if _API_LOG_LEVEL == 0:
        # Completely disable logging for zero overhead
        _logger.addHandler(logging.NullHandler())
        _logger.setLevel(logging.CRITICAL + 1)  # Higher than any level
        return

    # All enabled levels use loggging.DEBUG; verbosity is controlled by FLASHINFER_LOGLEVEL instead
    _logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    _logger.handlers.clear()

    # Create handler based on destination
    if _API_LOG_DEST == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif _API_LOG_DEST == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(_API_LOG_DEST, mode="a")

    # Use a simple formatter (we'll add timestamps manually to key lines)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    _logger.addHandler(handler)
    _logger.propagate = False  # Don't propagate to root logger


# Initialize logger at module load time
_setup_logger()


def _get_timestamp() -> str:
    """Get current timestamp in the format [YYYY-MM-DD HH:MM:SS]."""
    from datetime import datetime

    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _log_system_info():
    """Log system information once at module initialization."""
    if _API_LOG_LEVEL == 0:
        return

    lines = []
    lines.append("=" * 80)
    lines.append(f"{_get_timestamp()} FlashInfer API Logging - System Information")
    lines.append("=" * 80)

    try:
        # FlashInfer version
        try:
            from .version import __version__ as flashinfer_version

            lines.append(f"FlashInfer version: {flashinfer_version}")
        except Exception:
            lines.append("FlashInfer version: <unavailable>")

        # CUDA toolkit version
        cuda_version = torch.version.cuda
        if cuda_version:
            lines.append(f"CUDA toolkit version: {cuda_version}")
        else:
            lines.append("CUDA toolkit version: <unavailable - CPU-only build?>")

        # cuDNN version
        try:
            if torch.backends.cudnn.is_available():
                cudnn_version = torch.backends.cudnn.version()
                if cudnn_version:
                    lines.append(f"cuDNN version: {cudnn_version}")
                else:
                    lines.append("cuDNN version: <unavailable>")
            else:
                lines.append("cuDNN version: <not available>")
        except Exception as e:
            lines.append(f"cuDNN version: <error: {e}>")

        # GPU information (if CUDA is available)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            lines.append(f"Number of GPUs: {device_count}")

            # Log information for each GPU
            for i in range(device_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    capability = torch.cuda.get_device_capability(i)
                    sm_arch = capability[0] * 10 + capability[1]
                    lines.append(f"  GPU {i}: {gpu_name}")
                    lines.append(
                        f"    Compute capability: {capability[0]}.{capability[1]} (SM{sm_arch})"
                    )
                except Exception as e:
                    lines.append(f"  GPU {i}: <error: {e}>")
        else:
            lines.append("CUDA: Not available (CPU-only mode)")

        # PyTorch version
        lines.append(f"PyTorch version: {torch.__version__}")

    except Exception as e:
        lines.append(f"Error gathering system information: {e}")

    lines.append("=" * 80)
    lines.append("")  # Empty line for readability

    _logger.debug("\n".join(lines))


# Log system information once at module load time (if logging is enabled)
_log_system_info()


def _format_value(value: Any, level: int, indent: int = 0) -> str:
    """
    Format a value for logging based on the log level.

    Parameters
    ----------
    value : Any
        The value to format
    level : int
        The logging level (1, 2, or 3)
    indent : int
        The indentation level for nested structures

    Returns
    -------
    str
        Formatted string representation of the value
    """
    indent_str = "  " * indent

    # Handle None
    if value is None:
        return f"{indent_str}None"

    # Handle Enum types
    if isinstance(value, enum.Enum):
        # Show both the name and value of the enum
        return (
            f"{indent_str}{value.__class__.__name__}.{value.name} (value={value.value})"
        )

    # Handle torch.Tensor
    if isinstance(value, torch.Tensor):
        if level == 1:
            return f"{indent_str}Tensor(...)"

        # Level 3+: Show metadata
        lines = [f"{indent_str}Tensor("]
        lines.append(f"{indent_str}  shape={tuple(value.shape)}")
        lines.append(f"{indent_str}  stride={tuple(value.stride())}")
        lines.append(f"{indent_str}  dtype={value.dtype}")
        lines.append(f"{indent_str}  device={value.device}")
        lines.append(f"{indent_str}  requires_grad={value.requires_grad}")
        lines.append(f"{indent_str}  is_contiguous={value.is_contiguous()}")

        # Level 5: Add statistics
        if level >= 5:
            try:
                # Skip statistics if we're in CUDA graph capture mode
                # (operations like .min()/.max()/.mean() cause synchronization issues)
                is_capturing = False
                if value.is_cuda and hasattr(torch.cuda, "is_current_stream_capturing"):
                    with contextlib.suppress(Exception):
                        is_capturing = torch.cuda.is_current_stream_capturing()

                if is_capturing:
                    lines.append(
                        f"{indent_str}  [statistics skipped: CUDA graph capture in progress]"
                    )
                elif value.numel() > 0:
                    # Convert to float for statistics if possible
                    if value.dtype in [
                        torch.float16,
                        torch.float32,
                        torch.float64,
                        torch.bfloat16,
                        torch.float8_e4m3fn,
                        torch.float8_e5m2,
                    ]:
                        val_float = value.float()
                        lines.append(f"{indent_str}  min={val_float.min().item():.6f}")
                        lines.append(f"{indent_str}  max={val_float.max().item():.6f}")
                        lines.append(
                            f"{indent_str}  mean={val_float.mean().item():.6f}"
                        )
                        nan_count = torch.isnan(val_float).sum().item()
                        lines.append(f"{indent_str}  nan_count={nan_count}")
                        inf_count = torch.isinf(val_float).sum().item()
                        lines.append(f"{indent_str}  inf_count={inf_count}")
                    elif value.dtype in [
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    ]:
                        lines.append(f"{indent_str}  min={value.min().item()}")
                        lines.append(f"{indent_str}  max={value.max().item()}")
                        lines.append(
                            f"{indent_str}  mean={value.float().mean().item():.6f}"
                        )
            except Exception as e:
                lines.append(f"{indent_str}  [statistics error: {e}]")

        lines.append(f"{indent_str})")
        return "\n".join(lines)

    # Handle FP4Tensor (custom FlashInfer type)
    if hasattr(value, "__class__") and value.__class__.__name__ == "FP4Tensor":
        if level == 1:
            return f"{indent_str}FP4Tensor(...)"

        lines = [f"{indent_str}FP4Tensor("]
        lines.append(
            f"{indent_str}  data={_format_value(value.data, level, indent + 1)}"
        )
        lines.append(
            f"{indent_str}  scale={_format_value(value.scale, level, indent + 1)}"
        )
        lines.append(f"{indent_str}  scale_start_index={value.scale_start_index}")
        if hasattr(value, "original_shape") and value.original_shape is not None:
            lines.append(f"{indent_str}  original_shape={value.original_shape}")
        lines.append(f"{indent_str})")
        return "\n".join(lines)

    # Handle lists
    if isinstance(value, list):
        if len(value) == 0:
            return f"{indent_str}[]"
        if level == 1:
            return f"{indent_str}[list with {len(value)} items]"

        lines = [f"{indent_str}["]
        for i, item in enumerate(value):
            lines.append(
                f"{indent_str}  [{i}]: {_format_value(item, level, indent + 1)}"
            )
        lines.append(f"{indent_str}]")
        return "\n".join(lines)

    # Handle tuples
    if isinstance(value, tuple):
        if len(value) == 0:
            return f"{indent_str}()"
        if level == 1:
            return f"{indent_str}(tuple with {len(value)} items)"

        lines = [f"{indent_str}("]
        for i, item in enumerate(value):
            lines.append(
                f"{indent_str}  [{i}]: {_format_value(item, level, indent + 1)}"
            )
        lines.append(f"{indent_str})")
        return "\n".join(lines)

    # Handle dictionaries
    if isinstance(value, dict):
        if len(value) == 0:
            return f"{indent_str}{{}}"
        if level == 1:
            return f"{indent_str}{{dict with {len(value)} keys}}"

        lines = [f"{indent_str}{{"]
        for key, val in value.items():
            lines.append(
                f"{indent_str}  {repr(key)}: {_format_value(val, level, indent + 1)}"
            )
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    # Handle numeric types (int, float, bool)
    if isinstance(value, (int, float, bool, complex)):
        return f"{indent_str}{value}"

    # Handle strings
    if isinstance(value, str):
        return f"{indent_str}{repr(value)}"

    # Default: use repr
    try:
        return f"{indent_str}{repr(value)}"
    except Exception:
        return f"{indent_str}<{type(value).__name__} object>"


def _get_default_params(func: Callable, args: tuple, kwargs: dict) -> dict:
    """
    Extract parameters that have default values but were not explicitly provided.

    Parameters
    ----------
    func : Callable
        The function being called
    args : tuple
        Positional arguments that were provided
    kwargs : dict
        Keyword arguments that were provided

    Returns
    -------
    dict
        Dictionary of parameter names to default values for parameters that were not provided
    """
    try:
        sig = inspect.signature(func)
        default_params = {}

        # Determine which parameters were NOT provided
        for i, (param_name, param) in enumerate(sig.parameters.items()):
            # Skip if parameter has no default
            if param.default is inspect.Parameter.empty:
                continue

            # Check if this parameter was provided
            provided = False

            # Check positional args and keyword args
            if i < len(args) or param_name in kwargs:
                provided = True

            # If not provided, record the default value
            if not provided:
                default_params[param_name] = param.default

        return default_params
    except Exception:
        # If we can't inspect the signature, return empty dict
        return {}


def _log_function_inputs(
    func: Callable, func_name: str, args: tuple, kwargs: dict, level: int
) -> None:
    """
    Log function inputs BEFORE execution for crash safety.

    This ensures inputs are captured even if the function crashes with a CUDA error.

    Parameters
    ----------
    func : Callable
        The function being called (needed to extract default parameters)
    func_name : str
        Name of the function being called
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments
    level : int
        Logging level (3 or 5)
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{_get_timestamp()} FlashInfer API Call: {func_name}")
    lines.append("-" * 80)

    # Log explicitly provided inputs
    if args or kwargs:
        # Positional arguments
        if args:
            lines.append("Positional input arguments:")
            for i, arg in enumerate(args):
                lines.append(f"  arg[{i}]:")
                lines.append(_format_value(arg, level, indent=2))

        # Keyword arguments
        if kwargs:
            lines.append("Keyword input arguments:")
            for key, value in kwargs.items():
                lines.append(f"  {key}=")
                lines.append(_format_value(value, level, indent=2))
    else:
        lines.append("(No explicit arguments)")

    # Log default parameters that were not explicitly provided
    default_params = _get_default_params(func, args, kwargs)
    if default_params:
        lines.append("Default parameters (not explicitly provided):")
        for param_name, default_value in default_params.items():
            lines.append(f"  {param_name}= [DEFAULT]")
            lines.append(_format_value(default_value, level, indent=2))

    _logger.debug("\n".join(lines))


def _log_function_outputs(func_name: str, result: Any, level: int) -> None:
    """
    Log function outputs AFTER successful execution.

    Parameters
    ----------
    func_name : str
        Name of the function
    result : Any
        Function return value
    level : int
        Logging level (3 or 5)
    """
    lines = []
    # Log outputs
    lines.append("Output value:")
    lines.append(_format_value(result, level, indent=1))

    lines.append("=" * 80)
    lines.append("")  # Empty line for readability

    _logger.debug("\n".join(lines))


def flashinfer_api(func: Callable = None) -> Callable:
    """
    Decorator to FlashInfer's APIs.

    Currently logs input and output values of the function using Python's logging library.
    This decorator integrates with Python's standard logging infrastructure while
    maintaining zero overhead when disabled (FLASHINFER_LOGLEVEL=0).

    NOTE/TODO: Not all FlashInfer APIs are decorated with this decorator yet. This is a work in progress.

    Environment Variables
    ---------------------
    FLASHINFER_LOGLEVEL : int (default: 0)
        - 0: No logging (zero overhead - decorator returns original function)
        - 1: Log function name only (logged BEFORE execution - crash-safe)
        - 3: Log function name + inputs/outputs with metadata (inputs logged BEFORE execution - crash-safe)
        - 5: Log function name + inputs/outputs with metadata + tensor statistics (inputs logged BEFORE execution - crash-safe)

    FLASHINFER_LOGDEST : str (default: "stdout")
        - "stdout": Log to standard output
        - "stderr": Log to standard error
        - <path>: Log to specified file path
        - Use %i in path for process ID substitution (e.g., "log_%i.txt" -> "log_12345.txt")

    Examples
    --------
    Basic usage:

    >>> @flashinfer_api
    ... def my_function(x, y):
    ...     return x + y

    Notes
    -----
    - Key header lines include a timestamp in the format: [YYYY-MM-DD HH:MM:SS]
      (e.g., "FlashInfer API Call: function_name", "FlashInfer API Logging - System Information")
    - When FLASHINFER_LOGLEVEL=0, the decorator has truly zero overhead
      as it returns the original function unchanged.
    - Function names and inputs are logged BEFORE execution:
      - Level 1: Function name only
      - Levels 3-5: Function name + inputs with metadata
      This means critical debugging information is preserved even if the function
      crashes (e.g., CUDA illegal memory access, out-of-bounds, etc.).
    - Outputs are logged AFTER successful execution for levels 3 and 5.
    - **CUDA Graph Compatibility**: At level 5, tensor statistics (min/max/mean/nan_count)
      are automatically skipped during CUDA graph capture to avoid synchronization issues.
      The message "[statistics skipped: CUDA graph capture in progress]" will be logged.
    - The %i pattern is automatically replaced with the process ID for multi-process environments.
    - The logger does not propagate to the root logger to avoid duplicate logs.
    """
    # If logging is disabled, return original function with zero overhead
    if _API_LOG_LEVEL == 0:
        if func is None:
            return lambda f: f
        return func

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Determine function name (with class name if applicable)
            func_name = f.__name__
            if args and hasattr(args[0], "__class__"):
                try:
                    class_name = args[0].__class__.__name__
                    if "Wrapper" in class_name or class_name in [
                        "BatchMLAPagedAttentionWrapper"
                    ]:
                        func_name = f"{class_name}.{func_name}"
                except Exception:
                    pass

            # Log BEFORE execution (crash-safe for all levels!)
            try:
                if _API_LOG_LEVEL == 1:
                    # Level 1: Just log function name before execution (crash-safe)
                    _logger.debug(
                        f"{_get_timestamp()} FlashInfer API Call: {func_name}"
                    )
                elif _API_LOG_LEVEL >= 3:
                    # Level 3+: Log full inputs before execution (crash-safe)
                    _log_function_inputs(f, func_name, args, kwargs, _API_LOG_LEVEL)
            except Exception as e:
                _logger.error(f"[LOGGING ERROR in {func_name} (pre-execution)]: {e}")

            # Call the original function (may crash here with CUDA errors)
            result = f(*args, **kwargs)

            # Log outputs AFTER successful execution (level 3+ only)
            try:
                if _API_LOG_LEVEL >= 3:
                    # Level 3+: Log outputs (inputs were already logged above)
                    _log_function_outputs(func_name, result, _API_LOG_LEVEL)
            except Exception as e:
                _logger.error(f"[LOGGING ERROR in {func_name} (outputs)]: {e}")

            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
