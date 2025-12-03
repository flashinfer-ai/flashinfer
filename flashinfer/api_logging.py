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
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Optional
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

# Level 10 tensor dumping configuration
_DUMP_DIR = os.environ.get("FLASHINFER_DUMP_DIR", "flashinfer_dumps")
_DUMP_MAX_SIZE_GB = float(os.environ.get("FLASHINFER_DUMP_MAX_SIZE_GB", "20"))
_DUMP_MAX_COUNT = int(os.environ.get("FLASHINFER_DUMP_MAX_COUNT", "1000"))

# Global tracking for dump limits (reset per process)
_dump_count = 0
_dump_total_size_bytes = 0
_dump_call_counter = {}  # Track call count per function

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
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def _serialize_value(value: Any) -> Any:
    """
    Convert a value to a JSON-serializable format for metadata.
    """
    try:
        if isinstance(value, torch.dtype):
            # Special handling for torch.dtype
            return {
                "type": "torch.dtype",
                "value": str(value),  # e.g., "torch.bfloat16"
            }
        elif isinstance(value, enum.Enum):
            return {
                "type": "enum",
                "name": f"{type(value).__name__}.{value.name}",
                "value": value.value,
            }
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple, dict)):
            return {
                "type": type(value).__name__,
                "value": str(value)[:1000],
            }  # Truncate long structures
        else:
            return {
                "type": type(value).__name__,
                "repr": str(value)[:1000],
            }
    except Exception:
        return {
            "type": type(value).__name__,
            "repr": "<not serializable>",
        }


def _extract_tensors_and_metadata(
    args: tuple, kwargs: dict
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Extract tensors and non-tensor metadata from function arguments.

    Tensors are moved to CPU but preserve their stride/contiguity information.

    Returns
    -------
    tensors : Dict[str, torch.Tensor]
        Dictionary of tensor arguments with keys like "arg_0", "kwarg_name"
        All tensors are on CPU with original stride preserved.
    metadata : Dict[str, Any]
        Dictionary of non-tensor arguments (serializable to JSON)
    """
    tensors = {}
    metadata = {}

    # Process positional arguments
    for i, arg in enumerate(args):
        key = f"arg_{i}"
        if isinstance(arg, torch.Tensor):
            # Move to CPU while preserving stride information
            tensors[key] = arg.cpu()
        else:
            metadata[key] = _serialize_value(arg)

    # Process keyword arguments
    for key, value in kwargs.items():
        kwarg_key = f"kwarg_{key}"
        if isinstance(value, torch.Tensor):
            # Move to CPU while preserving stride information
            tensors[kwarg_key] = value.cpu()
        else:
            metadata[kwarg_key] = _serialize_value(value)

    return tensors, metadata


def _dump_function_inputs(
    func: Callable,
    func_name: str,
    args: tuple,
    kwargs: dict,
) -> Optional[str]:
    """
    Dump function inputs to disk BEFORE execution (crash-safe).

    This function:
    1. Extracts tensors and metadata from inputs
    2. Creates a timestamped directory
    3. Saves inputs.safetensors and partial metadata.json
    4. Tracks cumulative size and count limits

    Parameters
    ----------
    func : Callable
        The function being called
    func_name : str
        Name of the function
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments

    Returns
    -------
    Optional[str]
        Path to the dump directory, or None if dump was skipped
    """
    global _dump_count, _dump_total_size_bytes

    # Check count limit
    if _dump_count >= _DUMP_MAX_COUNT:
        _logger.warning(
            f"Dump limit reached ({_DUMP_MAX_COUNT} dumps). Skipping dump for {func_name}. "
            f"Increase FLASHINFER_DUMP_MAX_COUNT if needed."
        )
        return None

    try:
        # Get call counter for this function
        if func_name not in _dump_call_counter:
            _dump_call_counter[func_name] = 0
        _dump_call_counter[func_name] += 1
        call_seq = _dump_call_counter[func_name]

        # Create dump directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Include milliseconds
        dump_name = f"{timestamp}_{func_name}_call{call_seq:04d}"
        dump_dir = Path(_DUMP_DIR) / dump_name
        dump_dir.mkdir(parents=True, exist_ok=True)

        # Extract tensors and metadata from inputs
        input_tensors, input_metadata = _extract_tensors_and_metadata(args, kwargs)

        # Calculate input size
        input_size = sum(_get_tensor_size_bytes(t) for t in input_tensors.values())

        # Check size limit (conservative check - only inputs for now)
        max_size_bytes = _DUMP_MAX_SIZE_GB * 1024 * 1024 * 1024
        if _dump_total_size_bytes + input_size > max_size_bytes:
            _logger.warning(
                f"Dump size limit reached ({_DUMP_MAX_SIZE_GB} GB). Skipping dump for {func_name}. "
                f"Increase FLASHINFER_DUMP_MAX_SIZE_GB if needed."
            )
            # Clean up empty directory
            dump_dir.rmdir()
            return None

        # Save input tensors using torch.save (preserves stride/contiguity)
        if input_tensors:
            torch.save(input_tensors, dump_dir / "inputs.pt")

        # Create partial metadata (inputs only, outputs will be added later)
        metadata = {
            "function_name": func_name,
            "module": func.__module__ if hasattr(func, "__module__") else "<unknown>",
            "call_sequence": call_seq,
            "timestamp": timestamp,
            "process_id": os.getpid(),
            "input_metadata": input_metadata,
            "output_metadata": {},  # Placeholder, will be updated after execution
            "tensor_info": {
                "input_tensor_keys": list(input_tensors.keys()),
                "output_tensor_keys": [],  # Placeholder, will be updated after execution
                "input_size_bytes": input_size,
                "input_size_mb": input_size / (1024 * 1024),
            },
            "function_signature": str(inspect.signature(func))
            if hasattr(inspect, "signature")
            else "<unavailable>",
            "versions": {
                "torch": torch.__version__,
                "python": sys.version,
            },
            "execution_status": "inputs_saved",  # Will be updated to "completed" after outputs
        }

        # Try to get FlashInfer version
        try:
            from .version import __version__ as flashinfer_version

            metadata["versions"]["flashinfer"] = flashinfer_version  # type: ignore[index]
        except Exception:
            metadata["versions"]["flashinfer"] = "<unavailable>"  # type: ignore[index]

        # Save partial metadata
        with open(dump_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update global tracking (only input size for now)
        _dump_count += 1
        _dump_total_size_bytes += input_size

        _logger.debug(
            f"Dumped inputs to: {dump_dir} "
            f"(size: {input_size / (1024 * 1024):.2f} MB, "
            f"total: {_dump_count}/{_DUMP_MAX_COUNT} dumps)"
        )

        return str(dump_dir)

    except Exception as e:
        _logger.error(f"Failed to dump function call {func_name}: {e}")
        import traceback

        _logger.error(traceback.format_exc())
        return None


def _dump_function_outputs(dump_dir: str, result: Any) -> None:
    """
    Add function outputs to an existing dump directory (crash-safe).

    This function is called AFTER successful execution to append outputs
    to the dump that was created before execution.

    Parameters
    ----------
    dump_dir : str
        Path to the dump directory created by _dump_function_inputs
    result : Any
        Function return value
    """
    global _dump_total_size_bytes

    try:
        dump_path = Path(dump_dir)
        if not dump_path.exists():
            _logger.error(f"Dump directory not found: {dump_dir}")
            return

        # Extract tensors and metadata from outputs
        output_tensors = {}
        output_metadata = {}
        if isinstance(result, torch.Tensor):
            output_tensors["result"] = result.cpu()
        elif isinstance(result, tuple):
            for i, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    output_tensors[f"result_{i}"] = item.cpu()
                else:
                    output_metadata[f"result_{i}"] = _serialize_value(item)
        else:
            output_metadata["result"] = _serialize_value(result)

        # Calculate output size
        output_size = sum(_get_tensor_size_bytes(t) for t in output_tensors.values())

        # Save output tensors using torch.save (preserves stride/contiguity)
        if output_tensors:
            torch.save(output_tensors, dump_path / "outputs.pt")

        # Load existing metadata and update it
        metadata_path = dump_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update with output information
            metadata["output_metadata"] = output_metadata
            metadata["tensor_info"]["output_tensor_keys"] = list(output_tensors.keys())
            metadata["tensor_info"]["output_size_bytes"] = output_size
            metadata["tensor_info"]["output_size_mb"] = output_size / (1024 * 1024)
            metadata["tensor_info"]["total_size_bytes"] = (
                metadata["tensor_info"]["input_size_bytes"] + output_size
            )
            metadata["tensor_info"]["total_size_mb"] = metadata["tensor_info"][
                "total_size_bytes"
            ] / (1024 * 1024)
            metadata["execution_status"] = "completed"

            # Save updated metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Update global size tracking
            _dump_total_size_bytes += output_size

            _logger.debug(
                f"Dumped outputs to: {dump_dir} "
                f"(output size: {output_size / (1024 * 1024):.2f} MB, "
                f"total dump size: {metadata['tensor_info']['total_size_mb']:.2f} MB)"
            )
        else:
            _logger.error(f"metadata.json not found in {dump_dir}")

    except Exception as e:
        _logger.error(f"Failed to dump outputs to {dump_dir}: {e}")
        import traceback

        _logger.error(traceback.format_exc())


def _reconstruct_value(value: Any) -> Any:
    """
    Reconstruct special types from metadata format.

    Handles:
    - torch.dtype objects
    - enum.Enum objects (future)
    - Other serialized types
    """
    if isinstance(value, dict):
        value_type = value.get("type")

        if value_type == "torch.dtype":
            # Reconstruct torch.dtype from string
            dtype_str = value.get("value", "")
            # Parse strings like "torch.bfloat16", "torch.float16", etc.
            dtype_name = dtype_str.replace("torch.", "")
            try:
                return getattr(torch, dtype_name)
            except AttributeError:
                _logger.warning(f"Could not reconstruct dtype: {dtype_str}")
                return value

        # For other dict types, return as-is
        return value

    return value


def replay_from_dump(
    dump_dir: str, compare_outputs: bool = False, device: str = "cuda"
) -> Any:
    """
    Replay a function call from a dumped directory.

    This function:
    1. Loads metadata.json to get function info
    2. Loads inputs.safetensors to get input tensors
    3. Moves tensors to specified device (default: cuda)
    4. Reconstructs the function call
    5. Optionally compares with saved outputs

    Parameters
    ----------
    dump_dir : str
        Path to the dump directory
    compare_outputs : bool
        If True, load and compare with saved outputs
    device : str
        Target device for tensors. Options:
        - "cuda" (default): Load to cuda:0
        - "cpu": Load to CPU
        - "cuda:N": Load to specific CUDA device

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'args': Positional arguments (tensors on specified device)
        - 'kwargs': Keyword arguments (tensors on specified device)
        - 'metadata': Full metadata
        If compare_outputs=True, also includes:
        - 'expected_tensors': Expected output tensors
        - 'expected_metadata': Expected output metadata

    Examples
    --------
    >>> # Load to cuda:0 (default)
    >>> data = replay_from_dump("flashinfer_dumps/20251119_150823_mm_fp4_call0001/")
    >>> result = mm_fp4(*data['args'], **data['kwargs'])
    >>>
    >>> # Load to specific device
    >>> data = replay_from_dump("flashinfer_dumps/.../", device="cuda:1")
    >>>
    >>> # Load to CPU
    >>> data = replay_from_dump("flashinfer_dumps/.../", device="cpu")
    """
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    # Load metadata
    metadata_path = dump_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dump_dir}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    func_name = metadata["function_name"]

    # Load input tensors (with stride/contiguity preserved)
    inputs_path = dump_path / "inputs.pt"
    input_tensors = {}
    if inputs_path.exists():
        input_tensors = torch.load(str(inputs_path), map_location="cpu")

    # Move tensors to specified device (preserving stride)
    for key, tensor in input_tensors.items():
        input_tensors[key] = tensor.to(device)

    # Reconstruct args and kwargs
    args = []
    kwargs = {}
    input_metadata = metadata.get("input_metadata", {})

    # Sort keys to reconstruct in order
    tensor_keys = sorted(
        [k for k in input_tensors.keys() if k.startswith("arg_")],
        key=lambda x: int(x.split("_")[1]),
    )

    for key in tensor_keys:
        args.append(input_tensors[key])

    # Add non-tensor positional args
    for key in sorted(
        [k for k in input_metadata.keys() if k.startswith("arg_")],
        key=lambda x: int(x.split("_")[1]),
    ):
        arg_idx = int(key.split("_")[1])
        if arg_idx >= len(args):
            args.append(_reconstruct_value(input_metadata[key]))

    # Add keyword arguments
    for key in input_tensors.keys():
        if key.startswith("kwarg_"):
            kwarg_name = key.replace("kwarg_", "")
            kwargs[kwarg_name] = input_tensors[key]

    for key in input_metadata.keys():
        if key.startswith("kwarg_"):
            kwarg_name = key.replace("kwarg_", "")
            if kwarg_name not in kwargs:  # Don't override tensor kwargs
                kwargs[kwarg_name] = _reconstruct_value(input_metadata[key])

    # Try to import and get the function
    _logger.info(f"Replaying {func_name} from {dump_dir}")
    _logger.info(f"  Args: {len(args)}, Kwargs: {list(kwargs.keys())}")

    # The user needs to import the function themselves
    # We just return the reconstructed inputs and metadata for manual replay
    if not compare_outputs:
        _logger.warning(
            "Automatic function resolution not implemented. "
            "Please manually call: your_function(*args, **kwargs) "
            "where args and kwargs are loaded from the dump."
        )
        return {"args": args, "kwargs": kwargs, "metadata": metadata}
    else:
        # Load expected outputs (with stride/contiguity preserved)
        outputs_path = dump_path / "outputs.pt"
        expected_outputs = {}
        if outputs_path.exists():
            expected_outputs = torch.load(str(outputs_path), map_location="cpu")

            # Move output tensors to specified device (preserving stride)
            for key, tensor in expected_outputs.items():
                expected_outputs[key] = tensor.to(device)

        output_metadata = metadata.get("output_metadata", {})

        _logger.warning(
            "Automatic function resolution and comparison not implemented. "
            "Returning inputs, expected outputs, and metadata for manual replay."
        )

        return {
            "args": args,
            "kwargs": kwargs,
            "expected_tensors": expected_outputs,
            "expected_metadata": output_metadata,
            "metadata": metadata,
        }


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
        - 8: Level 5 logging + automatically enable cuDNN/cuBLAS/cuBLASLt API logging
        - 10: Level 8 logging + dump tensors to disk for reproducibility (preserves stride/contiguity)

    FLASHINFER_LOGDEST : str (default: "stdout")
        - "stdout": Log to standard output
        - "stderr": Log to standard error
        - <path>: Log to specified file path
        - Use %i in path for process ID substitution (e.g., "log_%i.txt" -> "log_12345.txt")

    Level 10 Tensor Dumping (additional variables):
    FLASHINFER_DUMP_DIR : str (default: "flashinfer_dumps")
        - Directory where tensor dumps are saved

    FLASHINFER_DUMP_MAX_SIZE_GB : float (default: 20)
        - Maximum total size of dumps in GB

    FLASHINFER_DUMP_MAX_COUNT : int (default: 1000)
        - Maximum number of function call dumps

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

            # Level 10: Dump inputs BEFORE execution (crash-safe)
            dump_dir = None
            if _API_LOG_LEVEL >= 10:
                try:
                    dump_dir = _dump_function_inputs(f, func_name, args, kwargs)
                    if dump_dir:
                        _logger.debug(f"Inputs dumped to: {dump_dir}")
                except Exception as e:
                    _logger.error(f"[DUMP ERROR (inputs) in {func_name}]: {e}")

            # Log BEFORE execution (crash-safe for all levels!)
            try:
                if _API_LOG_LEVEL == 1:
                    # Level 1: Just log function name before execution (crash-safe)
                    _logger.debug(
                        f"{_get_timestamp()} FlashInfer API Call: {func_name}"
                    )
                elif _API_LOG_LEVEL >= 3:
                    # Level 3+: Log full inputs before execution (crash-safe)
                    # For level 10, we use level 5 logging (includes statistics)
                    effective_level = min(_API_LOG_LEVEL, 5)  # Cap at 5 for logging
                    _log_function_inputs(f, func_name, args, kwargs, effective_level)
            except Exception as e:
                _logger.error(f"[LOGGING ERROR in {func_name} (pre-execution)]: {e}")

            # Call the original function (may crash here with CUDA errors)
            result = f(*args, **kwargs)

            # Log outputs AFTER successful execution (level 3+ only)
            try:
                if _API_LOG_LEVEL >= 3:
                    # Level 3+: Log outputs (inputs were already logged above)
                    effective_level = min(_API_LOG_LEVEL, 5)
                    _log_function_outputs(func_name, result, effective_level)
            except Exception as e:
                _logger.error(f"[LOGGING ERROR in {func_name} (outputs)]: {e}")

            # Level 10: Dump outputs AFTER successful execution (crash-safe)
            if _API_LOG_LEVEL >= 10 and dump_dir:
                try:
                    _dump_function_outputs(dump_dir, result)
                    _logger.info(f"Outputs dumped to: {dump_dir}")
                except Exception as e:
                    _logger.error(f"[DUMP ERROR (outputs) in {func_name}]: {e}")

            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
