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
import fnmatch
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
import importlib
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

# Configuration for Level 10 tensor dumping
_DUMP_DIR = os.environ.get("FLASHINFER_DUMP_DIR", "flashinfer_dumps")
_DUMP_MAX_SIZE_GB = float(os.environ.get("FLASHINFER_DUMP_MAX_SIZE_GB", "20"))
_DUMP_MAX_COUNT = int(os.environ.get("FLASHINFER_DUMP_MAX_COUNT", "1000"))

# Dump filtering: include/exclude patterns (fnmatch-style, comma-separated)
# Examples: "*decode*,*prefill*" or "BatchDecodeWrapper.run,mm_fp8"
_DUMP_INCLUDE = os.environ.get("FLASHINFER_DUMP_INCLUDE", "")
_DUMP_EXCLUDE = os.environ.get("FLASHINFER_DUMP_EXCLUDE", "")
_DUMP_INCLUDE_PATTERNS = [p.strip() for p in _DUMP_INCLUDE.split(",") if p.strip()]
_DUMP_EXCLUDE_PATTERNS = [p.strip() for p in _DUMP_EXCLUDE.split(",") if p.strip()]

# SafeTensors format option (default: use torch.save which preserves stride/contiguity)
_DUMP_SAFETENSORS = os.environ.get("FLASHINFER_DUMP_SAFETENSORS", "0") == "1"

# Global tracking for dump limits (reset per process)
_dump_count = 0
_dump_total_size_bytes = 0
_dump_call_counter = {}  # Track call count per function
_session_jsonl_initialized = False  # Track if session.jsonl header was written

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


def _warn_dump():
    """Warn users about security implications of Level 10 logging."""
    if _API_LOG_LEVEL >= 10:
        print("=" * 80)
        print(
            "WARNING: FlashInfer API Logging is set to Level 10 (Tensor Dumping).\n"
            "This will dump ALL input and outputs including tensors for FlashInfer APIs to disk in\n"
            "the configured dump directory. Ensure that you are NOT processing sensitive data\n"
            "or that the dump directory is secure. To disable dumping, unset FLASHINFER_LOGLEVEL or\n"
            "set it to below 10. For more information, see https://docs.flashinfer.ai/logging.html"
        )
        print(f"Current dump directory is: {_DUMP_DIR}")
        if _DUMP_SAFETENSORS:
            print(
                "⚠️  SAFETENSORS mode enabled: tensor stride/non-contiguity will NOT be preserved.\n"
                "    Tensors will be saved as contiguous. Use torch.save (default) to preserve strides."
            )
        if _DUMP_INCLUDE_PATTERNS:
            print(f"Include filter: {_DUMP_INCLUDE_PATTERNS}")
        if _DUMP_EXCLUDE_PATTERNS:
            print(f"Exclude filter: {_DUMP_EXCLUDE_PATTERNS}")
        print("=" * 80)


def _should_dump_function(func_name: str) -> bool:
    """
    Check if a function should be dumped based on include/exclude filters.

    Uses fnmatch-style patterns (wildcards: * for any chars, ? for single char).
    Matching is case-sensitive.

    Parameters
    ----------
    func_name : str
        The function name to check. For class methods, this is formatted as
        "ClassName.method_name" (e.g., "BatchDecodeWrapper.run").

    Returns
    -------
    bool
        True if the function should be dumped, False otherwise.

    Filter Logic
    ------------
    1. If FLASHINFER_DUMP_INCLUDE is set:
       - Function must match at least one include pattern
       - If it doesn't match any, return False (skip dump)
    2. If FLASHINFER_DUMP_EXCLUDE is set:
       - If function matches any exclude pattern, return False (skip dump)
    3. Otherwise, return True (dump the function)
    """
    # If include patterns are specified, func must match at least one
    if _DUMP_INCLUDE_PATTERNS:
        if not any(fnmatch.fnmatch(func_name, pat) for pat in _DUMP_INCLUDE_PATTERNS):
            return False

    # If exclude patterns are specified, func must not match any
    if _DUMP_EXCLUDE_PATTERNS:
        if any(fnmatch.fnmatch(func_name, pat) for pat in _DUMP_EXCLUDE_PATTERNS):
            return False

    return True


def _append_to_jsonl(filepath: Path, record: Dict[str, Any]) -> None:
    """
    Append a JSON record as a single line to a JSONL file.

    Parameters
    ----------
    filepath : Path
        Path to the JSONL file
    record : Dict[str, Any]
        Record to append (will be serialized as single-line JSON)
    """
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl_last_record(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Read the last record from a JSONL file.

    For metadata.jsonl, this returns the most complete state (completed if available,
    otherwise inputs_saved).

    Parameters
    ----------
    filepath : Path
        Path to the JSONL file

    Returns
    -------
    Optional[Dict[str, Any]]
        The last record, or None if file is empty/doesn't exist
    """
    if not filepath.exists():
        return None

    last_line = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if last_line:
        return json.loads(last_line)
    return None


def _get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def _serialize_value(value: Any) -> Any:
    """
    Convert a non-tensor value to a JSON-serializable format for metadata.

    This function is intended for serializing non-tensor arguments/values
    that are used in API input or output metadata. Tensor arguments are not handled here.
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
            tensors[key] = arg.cpu()
        else:
            metadata[key] = _serialize_value(arg)

    # Process keyword arguments
    for key, value in kwargs.items():
        kwarg_key = f"kwarg_{key}"
        if isinstance(value, torch.Tensor):
            tensors[kwarg_key] = value.cpu()
        else:
            metadata[kwarg_key] = _serialize_value(value)

    return tensors, metadata


def _dump_function_inputs(
    func: Callable,
    func_name: str,
    args: tuple,
    kwargs: dict,
    self_id: Optional[int] = None,
) -> Optional[str]:
    """
    Dump function inputs to disk BEFORE execution (crash-safe).

    This function:
    1. Extracts tensors and metadata from inputs
    2. Creates a timestamped directory
    3. Saves inputs.pt and partial metadata.json
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
    self_id : Optional[int]
        The id() of the 'self' object if this is a method call

    Returns
    -------
    Optional[str]
        Path to the dump directory, or None if dump was skipped
    """
    global _dump_count, _dump_total_size_bytes

    # Check include/exclude filters first (before any work is done)
    if not _should_dump_function(func_name):
        _logger.debug(
            f"Skipping dump for {func_name} (filtered by include/exclude patterns)"
        )
        return None

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
        pid = os.getpid()
        dump_name = f"{timestamp}_pid{pid}_{func_name}_call{call_seq:04d}"
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

        # Save input tensors
        if input_tensors:
            if _DUMP_SAFETENSORS:
                # SafeTensors format: faster, no pickle, but loses stride/contiguity
                try:
                    from safetensors.torch import save_file

                    # safetensors requires contiguous tensors
                    tensors_contiguous = {
                        k: v.contiguous() for k, v in input_tensors.items()
                    }
                    save_file(tensors_contiguous, str(dump_dir / "inputs.safetensors"))
                except ImportError:
                    _logger.error(
                        "safetensors package not installed. "
                        "Install with: pip install safetensors"
                    )
                    raise
            else:
                # torch.save format: preserves stride/contiguity
                torch.save(input_tensors, dump_dir / "inputs.pt")

        # Create partial metadata (inputs only, outputs will be added later)
        metadata: Dict[str, Any] = {
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
            "tensor_details": {},  # Detailed shape/dtype/stride info for reconstruction
            "tensor_format": "safetensors" if _DUMP_SAFETENSORS else "torch",
            "function_signature": str(inspect.signature(func))
            if hasattr(inspect, "signature")
            else "<unavailable>",
            "versions": {
                "torch": torch.__version__,
                "python": sys.version,
            },
            "execution_status": "inputs_saved",  # Will be updated to "completed" after outputs
        }

        # Add self_id to metadata if it is a class method call
        if self_id is not None:
            metadata["self_id"] = self_id

        # Add tensor details for random generation fallback
        for key, tensor in input_tensors.items():
            metadata["tensor_details"][key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()),
                "device": str(tensor.device),
            }

        # Try to get FlashInfer version
        try:
            from .version import __version__ as flashinfer_version

            metadata["versions"]["flashinfer"] = flashinfer_version  # type: ignore[index]
        except Exception:
            metadata["versions"]["flashinfer"] = "<unavailable>"  # type: ignore[index]

        # Add dump_dir to metadata for central session.jsonl reference
        metadata["dump_dir"] = str(dump_dir)

        # Save metadata to per-dump JSONL (first line: inputs_saved)
        _append_to_jsonl(dump_dir / "metadata.jsonl", metadata)

        # Append to central session.jsonl for quick scanning
        session_jsonl_path = Path(_DUMP_DIR) / "session.jsonl"
        _append_to_jsonl(session_jsonl_path, metadata)

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

        # Save output tensors
        if output_tensors:
            if _DUMP_SAFETENSORS:
                # SafeTensors format: faster, no pickle, but loses stride/contiguity
                from safetensors.torch import save_file

                tensors_contiguous = {
                    k: v.contiguous() for k, v in output_tensors.items()
                }
                save_file(tensors_contiguous, str(dump_path / "outputs.safetensors"))
            else:
                # torch.save format: preserves stride/contiguity
                torch.save(output_tensors, dump_path / "outputs.pt")

        # Load existing metadata from JSONL (last record) and update it
        metadata_jsonl_path = dump_path / "metadata.jsonl"
        metadata = _read_jsonl_last_record(metadata_jsonl_path)

        if metadata is not None:
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

            # Add output tensor details
            if "tensor_details" not in metadata:
                metadata["tensor_details"] = {}
            for key, tensor in output_tensors.items():
                metadata["tensor_details"][key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "stride": list(tensor.stride()),
                    "device": str(tensor.device),
                }

            # Append completion record to per-dump JSONL
            _append_to_jsonl(metadata_jsonl_path, metadata)

            # Append completion record to central session.jsonl
            session_jsonl_path = Path(_DUMP_DIR) / "session.jsonl"
            _append_to_jsonl(session_jsonl_path, metadata)

            # Update global size tracking
            _dump_total_size_bytes += output_size

            _logger.debug(
                f"Dumped outputs to: {dump_dir} "
                f"(output size: {output_size / (1024 * 1024):.2f} MB, "
                f"total dump size: {metadata['tensor_info']['total_size_mb']:.2f} MB)"
            )
        else:
            _logger.error(f"metadata.jsonl not found or empty in {dump_dir}")

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


def _resolve_function(module_name: str, function_name: str) -> Optional[Callable]:
    """Resolve a function from module name and function name."""
    try:
        module = importlib.import_module(module_name)
        # Handle nested function names (e.g. Class.method)
        parts = function_name.split(".")
        obj: Any = module
        for part in parts:
            obj = getattr(obj, part)
        if not callable(obj):
            return None
        return obj
    except Exception as e:
        _logger.warning(
            f"Could not resolve function {module_name}.{function_name}: {e}"
        )
        return None


def _compare_results(
    actual: Any, expected: Any, rtol: float = 1e-3, atol: float = 1e-3
) -> bool:
    """Recursively compare execution results."""
    # torch.Tensor comparison
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        # Check shape
        if actual.shape != expected.shape:
            _logger.warning(
                f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
            )
            return False
        # Check dtype
        if actual.dtype != expected.dtype:
            _logger.warning(
                f"Dtype mismatch: actual {actual.dtype} vs expected {expected.dtype}"
            )
            return False

        # Check values; apply relative and absolute tolerance.
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            diff = (actual - expected).abs().max().item()
            _logger.warning(f"Value mismatch: max diff {diff}")
            return False
        return True

    # list/tuple comparison
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            _logger.warning(
                f"Length mismatch: actual {len(actual)} vs expected {len(expected)}"
            )
            return False
        return all(
            _compare_results(a, e, rtol, atol)
            for a, e in zip(actual, expected, strict=True)
        )

    # dict comparison
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            _logger.warning(
                f"Key mismatch: actual {actual.keys()} vs expected {expected.keys()}"
            )
            return False
        return all(_compare_results(actual[k], expected[k], rtol, atol) for k in actual)

    # fallback for other types (including None). Just do a naive comparison.
    else:
        if actual != expected:
            _logger.warning(f"Value mismatch: actual {actual} vs expected {expected}")
            return False
        return True


def replay_from_dump(
    dump_dir: str,
    compare_outputs: bool = False,
    device: str = "cuda",
    run: bool = False,
    object_registry: Optional[Dict[Tuple[int, int], Any]] = None,
) -> Any:
    """
    Replay a function call from a dumped directory.

    This function:
    1. Loads metadata.jsonl to get function info
    2. Loads inputs.pt to get input tensors
    3. Moves tensors to specified device (default: cuda)
    4. Reconstructs the function call
    5. Optionally executes the function (if run=True)
    6. Optionally compares with saved outputs

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
    run : bool
        If True, try to resolve and execute the function
    object_registry : Optional[Dict[Tuple[int, int], Any]]
        Registry of stateful objects mapped by (process_id, self_id) tuple.
        This composite key ensures objects from different processes don't collide
        in multi-GPU environments where different processes may have objects
        at the same memory address.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'args': Positional arguments (tensors on specified device)
        - 'kwargs': Keyword arguments (tensors on specified device)
        - 'metadata': Full metadata
        - 'execution_result': Result of execution (if run=True)
        - 'comparison_match': Boolean indicating if result matched expected (if run=True and compare_outputs=True)
        If compare_outputs=True, also includes:
        - 'expected_tensors': Expected output tensors
        - 'expected_metadata': Expected output metadata
    """
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    # Load metadata from JSONL (last record has most complete state)
    metadata_jsonl_path = dump_path / "metadata.jsonl"
    if not metadata_jsonl_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {dump_dir}")

    metadata = _read_jsonl_last_record(metadata_jsonl_path)
    if metadata is None:
        raise ValueError(f"metadata.jsonl is empty in {dump_dir}")

    func_name = metadata["function_name"]

    # Load input tensors - auto-detect format (torch.save or safetensors)
    inputs_pt_path = dump_path / "inputs.pt"
    inputs_safetensors_path = dump_path / "inputs.safetensors"

    if inputs_pt_path.exists():
        input_tensors = torch.load(str(inputs_pt_path), map_location="cpu")
    elif inputs_safetensors_path.exists():
        try:
            from safetensors.torch import load_file

            input_tensors = load_file(str(inputs_safetensors_path), device="cpu")
        except ImportError:
            raise ImportError(
                "Dump was saved with safetensors but package not installed. "
                "Install with: pip install safetensors"
            ) from None
    else:
        raise FileNotFoundError(
            f"Neither inputs.pt nor inputs.safetensors found in {dump_dir}"
        )

    # Move tensors to specified device
    for key, tensor in input_tensors.items():
        input_tensors[key] = tensor.to(device)

    # Reconstruct args and kwargs
    args = []
    kwargs = {}
    input_metadata = metadata.get("input_metadata", {})

    # Get max arg index from both tensors and metadata
    max_arg_idx = -1

    for key in input_tensors.keys():
        if key.startswith("arg_"):
            idx = int(key.split("_")[1])
            max_arg_idx = max(max_arg_idx, idx)

    for key in input_metadata.keys():
        if key.startswith("arg_"):
            idx = int(key.split("_")[1])
            max_arg_idx = max(max_arg_idx, idx)

    # Reconstruct positional args in order so that we can replay
    # the function call exactly as it was logged.
    for i in range(max_arg_idx + 1):
        key = f"arg_{i}"
        if key in input_tensors:
            args.append(input_tensors[key])
        elif key in input_metadata:
            args.append(_reconstruct_value(input_metadata[key]))
        else:
            # Should not happen if dump is consistent, but safeguard
            _logger.warning(f"Missing argument {i} in dump.")
            args.append(None)

    # Add keyword arguments. Here the ordering is not important.
    for key in input_tensors.keys():
        if key.startswith("kwarg_"):
            kwarg_name = key.replace("kwarg_", "")
            kwargs[kwarg_name] = input_tensors[key]

    for key in input_metadata.keys():
        if key.startswith("kwarg_"):
            kwarg_name = key.replace("kwarg_", "")
            if kwarg_name not in kwargs:  # Don't override tensor kwargs
                kwargs[kwarg_name] = _reconstruct_value(input_metadata[key])

    _logger.info(f"Replaying {func_name} from {dump_dir}")
    _logger.info(f"  Args: {len(args)}, Kwargs: {list(kwargs.keys())}")

    result_dict: Dict[str, Any] = {"args": args, "kwargs": kwargs, "metadata": metadata}

    # Load expected outputs if needed - auto-detect format
    expected_outputs = {}
    output_metadata = {}
    if compare_outputs:
        outputs_pt_path = dump_path / "outputs.pt"
        outputs_safetensors_path = dump_path / "outputs.safetensors"

        if outputs_pt_path.exists():
            expected_outputs = torch.load(str(outputs_pt_path), map_location="cpu")
        elif outputs_safetensors_path.exists():
            try:
                from safetensors.torch import load_file

                expected_outputs = load_file(
                    str(outputs_safetensors_path), device="cpu"
                )
            except ImportError:
                raise ImportError(
                    "Dump was saved with safetensors but package not installed. "
                    "Install with: pip install safetensors"
                ) from None

        # Move output tensors to specified device
        for key, tensor in expected_outputs.items():
            expected_outputs[key] = tensor.to(device)

        output_metadata = metadata.get("output_metadata", {})
        result_dict["expected_tensors"] = expected_outputs
        result_dict["expected_metadata"] = output_metadata

    if run:
        module_name = metadata.get("module")
        self_id = metadata.get("self_id")
        process_id = metadata.get("process_id")

        func = None
        obj = None

        # Stateful replay logic for class methods calls.
        # Necessary for wrapped classes like BatchDecodeWithPagedKVCacheWrapper.
        # Use (process_id, self_id) as composite key to avoid collisions across processes.
        # In multi-GPU environments, different processes may have objects with the same
        # memory address (self_id), so we need to scope by process_id.
        if self_id is not None:
            registry_key = (process_id, self_id)
            if func_name.endswith(".__init__"):
                # This is a constructor call
                # Resolution: Get the class and instantiate it
                class_name = func_name.split(".")[
                    -2
                ]  # e.g. "Wrapper.__init__" -> "Wrapper"
                cls_obj = _resolve_function(module_name, class_name)
                if cls_obj and callable(cls_obj):
                    # Instantiate: obj = Class(*args[1:], **kwargs)
                    # Note: args[0] is 'self' placeholder in the dump for __init__, skip it
                    real_args = args[1:] if len(args) > 0 else []
                    try:
                        _logger.info(
                            f"Instantiating {class_name} (PID: {process_id}, ID: {self_id})..."
                        )
                        # We need to handle the case where __init__ is called.
                        # The safest way is to just call the class constructor.
                        # We assume the logged args match the constructor args.
                        obj = cls_obj(*real_args, **kwargs)
                        if object_registry is not None:
                            object_registry[registry_key] = obj
                        # __init__ returns None, but effectively we returned the object
                        execution_result = None
                        result_dict["execution_result"] = execution_result

                        # Since we successfully "ran" (instantiated), we can mark it done
                        # But there is no output to compare for __init__ usually (returns None)
                        if compare_outputs:
                            result_dict["comparison_match"] = (
                                True  # Trivial pass for __init__
                            )
                        return result_dict
                    except Exception as e:
                        _logger.error(f"Failed to instantiate {class_name}: {e}")
                        result_dict["execution_error"] = str(e)
                        return result_dict
            else:
                # Instance method call
                if object_registry is not None and registry_key in object_registry:
                    obj = object_registry[registry_key]
                    method_name = func_name.split(".")[-1]
                    if hasattr(obj, method_name):
                        func = getattr(obj, method_name)
                        # args[0] is 'self' placeholder, skip it
                        args = args[1:] if len(args) > 0 else []
                    else:
                        _logger.warning(f"Object {obj} has no method {method_name}")
                else:
                    _logger.warning(
                        f"Object (PID: {process_id}, ID: {self_id}) not found in registry."
                    )

        if func is None:
            func = _resolve_function(module_name, func_name)

        if func:
            try:
                _logger.info(f"Executing {module_name}.{func_name}...")
                execution_result = func(*args, **kwargs)
                result_dict["execution_result"] = execution_result

                if compare_outputs:
                    # Flatten execution result to dict for comparison
                    actual_outputs = {}
                    if isinstance(execution_result, torch.Tensor):
                        actual_outputs["result"] = execution_result
                    elif isinstance(execution_result, (tuple, list)):
                        for i, item in enumerate(execution_result):
                            if isinstance(item, torch.Tensor):
                                actual_outputs[f"result_{i}"] = item
                    elif isinstance(execution_result, dict):
                        # If result is already a dict of tensors? Unlikely for FlashInfer but possible
                        actual_outputs = execution_result

                    # Compare tensors
                    match = True
                    if expected_outputs:
                        match = _compare_results(actual_outputs, expected_outputs)

                    result_dict["comparison_match"] = match
                    if match:
                        _logger.info("Replay comparison passed!")
                    else:
                        _logger.warning("Replay comparison FAILED.")

            except Exception as e:
                _logger.error(f"Execution failed: {e}")
                import traceback

                _logger.error(traceback.format_exc())
                result_dict["execution_error"] = str(e)
        else:
            _logger.warning(
                f"Skipping execution: could not resolve {module_name}.{func_name}"
            )
    elif not compare_outputs:
        _logger.warning(
            "Automatic function resolution disabled. "
            "Pass run=True to execute, or manually call function."
        )

    return result_dict


def replay_sequence(root_dir: str, device: str = "cuda") -> list:
    """
    Replay a sequence of API calls from a root dump directory.

    This function iterates through all dump directories in the root directory,
    sorted by timestamp/sequence number, and replays them in order.

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing dump subdirectories
    device : str
        Target device for execution (default: "cuda")

    Returns
    -------
    list
        List of results from replay_from_dump calls
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root dump directory not found: {root_dir}")

    # Find all subdirectories that look like dumps
    # Pattern: YYYYMMDD_HHMMSS_milliseconds_pid<PID>_funcname_callXXXX
    dump_dirs = []
    for item in root_path.iterdir():
        if item.is_dir() and (item / "metadata.jsonl").exists():
            dump_dirs.append(item)

    # Sort by directory name (which starts with timestamp)
    dump_dirs.sort(key=lambda x: x.name)

    results = []
    total = len(dump_dirs)
    _logger.info(f"Found {total} dumps to replay from {root_dir}")

    # Registry for stateful objects (mapped by (process_id, self_id) tuple)
    # This composite key prevents collisions in multi-GPU/multi-process environments
    object_registry: Dict[Tuple[int, int], Any] = {}

    for i, dump_dir in enumerate(dump_dirs):
        _logger.info(f"[{i + 1}/{total}] Replaying {dump_dir.name}...")
        try:
            # We assume that for sequence replay, we want to EXECUTE the calls
            # and assume outputs are not necessarily present or we just want to verify it runs.
            # If outputs are present, we can compare.
            res = replay_from_dump(
                str(dump_dir),
                compare_outputs=True,
                device=device,
                run=True,
                object_registry=object_registry,
            )
            # Add dump_dir to the result for CLI reporting
            res["dump_dir"] = str(dump_dir)
            results.append(res)
        except Exception as e:
            # Let's record error and continue.
            _logger.error(f"Failed to replay {dump_dir.name}: {e}")
            results.append({"error": str(e), "dump_dir": str(dump_dir)})

    return results


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
_warn_dump()


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

    .. warning::
        This API logging feature is experimental and may change in future versions.

    Currently logs input and output values of the function using Python's logging library.
    This decorator integrates with Python's standard logging infrastructure while
    maintaining zero overhead when disabled (FLASHINFER_LOGLEVEL=0).

    Environment Variables
    ---------------------
    FLASHINFER_LOGLEVEL : int (default: 0)
        - 0: No logging (zero overhead - decorator returns original function)
        - 1: Log function name only (logged BEFORE execution - crash-safe)
        - 3: Log function name + inputs/outputs with metadata (inputs logged BEFORE execution - crash-safe)
        - 5: Log function name + inputs/outputs with metadata + tensor statistics (inputs logged BEFORE execution - crash-safe)
        - 10: Level 5 logging + dump metadata and input/output tensors to disk for reproducibility (preserves stride/contiguity)

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

    FLASHINFER_DUMP_SAFETENSORS : int (default: 0)
        - 0: Use torch.save format (preserves stride/contiguity)
        - 1: Use safetensors format (no pickle, but loses stride info)

    FLASHINFER_DUMP_INCLUDE : str (default: "")
        - Comma-separated list of patterns to include for dumping (fnmatch-style)

    FLASHINFER_DUMP_EXCLUDE : str (default: "")
        - Comma-separated list of patterns to exclude for dumping (fnmatch-style)

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
            self_id = None
            if args and hasattr(args[0], "__class__"):
                try:
                    class_name = args[0].__class__.__name__
                    if "Wrapper" in class_name or class_name in [
                        "BatchMLAPagedAttentionWrapper"
                    ]:
                        func_name = f"{class_name}.{func_name}"
                        self_id = id(args[0])
                except Exception:
                    pass

            # Level 10: Dump inputs BEFORE execution (crash-safe)
            dump_dir = None
            if _API_LOG_LEVEL >= 10:
                try:
                    dump_dir = _dump_function_inputs(
                        f, func_name, args, kwargs, self_id=self_id
                    )
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
