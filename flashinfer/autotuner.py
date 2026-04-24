import contextlib
import copy
import importlib
import inspect
import itertools
import json
import os
import tempfile
import threading
import weakref

import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional

import torch

# from tensorrt_llm.bindings.internal.runtime import delay_kernel
# from tensorrt_llm.logger import logger
from flashinfer.tllm_utils import delay_kernel

from .jit.core import logger
from .version import __version__ as _flashinfer_version

# This version should be updated whenever the nvfp4_cutlass backend is changed,
# such as when new kernels or configs are added. In such cases, the tuning configs
# should also be updated. Currently, this process is manual, but it should be automated in the future.
_nvfp4_cutlass_version = "0.1"


def _tactic_to_json(tactic):
    """Convert a tactic value to a JSON-compatible format.

    Any iterable (tuples, lists, C++ Array objects from TVM FFI, etc.) is
    recursively converted to plain Python lists so that ``json.dump`` can
    serialize them.  Scalars (int, float, bool, None) are returned as-is.
    """
    if isinstance(tactic, (tuple, list)):
        return [_tactic_to_json(v) for v in tactic]
    # Handle foreign iterable types (e.g. TVM FFI Array<int64_t>) that are
    # not plain tuple/list but still support iteration.
    if hasattr(tactic, "__iter__") and not isinstance(tactic, (str, bytes, dict)):
        return [_tactic_to_json(v) for v in tactic]
    if isinstance(tactic, bool):
        return tactic
    # Coerce numpy / pybind int types to plain Python int for JSON safety.
    if isinstance(tactic, int):
        return int(tactic)
    return tactic


def _json_to_tactic(val):
    """Convert a JSON-deserialized tactic value back to its original format.

    Lists are recursively converted to tuples so that compound tactics
    (e.g. CuteDSL's (tile_size, gemm1_tactic, gemm2_tactic)) are restored
    to their expected tuple form.
    """
    if isinstance(val, list):
        return tuple(_json_to_tactic(v) for v in val)
    return val


_METADATA_KEY = "_metadata"


def _get_cublas_version() -> str:
    """Return the cuBLAS version as ``major.minor.patch``.

    Checks sources in the same priority order as the runtime loader:
      1. LD_LIBRARY_PATH — probe the actual shared library via ctypes
         (tries cuBLAS and cuBLASLt .so variants via dynamic linker)
      2. pip package (any installed ``nvidia-cublas-*`` package)
      3. CUDA toolkit bundled with PyTorch (torch.version.cuda)

    All sources are normalized to ``major.minor.patch`` so that comparisons
    across different environments are meaningful.
    """
    import ctypes
    import ctypes.util
    import sys

    # Source 1: probe the actual loaded shared library via ctypes.
    # This respects LD_LIBRARY_PATH and reports the true runtime version.
    # We try both cuBLAS and cuBLASLt variants — whichever loads first wins.
    # Unversioned names are tried first (follow the dynamic linker);
    # ctypes.util.find_library is used as a fallback (queries ldconfig).
    if sys.platform == "win32":
        lib_specs = [("cublas.dll", "cublasGetProperty")]
    else:
        lib_specs = [
            ("libcublas.so", "cublasGetProperty"),
            ("libcublasLt.so", "cublasLtGetProperty"),
        ]
        for base, fn in (
            ("cublas", "cublasGetProperty"),
            ("cublasLt", "cublasLtGetProperty"),
        ):
            found = ctypes.util.find_library(base)
            if found:
                lib_specs.append((found, fn))
    for lib_name, fn_name in lib_specs:
        try:
            lib = ctypes.cdll.LoadLibrary(lib_name)
            fn = getattr(lib, fn_name)
            major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
            fn(0, ctypes.byref(major))
            fn(1, ctypes.byref(minor))
            fn(2, ctypes.byref(patch))
            return f"{major.value}.{minor.value}.{patch.value}"
        except (OSError, AttributeError):
            continue

    # Source 2: pip-installed nvidia-cublas package.
    # Pip versions may have 4 components (e.g. 13.2.1.1); truncate to
    # major.minor.patch to align with the ctypes output.
    # Package names are discovered dynamically to avoid hardcoding CUDA versions.
    try:
        import importlib.metadata as _ilm

        cublas_pkgs = sorted(
            (
                d.metadata["Name"]
                for d in _ilm.distributions()
                if (d.metadata["Name"] or "").startswith("nvidia-cublas")
            ),
            reverse=True,
        )
        for pkg in cublas_pkgs:
            try:
                pip_ver = _ilm.version(pkg)
                parts = pip_ver.split(".")
                return ".".join(parts[:3])
            except _ilm.PackageNotFoundError:
                continue
    except (ImportError, Exception):
        pass

    # Source 3: CUDA toolkit version from PyTorch (not the cuBLAS version
    # itself, but the best we can infer when neither source 1 nor 2 works).
    cuda_ver = getattr(torch.version, "cuda", None)
    if cuda_ver:
        return f"cuda-toolkit-{cuda_ver}"

    return "unknown"


def _collect_metadata() -> Dict[str, str]:
    """Collect environment metadata that can affect tactic-to-kernel mappings."""
    meta: Dict[str, str] = {}
    meta["flashinfer_version"] = _flashinfer_version
    meta["cuda_version"] = getattr(torch.version, "cuda", None) or "unknown"
    meta["cublas_version"] = _get_cublas_version()
    try:
        meta["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception:
        meta["cudnn_version"] = "unknown"
    try:
        meta["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        meta["gpu"] = "unknown"
    return meta


def get_config_path(is_module: bool):
    """Return the module name or file path for bundled per-GPU tuning configs."""
    dev_name = torch.cuda.get_device_name(0).replace(" ", "_")
    cutlass_ver = _nvfp4_cutlass_version.replace(".", "_")
    config_name = f"v{cutlass_ver}_trtllm_fused_moe_{dev_name}"
    if is_module:
        return f"flashinfer.tuning_configs.{config_name}"
    else:
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "tuning_configs",
            config_name + ".py",
        )


@dataclass(slots=True)
class DynamicTensorSpec:
    """
    A specification for a dynamic tensor dimension.
    Args:
        input_idx: A list of the indices of the input tensors.
        dim_idx: A list of the indices of the dimensions to tune.
            The length of input_idx and dim_idx must be the same.
            For every tensor mapped to the input_idx, their dimension mapped to the dim_idx must be the same.
        gen_tuning_buckets: A tuple of values to try or a function generating values.
        map_to_tuning_buckets: A function to map dimensions to valid values during inference.
        tensor_initializers: A list of functions to initialize the tensors.
    """

    input_idx: Tuple[int, ...]
    dim_idx: Tuple[int, ...]
    gen_tuning_buckets: Union[Tuple[int, ...], Callable]
    map_to_tuning_buckets: Callable
    tensor_initializers: List[Callable] = field(default_factory=lambda: None)

    def __post_init__(self):
        # Set default tensor_initializers if not provided
        if self.tensor_initializers is None:
            self.tensor_initializers = [
                lambda shapes, dtype, device: (
                    torch.rand(shapes, device=device) * 10 - 5
                ).to(dtype)
                for _ in range(len(self.input_idx))
            ]

    def __hash__(self) -> int:
        # FIXME: currently not hasing tensor_initializers
        return hash(
            (
                self.input_idx,
                self.dim_idx,
                # For gen_tuning_buckets, only hash if it's a tuple, otherwise hash its id
                self.gen_tuning_buckets
                if isinstance(self.gen_tuning_buckets, tuple)
                else id(self.gen_tuning_buckets),
                id(self.map_to_tuning_buckets),
            )
        )


@dataclass(slots=True, unsafe_hash=True)
class ConstraintSpec:
    """
    A specification for a constraint on a tensor dimension.
    Args:
        input_idx: The index of the input tensor.
        dim_idx: The index of the dimension to constrain.
        infer_shape: A function to infer the shape of the dimension.
    """

    input_idx: int
    dim_idx: int
    infer_shape: Callable


@dataclass(kw_only=True, unsafe_hash=True)
class TuningConfig:
    """Configuration for autotuning.

    This class specifies all the tuning configurations for a single tuning process.
    Args:
        dynamic_tensor_specs (Tuple[DynamicTensorSpec]): Specifications for how different tensor dimensions
            should be tuned to optimize performance. Each spec defines:
            - Which input tensor dimension is dynamic
            - How to generate tuning values
            - How to map dimensions to valid values during inference

            Example:
                >>> config = TuningConfig(
                ...     dynamic_tensor_specs=(
                ...         DynamicTensorSpec(
                ...             input_idx=[0],
                ...             dim_idx=[1],
                ...             gen_tuning_buckets=(32, 64, 128),
                ...             map_to_tuning_buckets=lambda x: ((x + 31) // 32) * 32
                ...         ),
                ...     )
                ... )
        constraint_specs (Tuple[ConstraintSpec]): Specifications for constraints on tensor dimensions.
            Each spec defines:
            - Which input tensor dimension is constrained
            - How to infer the shape of the dimension based on other dimensions

            Example:
                >>> config = TuningConfig(
                ...     constraint_specs=(
                ...         ConstraintSpec(
                ...             input_idx=1,
                ...             dim_idx=2,
                ...             infer_shape=lambda shapes: shapes[0][0] * 2
                ...         ),
                ...     )
                ... )
        use_cold_l2_cache (bool): Whether to use cold L2 cache.
            This flag is to create circular buffer of input tensors to avoid L2 cache hits to simulate cold L2 cache.
            Notice that not all tuning processes can benefit from this feature.
        use_cuda_graph (bool): Whether to use CUDA graph for the tuning process.
    """

    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: Tuple[ConstraintSpec, ...] = ()
    use_cold_l2_cache: bool = False
    use_cuda_graph: bool = False


@dataclass(unsafe_hash=True)
class StaticDim:
    val: int

    def _opt(self):
        return self.val


@dataclass(unsafe_hash=True)
class DynamicDim:
    """Range of one dimension"""

    min: int
    opt: int
    max: int

    def _opt(self):
        return self.opt


Dim = Union[DynamicDim, StaticDim]


@dataclass
class OptimizationProfile:
    """Ranges of all tensors, all dimension"""

    shapes: List[List[Dim]]
    tensor_initializers: List[Optional[Callable]]

    def get_hash_key(self):
        return self.get_opt_shapes()

    def get_opt_shapes(self):
        """Only the opt shapes are considered as hash key"""
        # TODO: remove duplicate shape generation
        opt_shapes = []
        for t in self.shapes:
            opt_shapes.append(tuple([d._opt() for d in t]))
        return tuple(opt_shapes)


# TODO: can/shall we use the torch builtin FakeTensor class?
@dataclass
class FakeTensor:
    dtype: torch.dtype
    device: torch.device
    shape: List[Dim]


class TunableRunner(ABC):
    @abstractmethod
    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile
    ) -> List[int]:
        """One tactic corresponding to one cuda kernel normally, but how to interpret the meaning
        of tactic is pure internal details of the runner.

        The autotuner will just pass the tactic value to the forward w/o any knowledge on what the tactic
        means.

        tactic==-1 has special meaning, means the fallback kernel which should be able to implement any shapes
        This fallback tactic is needed for 2 reasons:
            * when the autotuner cannot find a valid tactic in it's cache.
            * in eager mode, w/o autotuning the custom op should have at least one kernel, which makes the autotuning
              process an optional process, such that user can opt out.

        We choose not to have a standalone can_implement function, the tactics returned by get_valid_tactics should return
        valid kernel for these given input tensors.
        """
        return [-1]

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        """Return extra values to include in the autotune cache key.

        Override this method to differentiate cache entries that share the same
        input shapes but differ in other properties (e.g. output dtype).
        The returned tuple must be hashable.

        Returned values must be synthesis-invariant: the same tuple must
        be produced for the caller's real inputs and for tensors the
        autotuner would synthesize for the same profile (i.e., depend only
        on dtype, is-None flags, or scalar-argument values -- not on
        per-tensor content).
        """
        return ()

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,  # all others are keyword args only
    ) -> Any:
        """Forward pass for tunable runners.

        Args:
            inputs: List of input tensors (position-only argument)
            tactic: Integer ID specifying which implementation tactic to use.
                   -1 (default) represents the fallback tactic that must be implemented
                   to handle any input shapes when autotuning is disabled.
            do_preparation: When True, allows one-time setup operations to be performed
                          before tactic evaluation begins. These operations are excluded
                          from the performance measurements during autotuning. Notice that
                          anything prepared in this phase should be persistent in the forward
                          and can be accessed by the following forward calls.

        Returns:
            Any: Output of the forward pass

        """
        raise NotImplementedError

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@contextlib.contextmanager
def autotune(
    tune_mode: bool = True,
    cache: Optional[str] = None,
    tuning_buckets: Optional[Tuple[int, ...]] = None,
    round_up: Optional[bool] = None,
):
    """Context manager for autotuning with optional file-based caching.

    Controls how FlashInfer profiles and selects the best kernel implementation
    for each operation.  When ``tune_mode=True``, uncovered shapes are profiled
    on the fly; when ``False``, only previously cached results are used.

    .. note::
        The ``cache`` parameter is **experimental**.  Single-process and
        multi-threaded use is fully supported.  Multi-process and multi-node
        use works under low write contention but is best-effort: concurrent writes
        to a shared cache file may result in lost updates from race conditions.

    Args:
        tune_mode: If ``True``, profile uncovered shapes during execution.
            If ``False``, only use cached/loaded configs (no profiling).
        cache: Optional path to a JSON config file.
            On entry, configs are loaded from this file (if it exists).
            On exit, configs are saved back to this file (only when
            ``tune_mode=True``).
        tuning_buckets: Optional sequence of integer measurement points.
            When provided, replaces the default power-of-2 buckets for all
            operations within this context.  For example,
            ``tuning_buckets=(64, 128, 256, 512, 1024)`` profiles exactly
            those batch sizes.  Duplicates are removed and values are sorted
            automatically.  Must contain **at least one** value when provided;
            pass ``None`` (or omit) to inherit the current buckets.

            **Selecting buckets** -- more buckets give finer-grained profiling
            (better peak performance) at the cost of longer tuning time; fewer
            buckets make tuning faster but coarser.  When the set of possible
            runtime sizes is known in advance (e.g. vLLM knows which batch
            sizes will appear), passing those exact sizes yields the best
            results.  Focus on the size range that matters most for your
            workload: for latency-sensitive inference, add more small sizes;
            for throughput-oriented serving, cover the larger range.

        round_up: Controls how runtime sizes map to profiled buckets.

            * ``None`` (default) -- inherit from enclosing ``autotune()``
              context, or ``False`` if there is none.
            * ``False`` -- round **down** to the largest bucket <= the
              runtime size (floor semantics, the historical default).
            * ``True`` -- round **up** to the smallest bucket >= the runtime
              size (ceil semantics).

            For example, with buckets ``(128, 256, 512)`` and a runtime batch
            of 200: ``round_up=False`` selects 128 while ``round_up=True``
            selects 256.  Rounding up can improve performance when the best
            kernel for a larger bucket also performs well at nearby smaller
            sizes (see the PR discussion for benchmark data on cuDNN plans).

    Raises:
        ValueError: If ``tuning_buckets`` is provided but empty.

    .. rubric:: Edge-case behaviour

    **Empty buckets** -- ``autotune(tuning_buckets=())`` raises
    :class:`ValueError` immediately.

    **Unsupported sizes** -- Sizes that a kernel cannot handle are filtered
    out during profiling (``get_valid_tactics``).  If a tactic still fails at
    runtime the error is caught, a warning is logged, and that tactic is
    skipped.  At inference time, ``map_to_tuning_buckets`` maps the runtime
    size to the nearest profiled bucket, so an unsupported raw size never
    reaches the kernel directly.

    **Round-up beyond the largest bucket** -- When ``round_up=True`` and the
    runtime size exceeds every bucket, the value is clamped to the largest
    bucket (see :func:`~flashinfer.fused_moe.utils.round_to_nearest_bucket`).

    **Nested / sequential contexts** -- Overrides are managed on a per-thread
    stack.  A nested ``autotune()`` pushes its overrides; on exit the outer
    context's values are restored.  Sequential contexts are fully independent.
    Different buckets produce different cache keys, so entries never collide.

    **Using both parameters together** is fully supported:
    ``autotune(tuning_buckets=(100, 300, 600), round_up=True)`` profiles
    at 100, 300, and 600 and rounds runtime sizes *up* to the nearest of
    those buckets at inference time.

    Examples::

        # Tune and persist results to a cache file
        with autotune(True, cache="my_configs.json"):
            model(inputs)

        # Use custom measurement points
        with autotune(True, tuning_buckets=(64, 128, 256, 512)):
            model(inputs)

        # Round up to next bucket during inference
        with autotune(False, cache="my_configs.json", round_up=True):
            model(inputs)

        # Combine custom buckets with round-up
        with autotune(True, tuning_buckets=(100, 300, 600), round_up=True):
            model(inputs)  # profiles at 100, 300, 600

        # Nested contexts: inner overrides, outer restored on exit
        with autotune(True, tuning_buckets=(128, 256)):
            model(inputs)   # uses (128, 256)
            with autotune(True, tuning_buckets=(64, 512)):
                model(inputs)  # uses (64, 512)
            model(inputs)   # back to (128, 256)
    """
    tuner = AutoTuner.get()

    if tuning_buckets is not None and len(tuning_buckets) == 0:
        raise ValueError(
            "tuning_buckets must contain at least one value when provided; "
            "pass None (or omit) to inherit the current buckets"
        )

    # Load configs from cache file on entry (if it exists).
    # cache_valid is False when the file exists but has a metadata mismatch;
    # in that case we skip saving on exit to avoid overwriting configs from
    # a different environment.
    cache_valid = True
    if cache is not None:
        with tuner._lock:
            tuner._file_configs.clear()
            tuner._logged_file_hits.clear()
        if os.path.isfile(cache):
            cache_valid = tuner.load_configs(cache)

    # Push tuning bucket overrides onto per-thread stack.  Inherits from the
    # current top-of-stack when a parameter is not explicitly supplied.
    override_stack = tuner._get_override_stack()
    current_buckets = override_stack[-1][0] if override_stack else None
    current_round_up = override_stack[-1][1] if override_stack else False
    new_buckets = (
        tuple(sorted(set(tuning_buckets)))
        if tuning_buckets is not None
        else current_buckets
    )
    new_round_up = round_up if round_up is not None else current_round_up
    pushed = tuning_buckets is not None or round_up is not None
    if pushed:
        override_stack.append((new_buckets, new_round_up))

    # Reference-counted tuning mode: is_tuning_mode stays True as long as
    # at least one autotune(True) context is active, even if an
    # autotune(False) context overlaps on another thread.
    try:
        with tuner._lock:
            if tune_mode:
                tuner._active_tuning_contexts += 1
            old_mode = tuner.is_tuning_mode
            tuner.is_tuning_mode = tuner._active_tuning_contexts > 0
            autotune_enabled = tune_mode and not old_mode
        if autotune_enabled:
            logger.info("[Autotuner]: Autotuning process starts ...")
    except BaseException:
        if pushed:
            override_stack.pop()
        raise

    try:
        yield
    finally:
        with tuner._lock:
            if tune_mode:
                tuner._active_tuning_contexts -= 1
            tuner.is_tuning_mode = tuner._active_tuning_contexts > 0

        # Pop the override we pushed (thread-local, no lock needed).
        if pushed:
            override_stack.pop()

        if autotune_enabled:
            logger.info("[Autotuner]: Autotuning process ends")

        # Save configs on exit when tuning with a cache path,
        # but only if new profiling results were added this session
        # and the cache file was valid (no environment mismatch).
        if cache is not None and cache_valid and tune_mode and tuner._dirty:
            tuner.save_configs(cache)


@dataclass
class AutoTunerStatistics:
    """Statistics collected by the AutoTuner.

    Attributes:
        cache_misses (int): Number of cache misses requiring fallback
        cache_miss_config_collection (Dict[str, Set[OptimizationProfile]]): Collection of configs that caused cache misses
        failed_profiling_count (Dict[str, int]): Number of failed profiling attempts per operation
        tuned_op_total_configs (Dict[str, int]): Total configurations tried per operation
        tuned_op_successful_configs (Dict[str, int]): Successful configurations per operation
    """

    cache_misses: int = 0
    cache_miss_config_collection: Dict[str, Set[tuple]] = field(default_factory=dict)
    failed_profiling_count: Dict[
        str, Set[Tuple[str, TunableRunner, OptimizationProfile]]
    ] = field(default_factory=dict)
    tuned_op_total_configs: Dict[str, int] = field(default_factory=dict)
    tuned_op_successful_configs: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a string representation of collected statistics."""
        stats_str = ""
        stats_str += f"Cache misses: {self.cache_misses}\n"
        if self.cache_miss_config_collection:
            stats_str += "Cache miss config collection:\n"
            for op, profiles in sorted(self.cache_miss_config_collection.items()):
                stats_str += f"  {op}:\n"
                for profile in sorted(profiles, key=str):
                    stats_str += f"    - Config: {profile}\n"

        if self.tuned_op_total_configs:
            stats_str += "Tuned operations:\n"
            for op in sorted(self.tuned_op_total_configs.keys()):
                total = self.tuned_op_total_configs[op]
                successful = self.tuned_op_successful_configs.get(op, 0)
                failed = len(self.failed_profiling_count.get(op, set()))
                success_rate = (successful / total * 100) if total > 0 else 0
                stats_str += f"  {op}:\n"
                stats_str += f"    - Total configs tried: {total}\n"
                stats_str += f"    - Successful configs: {successful}\n"
                stats_str += f"    - Failed profiling count: {failed}\n"
                if failed > 0:
                    stats_str += "    - Failed profiling combinations:\n"
                    for failed_key in self.failed_profiling_count[op]:
                        stats_str += f"      - {failed_key}\n"
                stats_str += f"    - Success rate: {success_rate:.1f}%\n"

        return stats_str


@lru_cache(maxsize=None)
def load_from_file(key):
    module_name = get_config_path(is_module=True)
    try:
        module = importlib.import_module(module_name)
        best_configs = module.best_configs
    except (ImportError, AttributeError):
        best_configs = None
    if best_configs is not None:
        k = str((key[0], key[1], key[3]))
        if k in best_configs:
            logger.info(f"[Autotuner]: Loading configs for {k} from file.")
            return True, best_configs[k][0], best_configs[k][1], None
    logger.info(
        f"[Autotuner]: Loading configs for {key} from file failed; Using default configs instead."
    )
    return False, 0, -1, None


class AutoTuner:
    """AutoTuner for optimizing TensorRT-LLM operations.

    This class handles automatic performance tuning of tensor operations by profiling
    different implementations and caching the best performing configurations.

    Args:
        warmup (int): Number of warmup iterations before profiling (default: 3)
        repeat (int): Number of profiling iterations for averaging (default: 10)
        stream_delay_micro_secs (int): Delay on CUDA stream before the profiled kernel runs in microseconds (default: 1000)
    """

    _CUDA_GRAPH_DELAY_MICRO_SECS = 100
    _instance = None
    _class_lock = threading.Lock()

    def __init__(self, warmup=3, repeat=10, stream_delay_micro_secs=1000):
        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs
        self.profiling_cache = {}
        self.is_tuning_mode = False
        self._active_tuning_contexts = 0

        # Reentrant lock protecting all mutable state on this instance.
        # RLock is used because choose_one() calls search_cache() internally.
        self._lock = threading.RLock()

        # Add statistics tracking
        self.stats = AutoTunerStatistics()

        self.profiling_debug = True

        # User-loaded configs from JSON files (populated by load_configs or autotune(cache=))
        self._file_configs: Dict[str, Tuple] = {}
        # Track which file config keys have been logged (to avoid per-call spam)
        self._logged_file_hits: Set[Tuple[str, str]] = set()
        # Set when new profiling results are added; cleared on save.
        self._dirty = False
        self._dirty_seq = 0

        # Per-thread stack of (tuning_buckets, round_up) overrides set by
        # autotune() context manager.  Using threading.local ensures concurrent
        # autotune() contexts on different threads don't clobber each other.
        self._override_local = threading.local()
        # Cache overridden TuningConfig objects to keep stable object identity
        # for _find_nearest_profile's LRU cache.
        # Two-level: WeakKeyDictionary[TuningConfig, Dict[(buckets, round_up), TuningConfig]]
        # keyed by identity so configs differing only in tensor_initializers
        # (whose __hash__ is the same) don't collide.
        self._override_config_cache: weakref.WeakKeyDictionary = (
            weakref.WeakKeyDictionary()
        )

    def _get_override_stack(self) -> List:
        """Return the per-thread override stack, creating it on first access."""
        local = self._override_local
        if not hasattr(local, "stack"):
            local.stack = []
        return local.stack

    @property
    def _override_tuning_buckets(self) -> Optional[Tuple[int, ...]]:
        """Currently active tuning-bucket override for this thread, or ``None``."""
        stack = self._get_override_stack()
        if stack:
            return stack[-1][0]
        return None

    @property
    def _override_round_up(self) -> bool:
        """Whether the current thread's active override requests round-up semantics."""
        stack = self._get_override_stack()
        if stack:
            return stack[-1][1]
        return False

    @classmethod
    def get(cls):
        # Double-checked locking for thread-safe singleton creation
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = AutoTuner()
        return cls._instance

    def search_cache(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
        inputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[bool, int, int, OptimizationProfile]:
        """Search for cached profiling results matching the current configuration.

        Searches the following sources in priority order:
            1. In-memory profiling_cache (from live autotuning in the current process)
            2. User-loaded configs (via load_configs() or autotune(cache=...))
            3. Bundled package configs (legacy .py files)
            4. Fallback tactic (-1)

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            input_shapes (Tuple[torch.Size]): Shapes of the input tensors
            tuning_config (TuningConfig): Tuning configuration
            inputs (Optional[List[torch.Tensor]]): Raw input tensors, used to compute
                per-runner cache key extras via get_cache_key_extras().

        Returns:
            A tuple containing:
            [is_cache_hit, runner_id, tactic, stored_profile]

        Note:
            input_shapes and inputs feed orthogonal paths inside
            this method: input_shapes flows only into
            _find_nearest_profile (bucket matching) and inputs
            flows only into r.get_cache_key_extras(inputs).  Callers
            may therefore pass them describing different tensor sets
            (e.g. a profile's opt_shapes alongside the caller's real
            inputs) so long as get_cache_key_extras is
            synthesis-invariant; see: TunableRunner.get_cache_key_extras.
        """
        with self._lock:
            for r in runners:
                extras = r.get_cache_key_extras(inputs) if inputs is not None else ()
                cache_key = AutoTuner._get_cache_key(
                    custom_op, r, input_shapes, tuning_config, extras
                )
                # 1. In-memory cache (from live tuning)
                if cache_key in self.profiling_cache:
                    return True, *self.profiling_cache[cache_key]

                # Build the hash-free file key used by both user configs and bundled configs
                file_key = str((cache_key[0], cache_key[1], cache_key[3]))

                # 2. User-loaded configs (from load_configs or autotune(cache=...))
                #    Always consulted, even during tuning mode — loaded configs take priority
                #    so that already-tuned shapes are never re-profiled.
                if file_key in self._file_configs:
                    runner_name, tactic = self._file_configs[file_key]
                    runner_id = next(
                        (
                            i
                            for i, runner in enumerate(runners)
                            if runner.__class__.__name__ == runner_name
                        ),
                        0,  # fallback to first runner if name not found
                    )
                    log_key = (custom_op, runner_name)
                    if log_key not in self._logged_file_hits:
                        self._logged_file_hits.add(log_key)
                        logger.info(
                            f"[Autotuner]: Config cache hit for {custom_op} "
                            f"(runner={runner_name}, source=config file)"
                        )
                    return True, runner_id, tactic, None

                # 3. Bundled package configs (legacy .py files)
                if (
                    os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0") == "1"
                    and not self.is_tuning_mode
                ):
                    output = load_from_file(cache_key)
                    if output[0]:  # is_cache_hit
                        return output

            # 4. Fallback
            return False, 0, -1, None

    def _apply_tuning_overrides(self, tuning_config: TuningConfig) -> TuningConfig:
        """Return a TuningConfig with overridden buckets/rounding if overrides are active.

        The result is cached so the same logical override produces the same
        object, keeping ``_find_nearest_profile``'s LRU cache effective.
        """
        buckets = self._override_tuning_buckets
        round_up_flag = self._override_round_up

        per_config = self._override_config_cache.get(tuning_config)
        cache_key = (buckets, round_up_flag)
        if per_config is not None and cache_key in per_config:
            return per_config[cache_key]

        from .fused_moe.utils import make_bucket_mapper, next_positive_power_of_2

        new_specs = []
        for spec in tuning_config.dynamic_tensor_specs:
            new_gen: Union[Tuple[int, ...], Callable]
            new_map: Callable
            if buckets is not None:
                new_gen = tuple(sorted(set(buckets)))
                new_map = make_bucket_mapper(new_gen, round_map=round_up_flag)
            elif round_up_flag:
                if isinstance(spec.gen_tuning_buckets, (list, tuple)):
                    sorted_gen = tuple(sorted(set(spec.gen_tuning_buckets)))
                    new_gen = sorted_gen
                    new_map = make_bucket_mapper(sorted_gen, round_map=True)
                else:
                    # gen_tuning_buckets is a callable — keep it, but build a
                    # mapper that rounds up to power-of-2 and clamps to the
                    # generated bucket set so we never exceed the last bucket.
                    gen_fn = spec.gen_tuning_buckets
                    new_gen = gen_fn

                    def _clamped_po2_mapper(x, _gen_fn=gen_fn):
                        buckets = tuple(sorted(set(_gen_fn(x))))
                        return make_bucket_mapper(buckets, round_map=True)(
                            next_positive_power_of_2(x)
                        )

                    new_map = _clamped_po2_mapper
            else:
                new_specs.append(spec)
                continue

            new_specs.append(
                DynamicTensorSpec(
                    input_idx=spec.input_idx,
                    dim_idx=spec.dim_idx,
                    gen_tuning_buckets=new_gen,
                    map_to_tuning_buckets=new_map,
                    tensor_initializers=spec.tensor_initializers,
                )
            )

        new_config = TuningConfig(
            dynamic_tensor_specs=tuple(new_specs),
            constraint_specs=tuning_config.constraint_specs,
            use_cold_l2_cache=tuning_config.use_cold_l2_cache,
            use_cuda_graph=tuning_config.use_cuda_graph,
        )
        self._override_config_cache.setdefault(tuning_config, {})[cache_key] = (
            new_config
        )
        return new_config

    def choose_one(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[TunableRunner, int]:
        """Choose the best runner and tactic combination through performance profiling.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            tuning_config (TuningConfig): Configuration for the tuning process
            inputs (List[torch.Tensor]): Input tensors for profiling
            **kwargs: Arbitrary keyword arguments, will be passed to get_valid_tactics and forward method of each runner

        Returns:
            Tuple[TunableRunner, int]: A tuple containing:
                - The selected runner implementation
                - The best tactic ID for that runner (-1 if using fallback)

        Note:
            The method profiles different implementations and tactics to find the
            optimal combination based on performance measurements. It caches results
            to avoid redundant profiling of the same configuration.
            Although runners[0] with tactic=-1 is always treated as the fallback runner.
            Runner authors are suggested to provide a fallback implementation for each runner to avoid potential issues.
        """
        # Hold the lock for the entire method.  In non-tuning mode this is a
        # fast cache lookup; in tuning mode it serializes GPU profiling which
        # must not run concurrently (measurements would interfere).
        # Note: this is a single global lock, so multi-threaded tuning on
        # separate GPUs is serialized.  Use multi-process (one per GPU) for
        # parallel multi-GPU tuning.
        with self._lock:
            # Apply tuning bucket / rounding overrides from autotune() context.
            if self._override_tuning_buckets is not None or self._override_round_up:
                tuning_config = self._apply_tuning_overrides(tuning_config)

            input_shapes = tuple(self._get_input_sizes(inputs))

            # Early return if it's not tuning, use cache found one or fallback one
            if not self.is_tuning_mode:
                is_cache_hit, runner_id, tactic, stored_profile = self.search_cache(
                    custom_op, runners, input_shapes, tuning_config, inputs=inputs
                )
                runner = runners[runner_id]
                # TODO: check the stored runner and tactic can implement this shape here
                # Should not directly try (runner, tactic) here, or it will hurt a lot of inference perf.

                # Record the cache miss config.
                # Expect no cache miss in inference. Thus, any cache miss should be recorded.
                if not is_cache_hit:
                    logger.debug(
                        f"[AutoTuner]: Using fallback tactic for {custom_op} with input shapes {input_shapes}"
                    )
                    logger.debug(
                        f"[AutoTuner]: Generated key{AutoTuner._get_cache_key(custom_op, runners[0], input_shapes, tuning_config, runners[0].get_cache_key_extras(inputs))}"
                    )
                return runner, tactic

            assert len(runners) > 0, "At least one runner is required"
            assert all([isinstance(r, TunableRunner) for r in runners]), (
                "All Given runners must be subclass of TunableRunner"
            )

            profiles = self._generate_optimization_profiles(tuning_config, inputs)
            # Record the total configs to try
            self.stats.tuned_op_total_configs[custom_op] = len(profiles)

            # Pre-compute runner arg names to avoid calling inspect.signature in the loop
            runner_arg_names_map = {}
            for r in runners:
                runner_arg_names_map[r] = {
                    param.name
                    for param in inspect.signature(r.forward).parameters.values()
                }

            pbar = None
            for _step, p in enumerate(profiles):
                try:
                    # Check the cache before synthesizing profile inputs.
                    # `_prepare_input_tensors` launches a GPU kernel per
                    # `DynamicTensorSpec`; on a cache hit the synthesized
                    # tensors are immediately discarded.  Skipping them
                    # matters for callers that invoke `choose_one`
                    # repeatedly inside `autotune(True)` -- otherwise every
                    # warm-cache forward still fires those kernels, which
                    # CUPTI / nsys will attribute to the measured region
                    # and inflate per-forward timings (asymmetrically
                    # across runners, since their `dynamic_tensor_specs`
                    # differ -- enough to invert measured rankings).
                    #
                    # Passing `inputs=inputs` is safe: `_get_cache_key`
                    # uses `p.get_opt_shapes()` plus
                    # `get_cache_key_extras`, whose contract is to return
                    # dtype-like properties preserved by the synthesis
                    # initializers.  Matches the non-tuning branch above
                    # and the post-loop `search_cache` call below.
                    is_cache_hit, runner_id, tactic, _ = self.search_cache(
                        custom_op,
                        runners,
                        p.get_opt_shapes(),
                        tuning_config,
                        inputs=inputs,
                    )
                    if not is_cache_hit:
                        # Synthesize inputs only on the profiling path.
                        tensors = self._prepare_input_tensors(p, inputs)
                        if pbar is None:
                            pbar = tqdm.tqdm(
                                total=len(profiles),
                                initial=_step,
                                desc=f"[AutoTuner]: Tuning {custom_op}",
                                unit="profile",
                                leave=True,
                            )
                        min_time = float("inf")
                        # Initialize runner and tactic as None in case of no valid tactic or runners are found
                        runner_id, tactic = None, None
                        skipped_count = 0
                        for r_id, r in enumerate(runners):
                            # TODO: use FakeTensor here.
                            valid_tactics = r.get_valid_tactics(tensors, p)
                            runner_arg_names = runner_arg_names_map[r]
                            if (
                                "do_preparation" in runner_arg_names
                                and len(valid_tactics) > 0
                            ):
                                r(tensors, tactic=-1, do_preparation=True, **kwargs)
                            for tac in valid_tactics:
                                try:
                                    time_measured = self._profile_single_kernel(
                                        r, tensors, tac, tuning_config, **kwargs
                                    )
                                except torch.cuda.OutOfMemoryError:
                                    raise
                                except Exception as e:
                                    skipped_count += 1
                                    shapes = self._get_input_sizes(tensors)
                                    logger.debug(
                                        f"[Autotuner]: Skipping tactic {r} {tac}, due to failure while profiling: {e}"
                                    )
                                    logger.debug(
                                        f"[Autotuner]: Failed when profiling {r} {tac}, shapes={shapes}. Error occurred: {e}"
                                    )

                                    # Clear any pending async CUDA errors (e.g.
                                    # cudaErrorIllegalInstruction from a failed
                                    # kernel warmup run) so they don't surface
                                    # later during CUDA graph capture.
                                    # torch.cuda.synchronize() surfaces the error
                                    # but does NOT clear the sticky CUDA error flag;
                                    # only cudaGetLastError() resets it.
                                    with contextlib.suppress(Exception):
                                        torch.cuda.synchronize()
                                    with contextlib.suppress(Exception):
                                        torch.cuda.cudart().cudaGetLastError()

                                    # Record the failed profiling combinations
                                    if (
                                        custom_op
                                        not in self.stats.failed_profiling_count
                                    ):
                                        self.stats.failed_profiling_count[custom_op] = (
                                            set()
                                        )
                                    self.stats.failed_profiling_count[custom_op].add(
                                        AutoTuner._get_cache_key(
                                            custom_op,
                                            r,
                                            p.get_opt_shapes(),
                                            tuning_config,
                                            r.get_cache_key_extras(tensors),
                                        )
                                    )

                                    # Set time_measured to inf to notify the failure of the tactic. This can happen when `get_valid_tactics` mistakenly return wrong tactics
                                    # or some runtime error occurs during profiling.
                                    time_measured = float("inf")
                                if time_measured < min_time:
                                    min_time = time_measured
                                    runner_id, tactic = r_id, tac

                        if skipped_count > 0:
                            logger.info(
                                f"[Autotuner]: Skipped {skipped_count} unsupported tactic(s) for {custom_op} "
                                f"(enable debug logs to see details)"
                            )

                        if runner_id is not None:
                            # At least one valid (runner, tactic) pair is found
                            cache_key = AutoTuner._get_cache_key(
                                custom_op,
                                runners[runner_id],
                                p.get_opt_shapes(),
                                tuning_config,
                                runners[runner_id].get_cache_key_extras(tensors),
                            )
                            # inspect call stack
                            self.profiling_cache[cache_key] = (runner_id, tactic, p)
                            self._dirty = True
                            self._dirty_seq += 1
                            self.stats.tuned_op_successful_configs[custom_op] = (
                                self.stats.tuned_op_successful_configs.get(custom_op, 0)
                                + 1
                            )
                            logger.debug(
                                f"[Autotuner]: profiling chosen runner: {runners[runner_id]} {tactic} for {cache_key}"
                            )

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.warning(
                        "[Autotuner]: OOM detected, falling back to default tactic"
                    )
                    return runners[0], -1

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            # Get the best runner and tactic from cache
            # If no valid tactic is found, the fallback runner and tactic will be used
            _, runner_id, tactic, _ = self.search_cache(
                custom_op, runners, input_shapes, tuning_config, inputs=inputs
            )

            return runners[runner_id], tactic

    def _get_input_sizes(self, inputs: List[torch.Tensor]) -> List[torch.Size]:
        """Return ``torch.Size`` for each input, using ``(0,)`` for non-Tensor values."""
        sizes = [
            input.size() if isinstance(input, torch.Tensor) else torch.Size((0,))
            for input in inputs
        ]

        return sizes

    def _profile_single_kernel(
        self,
        runner: TunableRunner,
        inputs: List[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        **kwargs,
    ) -> float:
        """Profile a single kernel implementation for performance measurement.

        Args:
            runner (TunableRunner): The runner implementation to profile
            inputs (List[torch.Tensor]): Input tensors for the kernel
            tactic (int): Tactic ID to use for this profiling run
            tuning_config (TuningConfig): Tuning configuration

        Returns:
            Average execution time in milliseconds

        Note:
            The method performs warmup runs, then measures multiple iterations
            to get an average execution time. Stream synchronization and delays
            are used to ensure accurate timing.
        """
        input_tensor_batches = self._prepare_input_tensors_with_batches(
            inputs, tuning_config
        )

        stream = torch.cuda.current_stream()
        avg_time = float("inf")

        def pure_profile(stream: torch.cuda.Stream, repeat: int) -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            graph = torch.cuda.CUDAGraph()

            def _run_kernels():
                for r in range(repeat):
                    runner(
                        input_tensor_batches[r % len(input_tensor_batches)],
                        tactic=tactic,
                        **kwargs,
                    )

            with torch.cuda.stream(stream):
                if tuning_config.use_cuda_graph:
                    with torch.cuda.graph(graph):
                        _run_kernels()

                stream.synchronize()

                # Delay the profiled kernel launch to eliminate affects of host time overhead in profiling.
                delay_kernel_time_usec = (
                    self._CUDA_GRAPH_DELAY_MICRO_SECS
                    if tuning_config.use_cuda_graph
                    else self.stream_delay_micro_secs
                )
                delay_kernel(delay_kernel_time_usec)

                start.record()

                if tuning_config.use_cuda_graph:
                    graph.replay()
                else:
                    _run_kernels()

                end.record()
                stream.synchronize()

                return start.elapsed_time(end) / repeat

        # warm up, no timing
        for _ in range(self.warmup):
            runner(input_tensor_batches[-1], tactic=tactic, **kwargs)

        avg_time = pure_profile(stream, self.repeat)

        shapes = self._get_input_sizes(inputs)
        logger.debug(
            f"[Autotuner]: profiling {runner} {tactic}, shapes={shapes}, avg_time {avg_time}"
        )

        return avg_time

    def _generate_optimization_profiles(
        self, tuning_config: TuningConfig, inputs: List[torch.Tensor]
    ) -> List[OptimizationProfile]:
        """Generate optimization profiles for autotuning.

        Args:
            tuning_config (TuningConfig): Tuning configuration
            inputs (List[torch.Tensor]): List of input tensors

        Returns:
            List of OptimizationProfile objects representing different configurations

        Note:
            This method performs a cartesian product of all possible dimension
            combinations specified in dynamic_tensor_specs.
        """
        # every dimension created from the concrete input tensor shape
        # generate some dynamic dimension description based on the dynamic_tensors

        # Zero handles the case where a TRTLLM op has optional or scalar inputs.
        base_profile = OptimizationProfile(
            [
                (
                    [StaticDim(x) for x in t.size()]
                    if isinstance(t, torch.Tensor)
                    else [StaticDim(0)]
                )
                for t in inputs
            ],
            [None] * len(inputs),
        )

        generated_profiles: List[OptimizationProfile] = []

        dynamic_dims: List[Tuple[Any, ...]] = []

        for spec in tuning_config.dynamic_tensor_specs:
            assert inspect.isfunction(spec.gen_tuning_buckets) or isinstance(
                spec.gen_tuning_buckets, (list, tuple)
            ), (
                "The given dynamic dimension must provide a opt value generation function or a list of opt values"
            )
            assert len(spec.input_idx) == len(spec.dim_idx), (
                f"The number of input indices and dimension indices must be the same, got {len(spec.input_idx)} and {len(spec.dim_idx)}"
            )
            assert len(spec.tensor_initializers) == len(spec.input_idx), (
                f"The number of tensor initializers and input indices must be the same, got {len(spec.tensor_initializers)} and {len(spec.input_idx)}"
            )
            for i, idx in enumerate(spec.input_idx):
                base_profile.tensor_initializers[idx] = spec.tensor_initializers[i]

            if inspect.isfunction(spec.gen_tuning_buckets):
                opt_shapes = spec.gen_tuning_buckets(
                    base_profile.shapes[spec.input_idx[0]][spec.dim_idx[0]]._opt()
                )
            else:
                opt_shapes = spec.gen_tuning_buckets

            # Normalize candidate buckets to be monotonically non-decreasing and non-empty
            opt_shapes = tuple(sorted(set(opt_shapes)))
            assert len(opt_shapes) > 0, "Empty tuning buckets are not allowed"

            opt_shapes_max = {
                v1: v2
                for v1, v2 in zip(
                    opt_shapes, tuple(opt_shapes[1:]) + (float("inf"),), strict=True
                )
            }
            dynamic_dims.append(
                (spec.input_idx, spec.dim_idx, opt_shapes_max, opt_shapes)
            )

        # grid search, do cartesian product for all the dynamic axis
        dim_grids = itertools.product(*[d[-1] for d in dynamic_dims])
        for opt_point in dim_grids:
            p = copy.deepcopy(base_profile)
            for pos, (input_idx, dim_idx, opt_shapes_max, _opt_shapes) in enumerate(
                dynamic_dims
            ):
                opt_value = opt_point[pos]
                # TODO: fix me, how to set the min and max?
                min_value = opt_value
                max_value = opt_shapes_max[opt_value]
                for i in range(len(input_idx)):
                    p.shapes[input_idx[i]][dim_idx[i]] = DynamicDim(
                        min_value, opt_value, max_value
                    )

            # Adjust the profile to satisfy the constraints
            for constraint_spec in tuning_config.constraint_specs:
                min_value = opt_value = max_value = constraint_spec.infer_shape(
                    p.get_opt_shapes()
                )
                p.shapes[constraint_spec.input_idx][constraint_spec.dim_idx] = (
                    DynamicDim(min_value, opt_value, max_value)
                )
            generated_profiles.append(p)
            logger.debug(f"[Autotuner]: generated profile: {p}")
        return generated_profiles

    @classmethod
    @lru_cache(maxsize=None)
    def _find_nearest_profile(
        cls, shapes: Tuple[torch.Size], tuning_config: TuningConfig
    ) -> Tuple:
        """Find the nearest optimization profile for given inputs
        User can define their own nearest profile generation method to reduce the host overhead.

        Args:
            shapes: Tuple of input tensor shapes
            tuning_config: Tuning configuration

        Return:
            Tuple: A tuple containing:
                - attributes: Tuple of runner attributes, sorted.
                - profile: Tuple of input tensor shapes
        """
        base_profile = list(list(shape) for shape in shapes)

        for spec in tuning_config.dynamic_tensor_specs:
            mapped_val = spec.map_to_tuning_buckets(
                base_profile[spec.input_idx[0]][spec.dim_idx[0]]
            )
            # Apply the same mapped bucket to all linked dimensions in this spec.
            for input_i, dim_i in zip(spec.input_idx, spec.dim_idx, strict=True):
                base_profile[input_i][dim_i] = mapped_val

        # associated dimensions dependent on other free dynamic dimensions, so assign -1 in the profile
        for constraint_spec in tuning_config.constraint_specs:
            base_profile[constraint_spec.input_idx][constraint_spec.dim_idx] = -1
        return tuple(tuple(shape) for shape in base_profile)

    @classmethod
    def _get_cache_key(
        cls,
        custom_op: str,
        runner: TunableRunner,
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
        extras: tuple = (),
    ) -> Tuple:
        return (
            custom_op,
            runner.__class__.__name__,
            hash(runner),
            cls._find_nearest_profile(input_shapes, tuning_config),
            extras,
        )

    def _create_tensor_like(
        self, origin_tensor: torch.Tensor, dims: List[Dim], initializer: Callable
    ) -> torch.Tensor:
        """Create a new tensor matching the properties of the original tensor.

        Args:
            origin_tensor (torch.Tensor): Template tensor to match
            dims (List[Dim]): List of dimensions for the new tensor

        Returns:
            New tensor with specified dimensions and matching properties

        Note:
            Creates a zero tensor with the same dtype and device as the original,
            but with dimensions specified by the dims parameter.
        """
        dtype = origin_tensor.dtype
        device = origin_tensor.device
        shapes = []
        for d in dims:
            if isinstance(d, StaticDim):
                shapes.append(d.val)
            else:
                # TODO: how to make sure the created Tensor has the min/max info
                assert isinstance(d, DynamicDim)
                shapes.append(d.opt)
        return initializer(shapes, dtype, device)

    def _prepare_input_tensors(
        self, profile: OptimizationProfile, inputs: List[Optional[torch.Tensor]]
    ) -> List[Optional[torch.Tensor]]:
        """Create tensors matching *profile* shapes; reuse static inputs as-is."""
        default_initializer = lambda shapes, dtype, device: (
            torch.rand(shapes, device=device) * 10 - 5
        ).to(dtype)
        tensors: List[Optional[torch.Tensor]] = []
        for i, p in enumerate(profile.shapes):
            if inputs[i] is None:
                # Some callers pass None for optional tensors (e.g. routing_logits
                # in non-routed MoE). Preserve None as-is.
                tensors.append(None)
            elif any(isinstance(d, DynamicDim) for d in p):
                tensor = self._create_tensor_like(
                    inputs[i],
                    p,
                    profile.tensor_initializers[i] or default_initializer,
                )
                tensors.append(tensor)
            else:
                tensors.append(inputs[i])
        return tensors

    def save_configs(self, path: str) -> None:
        """Save the current profiling cache to a JSON file.

        Serializes all cached (runner, tactic) results so they can be loaded
        later via ``load_configs()`` or ``autotune(cache=...)``, avoiding the
        need to re-run autotuning.

        When configs were previously loaded via ``load_configs()``, those
        entries are included in the output as well (with in-memory profiling
        results taking priority for overlapping keys). This ensures the saved
        file is always a complete, self-contained config.

        Note:
            This is called automatically on exit from
            ``with autotune(True, cache=path):``. Direct calls are only needed
            for advanced use cases.

        Args:
            path: File path to write the JSON config to.

        Example::

            # Preferred: use autotune(cache=...) for automatic save/load
            with autotune(True, cache="/path/to/config.json"):
                model(inputs)

            # Advanced: manual save after tuning
            with autotune(True):
                model(inputs)
            AutoTuner.get().save_configs("/path/to/config.json")
        """
        with self._lock:
            seq_at_snapshot = self._dirty_seq
            configs: Dict[str, Any] = {}

            # Include previously loaded file configs as a base
            for file_key, (runner_name, tactic) in self._file_configs.items():
                configs[file_key] = [runner_name, _tactic_to_json(tactic)]

            num_previous = len(configs)

            # Overlay in-memory profiling results (take priority over loaded configs)
            for cache_key, cache_value in self.profiling_cache.items():
                custom_op, runner_class_name, _runner_hash, profile, _extras = cache_key
                runner_id, tactic, _opt_profile = cache_value

                # Use hash-free key: (custom_op, runner_class_name, profile)
                file_key = str((custom_op, runner_class_name, profile))

                # Store runner class name (not positional index) for robustness
                tactic_json = _tactic_to_json(tactic)
                configs[file_key] = [runner_class_name, tactic_json]

        current_meta = _collect_metadata()

        # Re-read the file from disk and merge to reduce lost updates when
        # multiple processes save to the same path.  Entries from this
        # process take priority over on-disk entries.
        abs_path = os.path.abspath(path)
        original_metadata = None
        try:
            with open(abs_path, "r") as f:
                disk_configs = json.load(f)
            # Preserve the original _metadata from disk (the "created by" record).
            original_metadata = disk_configs.pop(_METADATA_KEY, None)
            disk_configs.update(configs)
            configs = disk_configs
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # file doesn't exist yet or is being replaced -- proceed with what we have

        # Compute after disk merge so the count reflects the actual file delta.
        num_new = len(configs) - num_previous

        # Atomic write: write to a temp file then replace the target.
        # This prevents readers from seeing a partially-written file and
        # guards against data loss if the process is killed mid-write.
        # The temp file is created in the same directory (dir=dir_name) so
        # that os.replace() is a same-filesystem rename, which is atomic.
        dir_name = os.path.dirname(abs_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name, suffix=".tmp", prefix=".autotuner_"
        )
        try:
            # Place metadata first in the output for readability.
            ordered = {}
            ordered[_METADATA_KEY] = original_metadata or current_meta
            for k in sorted(configs):
                ordered[k] = configs[k]

            with os.fdopen(fd, "w") as f:
                json.dump(ordered, f, indent=2)
            os.replace(tmp_path, abs_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        with self._lock:
            # Only clear dirty if no new results arrived during the save.
            if self._dirty_seq == seq_at_snapshot:
                self._dirty = False

        logger.info(
            f"[Autotuner]: Saved {len(configs)} configs to {path} "
            f"({num_new} new, {num_previous} from previous config)"
        )

    def load_configs(self, path: str) -> bool:
        """Load autotuner configs from a JSON file.

        Populates the internal config lookup table so that ``search_cache()``
        can return pre-tuned results without re-running autotuning.

        If the file contains ``_metadata`` that does not match the current
        environment (different FlashInfer version, GPU, cuBLAS, etc.), the
        entire cache is **skipped** to avoid silently using invalid tactics.

        Note:
            This is called automatically on entry to
            ``with autotune(cache=path):``. Direct calls are only needed
            for advanced use cases.

        Args:
            path: File path to the JSON config file (produced by
                ``save_configs()``).

        Returns:
            True if configs were loaded successfully, False if the cache was
            skipped due to an environment mismatch.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.

        Example::

            # Preferred: use autotune(cache=...) for automatic save/load
            with autotune(False, cache="/path/to/config.json"):
                model(inputs)

            # Advanced: manual load
            AutoTuner.get().load_configs("/path/to/config.json")
        """
        with open(path, "r") as f:
            configs = json.load(f)

        # Remove metadata keys so they don't end up in _file_configs.
        saved_meta = configs.pop(_METADATA_KEY, None)

        # If the cache was created in a different environment, skip it
        # entirely to avoid silently using invalid or suboptimal tactics.
        if saved_meta is not None:
            current_meta = _collect_metadata()
            mismatches = {
                k: (saved_meta.get(k), current_meta.get(k))
                for k in current_meta
                if saved_meta.get(k) not in (current_meta.get(k), "*")
            }
            if mismatches:
                details = ", ".join(
                    f"{k}: saved={old} vs current={new}"
                    for k, (old, new) in mismatches.items()
                )
                logger.warning(
                    f"[Autotuner]: Cache file {path} was created in a different "
                    f"environment ({details}). Ignoring cached configs. "
                    f"Results will not be saved to this file to avoid "
                    f"overwriting configs from a different environment. "
                    f"Use a different cache path to save configs for the "
                    f"current environment."
                )
                return False

        with self._lock:
            for key, value in configs.items():
                runner_name = value[0]
                tactic = _json_to_tactic(value[1])
                self._file_configs[key] = (runner_name, tactic)

        logger.info(f"[Autotuner]: Loaded {len(configs)} configs from {path}")
        return True

    def _prepare_input_tensors_with_batches(
        self,
        inputs: List[torch.Tensor],
        tuning_config: TuningConfig,
    ) -> List[List[torch.Tensor]]:
        """Create multiple input copies to flush the L2 cache between profiling iterations."""
        if not tuning_config.use_cold_l2_cache:
            return [inputs]

        one_buffer_bytes = sum(
            input.numel() * input.element_size()
            if isinstance(input, torch.Tensor)
            else 0
            for input in inputs
        )
        if one_buffer_bytes <= 0:
            logger.debug(
                "[Autotuner] No tensor inputs or zero-sized tensors; falling back to single-batch profiling."
            )
            return [inputs]

        num_buffers = self._get_l2_cache_size_in_bytes() * 3 // one_buffer_bytes + 1
        num_buffers = min(num_buffers, self.repeat + 1)

        inputs_list = [inputs]
        for _ in range(num_buffers - 1):
            inputs_list.append(
                list(t.clone() if isinstance(t, torch.Tensor) else t for t in inputs)
            )

        logger.debug(
            f"[Autotuner] use_cold_l2_cache={tuning_config.use_cold_l2_cache}, use {num_buffers} different tensors for profiling"
        )
        return inputs_list

    def clear_cache(self) -> None:
        """Clear the profiling cache and user-loaded file configs."""
        with self._lock:
            self.profiling_cache.clear()
            self._file_configs.clear()
            self._logged_file_hits.clear()
            self._dirty = False
            self._dirty_seq = 0

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()

    def _get_l2_cache_size_in_bytes(self, device_id: Optional[int] = None) -> int:
        """Return the L2 cache size in bytes for the given (or current) CUDA device."""
        if device_id is None:
            device_id = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device_id).L2_cache_size
