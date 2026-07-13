import contextlib
import copy
import functools
import importlib
import inspect
import itertools
import json
import os
import statistics
import tempfile
import threading
import time
import weakref

import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    TypeAlias,
    Tuple,
    Union,
)

import torch

from flashinfer.tllm_utils import delay_kernel
from flashinfer.utils import next_positive_power_of_2

from flashinfer.jit.core import logger
from flashinfer.version import __version__ as _flashinfer_version
from flashinfer.autotuner.initializers import (
    TensorInitializer,
    autotuner_initializer_rand_scaled,
)

# This version should be updated whenever the nvfp4_cutlass backend is changed,
# such as when new kernels or configs are added. In such cases, the tuning configs
# should also be updated. Currently, this process is manual, but it should be automated in the future.
_nvfp4_cutlass_version = "0.1"


# Re-read at every call (not at import) so harnesses that toggle this env
# var mid-process see the change.
def _is_value_aware_autotune() -> bool:
    return os.environ.get("FLASHINFER_DIST_AWARE_AUTOTUNE", "0") == "1"


_AUTOTUNE_DUMP_PATH = os.environ.get("FLASHINFER_AUTOTUNE_DUMP", "")


def _is_autotune_verbose_progress() -> bool:
    return os.environ.get("FLASHINFER_AUTOTUNE_VERBOSE_PROGRESS", "0") == "1"


def _is_autotune_nvtx_enabled() -> bool:
    """Return whether value-aware profiling should emit Nsight annotations."""
    return os.environ.get("FLASHINFER_AUTOTUNE_NVTX", "0") == "1"


@contextlib.contextmanager
def _autotune_nvtx_range(message: str):
    """Emit an opt-in host NVTX range around autotuner control work."""
    if _is_autotune_nvtx_enabled():
        torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        if _is_autotune_nvtx_enabled():
            torch.cuda.nvtx.range_pop()


def _stable_hash_key(obj: Any) -> Any:
    try:
        hash(obj)
        return obj
    except TypeError:
        return id(obj)


def _tactic_to_json(tactic: Any) -> Any:
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


def _json_to_tactic(val: Any) -> Any:
    """Convert a JSON-deserialized tactic value back to its original format.

    Lists are recursively converted to tuples so that compound tactics
    (e.g. CuteDSL's (tile_size, gemm1_tactic, gemm2_tactic)) are restored
    to their expected tuple form.
    """
    if isinstance(val, list):
        return tuple(_json_to_tactic(v) for v in val)
    return val


def _tactic_to_json_hashable(tactic):
    """Convert a tactic to a hashable form suitable for storing in a set.

    Like ``_tactic_to_json`` but returns tuples instead of lists so the
    result can be added to a ``set`` or used as a dict key.
    """
    if isinstance(tactic, (tuple, list)):
        return tuple(_tactic_to_json_hashable(v) for v in tactic)
    if hasattr(tactic, "__iter__") and not isinstance(tactic, (str, bytes, dict)):
        return tuple(_tactic_to_json_hashable(v) for v in tactic)
    if isinstance(tactic, bool):
        return tactic
    if isinstance(tactic, int):
        return int(tactic)
    return tactic


def round_to_nearest_bucket(
    x: int, buckets: Sequence[int], round_map: bool = False
) -> int:
    """Map *x* to the nearest bucket using floor or ceil semantics.

    Args:
        x: The value to map.
        buckets: Bucket values in **ascending** order.  Must not be empty.
        round_map: Rounding direction.

            * ``False`` (default) -- **floor**: return the largest bucket
              that is ``<= x``.  If *x* is smaller than every bucket, the
              smallest bucket is returned (clamped).
            * ``True`` -- **ceil**: return the smallest bucket that is
              ``>= x``.  If *x* is larger than every bucket, the largest
              bucket is returned (clamped).

    Returns:
        The matched bucket value.  Always one of the elements in *buckets*.

    Examples::

        >>> round_to_nearest_bucket(350, [100, 200, 500, 1000])
        200
        >>> round_to_nearest_bucket(350, [100, 200, 500, 1000], round_map=True)
        500
        >>> round_to_nearest_bucket(2000, [100, 200, 500, 1000], round_map=True)
        1000
    """
    if len(buckets) == 0:
        raise ValueError("buckets must be non-empty")
    if round_map:
        for b in buckets:
            if b >= x:
                return b
        return buckets[-1]
    else:
        for b in reversed(buckets):
            if b <= x:
                return b
        return buckets[0]


@functools.lru_cache(maxsize=16384)
def make_bucket_mapper(
    buckets: tuple[int, ...], round_map: bool = False
) -> Callable[[int], int]:
    """Create a mapper function for :class:`DynamicTensorSpec.map_to_tuning_buckets`.

    The returned callable maps any integer *x* to the nearest value in
    *buckets*, using floor or ceil semantics controlled by *round_map*.
    Duplicates in *buckets* are removed and values are sorted internally.

    Args:
        buckets: The set of allowed bucket values.
        round_map: If ``False`` (default) the mapper rounds **down** (floor);
            if ``True`` it rounds **up** (ceil).  In both cases the result is
            clamped to the bucket range -- see
            :func:`round_to_nearest_bucket` for details.

    Returns:
        A ``Callable[[int], int]`` suitable for passing as
        ``map_to_tuning_buckets`` to :class:`DynamicTensorSpec`.

    Examples::

        >>> mapper = make_bucket_mapper((100, 200, 500, 1000), round_map=False)
        >>> mapper(350)
        200
        >>> mapper_up = make_bucket_mapper((100, 200, 500, 1000), round_map=True)
        >>> mapper_up(350)
        500
    """
    if len(buckets) == 0:
        raise ValueError("buckets must be non-empty")
    sorted_buckets = tuple(sorted(set(buckets)))

    def _mapper(x: int) -> int:
        return round_to_nearest_bucket(x, sorted_buckets, round_map)

    return _mapper


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


def _collect_metadata() -> dict[str, str]:
    """Collect environment metadata that can affect tactic-to-kernel mappings.

    Tactics in flashinfer's autotune cache are stored as plan indices into
    cuDNN's ``policy=ALL`` plan list (or backend-internal kernel ids for
    other backends).  Anything that can shuffle that ordering must be
    captured here so ``load_configs`` rejects a cache that no longer
    matches the runtime environment.

    Specifically tracked:

        * ``flashinfer_version``  -- our own bucketing / ordering changes
        * ``cuda_version``        -- CUDA driver/runtime ABI
        * ``cublas_version``      -- cuBLAS plan availability inside cuDNN
        * ``cudnn_version``       -- cuDNN **backend** version
        * ``cudnn_frontend_version`` -- cuDNN-frontend Python wrapper
                                        version (independent of backend);
                                        plan_index ordering can change
                                        when only the frontend is updated
                                        but the backend is not
        * ``gpu``                 -- device name (different SM may have
                                     different available engines)
    """
    meta: dict[str, str] = {}
    meta["flashinfer_version"] = _flashinfer_version
    meta["cuda_version"] = getattr(torch.version, "cuda", None) or "unknown"
    meta["cublas_version"] = _get_cublas_version()
    try:
        meta["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception:
        meta["cudnn_version"] = "unknown"
    try:
        # cudnn-frontend is an optional dependency; failing to import is
        # not fatal -- we just record "unknown" and let the metadata
        # check fall back to the backend-version field.
        import cudnn as _cudnn_frontend

        meta["cudnn_frontend_version"] = getattr(
            _cudnn_frontend, "__version__", "unknown"
        )
    except Exception:
        meta["cudnn_frontend_version"] = "unknown"
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
    A specification for dynamic tensor profiling.

    This describes both shape buckets and, optionally, representative value
    buckets for tensors whose content distribution affects kernel performance.
    Args:
        input_idx: A tuple of the indices of the input tensors.
        dim_idx: A tuple of the indices of the dimensions to tune.
            The length of input_idx and dim_idx must be the same.
            For every tensor mapped to the input_idx, their dimension mapped to the dim_idx must be the same.
        gen_tuning_buckets: A tuple of values to try or a function generating values.
        map_to_tuning_buckets: A function to map dimensions to valid values during inference.
        tensor_initializers: A list of functions to initialize the tensors.
        value_specs: Optional value-bucket profiling rules attached to this
            dynamic tensor spec.
    """

    input_idx: tuple[int, ...]
    dim_idx: tuple[int, ...]
    gen_tuning_buckets: tuple[int, ...] | Callable[[int], Iterable[int]]
    map_to_tuning_buckets: Callable[[int], int]
    tensor_initializers: Sequence[TensorInitializer] | None = field(
        default_factory=lambda: None
    )
    value_specs: Tuple["DynamicValueSpec", ...] = ()

    def __post_init__(self):
        # Set default tensor_initializers if not provided
        if self.tensor_initializers is None:
            self.tensor_initializers = [
                autotuner_initializer_rand_scaled for _ in range(len(self.input_idx))
            ]
        self.value_specs = tuple(self.value_specs or ())

    def __hash__(self) -> int:
        # FIXME: currently not hashing tensor_initializers
        return hash(
            (
                self.input_idx,
                self.dim_idx,
                # For gen_tuning_buckets, only hash if it's a tuple, otherwise hash its id
                self.gen_tuning_buckets
                if isinstance(self.gen_tuning_buckets, tuple)
                else id(self.gen_tuning_buckets),
                id(self.map_to_tuning_buckets),
                self.value_specs,
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


class ValueSampleStager(ABC):
    """Stage non-isomorphic value samples into captured graph inputs."""

    @abstractmethod
    def prepare(
        self,
        input_batches: List[List[Optional[torch.Tensor]]],
        input_idx: int,
    ) -> Any:
        """Allocate persistent staging state before graph capture."""

    @abstractmethod
    def stage(
        self,
        source: torch.Tensor,
        input_batches: List[List[Optional[torch.Tensor]]],
        input_idx: int,
        state: Any,
    ) -> Tuple[int, int]:
        """Stage one sample and return ``(copies, host_to_device_copies)``."""


@dataclass(slots=True)
class DynamicValueSpec:
    """Specification for value-bucket profiling within a DynamicTensorSpec.

    This varies the content/distribution of a profiled tensor while reusing the
    shape buckets and tensor initializers owned by the parent DynamicTensorSpec.

    Args:
        input_idx: Index of the input tensor whose values matter.
        gen_value_buckets: Bucket IDs to profile during tuning (tuple, a zero-argument
            callable, or a callable that accepts the generated OptimizationProfile).
        map_to_value_bucket: Maps actual runtime inputs to a bucket ID at inference time.
            Must be cheap (~microseconds). It may accept either:
            - ``tensor``; or
            - ``tensor, inputs, kwargs`` for methods that need light call context.
        tensor_value_generator: Generates representative tensor values for a given
            bucket during profiling. It may accept either:
            - ``bucket_id, profiled_tensor``; or
            - ``bucket_id, profiled_tensor, original_tensor``; or
            - ``bucket_id, profiled_tensor, original_tensor, inputs``.
        sample_value_generator: Optional generator used for every outer value
            realization. It follows the ``tensor_value_generator`` calling
            convention, with an optional fifth ``sample_index`` argument, and
            may return staging data with a different shape.
        sample_value_stager: Optional object that stages a sample generator's
            result into the captured graph inputs.
    """

    input_idx: int
    gen_value_buckets: Union[Tuple[int, ...], Callable]
    map_to_value_bucket: Callable
    tensor_value_generator: Callable
    sample_value_generator: Optional[Callable] = None
    sample_value_stager: Optional[ValueSampleStager] = None

    def __hash__(self):
        return hash(
            (
                self.input_idx,
                self.gen_value_buckets
                if isinstance(self.gen_value_buckets, tuple)
                else _stable_hash_key(self.gen_value_buckets),
                _stable_hash_key(self.map_to_value_bucket),
                _stable_hash_key(self.tensor_value_generator),
                _stable_hash_key(self.sample_value_generator),
                _stable_hash_key(self.sample_value_stager),
            )
        )

    def __eq__(self, other):
        return isinstance(other, DynamicValueSpec) and hash(self) == hash(other)


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
                ...             input_idx=(0,),
                ...             dim_idx=(1,),
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
        value_sample_count: Optional callback that receives the complete value-bucket
            tuple and the default outer sample count. It may reduce the number of
            generated value realizations for deterministic buckets.
    """

    dynamic_tensor_specs: tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: tuple[ConstraintSpec, ...] = ()
    use_cold_l2_cache: bool = False
    use_cuda_graph: bool = False
    value_sample_count: Optional[Callable[[Tuple[Any, ...], int], int]] = None
    # Optional explicit value-bucket tuple representing the unmodified/default
    # input profile.  Its winner is also published under the ordinary
    # shape-only cache key, allowing one value-aware sweep to serve eager and
    # value-aware execution without conflating the two cache contracts.
    default_value_buckets: Optional[Tuple[Any, ...]] = None
    # Optional callback invoked once per profile bucket, after dynamic
    # tensors are synthesized but before the per-tactic profile loop.
    # Receives the full list of tensors and returns a (possibly modified)
    # list. Use this to inject a deterministic, realistic distribution
    # for inputs whose default tensor_initializer would be random
    # (e.g. token_selected_experts in MoE workloads).
    inputs_pre_hook: Callable | None = None


@dataclass(frozen=True)
class StaticDim:
    val: int


@dataclass(frozen=True)
class DynamicDim:
    """Range of one dimension"""

    min: int
    opt: int
    max: int


Dim = DynamicDim | StaticDim


def _get_opt(dim: Dim) -> int:
    if isinstance(dim, DynamicDim):
        return dim.opt
    else:
        return dim.val


@dataclass
class OptimizationProfile:
    """Ranges of all tensors, all dimension"""

    shapes: list[list[Dim]]
    tensor_initializers: list[TensorInitializer | None]
    value_buckets: Tuple = ()

    def get_hash_key(self):
        return (self.get_opt_shapes(), self.value_buckets)

    def get_opt_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Only the opt shapes are considered as hash key"""
        return tuple(tuple(_get_opt(d) for d in t) for t in self.shapes)


# TODO: can/shall we use the torch builtin FakeTensor class?
@dataclass
class FakeTensor:
    dtype: torch.dtype
    device: torch.device
    shape: list[Dim]


class TunableRunner(ABC):
    @abstractmethod
    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[int]:
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

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[Any, ...]:
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

    def get_tactic_groups(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile
    ) -> Optional[List[List[Any]]]:
        """Return independently tunable tactic groups, or ``None`` for exhaustive search.

        Each group is profiled independently. The tuner passes the fastest tactic
        from every group to :meth:`compose_tactics` and profiles the composed tactic
        once before caching it.
        """
        return None

    def compose_tactics(
        self,
        group_winners: List[Any],
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> Any:
        """Compose independently selected group winners into an executable tactic."""
        raise NotImplementedError

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(
        self,
        inputs: list[torch.Tensor],
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

    def __hash__(self) -> int:
        # Subclasses may carry unhashable instance attributes (e.g. _algo_cache
        # dicts added by GEMM runners). Skip *_cache fields entirely and fall
        # back to id() for any remaining unhashable values.
        hashable_vals: list[Any] = []
        for k, v in self.__dict__.items():
            if k.endswith("_cache"):
                continue
            try:
                hash(v)
                hashable_vals.append(v)
            except TypeError:
                hashable_vals.append(id(v))
        return hash(tuple(hashable_vals))


@contextlib.contextmanager
def autotune(
    tune_mode: bool = True,
    cache: str | None = None,
    tuning_buckets: tuple[int, ...] | None = None,
    round_up: bool | None = None,
    skip_ops: str | set[str] | None = None,
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

        skip_ops: Optional set of ``custom_op`` names to exclude from
            autotuning.  Operations whose ``custom_op`` string matches an
            entry in this set will skip profiling entirely and use their
            fallback (heuristic) tactic instead.  This is useful when a
            framework runs a single dummy forward pass inside
            ``autotune()`` but wants to avoid the compilation cost of
            autotuning specific ops whose heuristics are already
            near-optimal.  For example,
            ``skip_ops={"fp4_gemm"}`` skips autotuning for ``mm_fp4``
            while still tuning MoE and other operations.
            Nested contexts **union** their skip sets: an inner
            ``autotune(skip_ops={"B"})`` inside an outer
            ``autotune(skip_ops={"A"})`` skips both ``"A"`` and ``"B"``.
            Common op names: ``"fp4_gemm"``, ``"bf16_gemm"``,
            ``"fp8_gemm"``, ``"mxfp8_gemm"``.

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

        # Skip autotuning for specific ops (use heuristic fallback)
        with autotune(True, skip_ops={"fp4_gemm"}):
            model(inputs)  # mm_fp4 uses heuristic, other ops are autotuned
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

    # Push skip_ops onto per-thread stack.  Each entry is the cumulative
    # union so that _effective_skip_ops is an O(1) read from the top.
    skip_ops_stack = tuner._get_skip_ops_stack()
    if skip_ops is not None:
        skip_ops_set = {skip_ops} if isinstance(skip_ops, str) else skip_ops
        current = skip_ops_stack[-1] if skip_ops_stack else frozenset()
        skip_ops_stack.append(current | frozenset(skip_ops_set))

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
            value_status = "enabled" if _is_value_aware_autotune() else "disabled"
            logger.info(
                f"[Autotuner]: Autotuning process starts (value-aware: {value_status}) ..."
            )
    except BaseException:
        if pushed:
            override_stack.pop()
        if skip_ops is not None:
            skip_ops_stack.pop()
        raise

    try:
        yield
    finally:
        with tuner._lock:
            if tune_mode:
                tuner._active_tuning_contexts -= 1
            tuner.is_tuning_mode = tuner._active_tuning_contexts > 0

        # Pop the overrides we pushed (thread-local, no lock needed).
        if pushed:
            override_stack.pop()
        if skip_ops is not None:
            skip_ops_stack.pop()

        if autotune_enabled:
            for cb in tuner._post_autotune_callbacks:
                try:
                    cb()
                except Exception as e:
                    logger.warning(f"[Autotuner] post-autotune callback failed: {e}")
            tuner._post_autotune_callbacks.clear()
            logger.info("[Autotuner]: Autotuning process ends")

        # Save configs on exit when tuning with a cache path,
        # but only if new profiling results were added this session
        # and the cache file was valid (no environment mismatch).
        if cache is not None and cache_valid and tune_mode and tuner._dirty:
            tuner.save_configs(cache)


# Thread-local "currently inside the per-tactic measurement window" flag.
# Set/cleared by ``_profile_measurement_scope`` and queried by
# ``is_in_profile_measurement``.  Distinct from
# ``AutoTuner.is_tuning_mode``, which is True for the entire ``autotune(True)``
# context (cache hits, prep calls, post-``choose_one`` runs, and concurrent
# threads inclusive); ``is_in_profile_measurement`` is True only on the
# specific thread that is actively timing a tactic, and only during the
# warmup + measurement window inside ``_profile_single_kernel``.
_profile_measurement_thread_local = threading.local()


@contextlib.contextmanager
def _profile_measurement_scope():
    """Mark the calling thread as inside the autotuner's per-tactic
    measurement window.  Nested entries are honored via prev/restore so
    correctness doesn't depend on a single entry path.
    """
    prev = getattr(_profile_measurement_thread_local, "active", False)
    _profile_measurement_thread_local.active = True
    try:
        yield
    finally:
        _profile_measurement_thread_local.active = prev


def is_in_profile_measurement() -> bool:
    """Return True iff the calling thread is currently inside the
    autotuner's per-tactic measurement window (warmup + timed run inside
    ``AutoTuner._profile_single_kernel``).

    This is *narrower* than ``AutoTuner.get().is_tuning_mode``:

    - ``is_tuning_mode`` is True for the entire ``autotune(True)`` context,
      including cache lookups, ``do_preparation`` calls, the final invocation
      after ``choose_one`` returns, and any concurrent threads' work.
    - ``is_in_profile_measurement`` is True only on the calling thread, and
      only during the actual measurement window of a single tactic.

    Wrappers that need to bypass per-call optimizations (e.g. CUDA-graph
    preallocated buffers sized for a specific construction-time tile) so
    that the autotuner's tactic comparison is unbiased should consult
    ``is_in_profile_measurement()`` rather than ``is_tuning_mode``.
    """
    return getattr(_profile_measurement_thread_local, "active", False)


_tune_process_group: Optional["torch.distributed.ProcessGroup"] = None


def set_autotune_process_group(
    group: Optional["torch.distributed.ProcessGroup"],
) -> None:
    """All-reduce (mean) per-tactic profile timings across ``group`` so every
    rank's ``argmin`` picks the same tactic.

    Without it, GPU timing noise makes ranks diverge on tactic choice, which
    deadlocks NCCL symmetric-memory allocation (``NCCL_WIN_COLL_SYMMETRIC``).
    Prefer a CPU (``gloo``) subgroup; a NCCL group also works. ``None``
    disables (default); not thread-safe.

    Caller contract: every rank must enter ``_profile_single_kernel`` the same
    number of times in the same order, or the reduction itself deadlocks. Across
    ranks that requires identical: ``get_valid_tactics`` and shape buckets;
    ``skip_ops`` (a skipped op returns before the tactic loop, doing zero
    reduces); and ``profiling_cache`` / loaded ``autotune(cache=...)`` at entry
    (a cache hit skips that profile's reduce) -- so set the group from the first
    ``choose_one`` with identical (ideally empty) starting caches. Residual: a
    per-rank OOM *outside* ``_profile_single_kernel`` (input synthesis /
    ``do_preparation``) can still early-return and desync.

    Example::

        set_autotune_process_group(cpu_group)  # gloo subgroup
        try:
            with autotune(True):
                model(inputs)
        finally:
            set_autotune_process_group(None)
    """
    global _tune_process_group
    _tune_process_group = group


def get_autotune_process_group() -> Optional["torch.distributed.ProcessGroup"]:
    """Return the process group previously passed to ``set_autotune_process_group``."""
    return _tune_process_group


@dataclass(frozen=True)
class ProfilingCacheKey:
    """Immutable key identifying a profiled (op, runner, shape) combination.

    ``runner_hash`` is excluded from the file key because it captures runtime
    identity (address / id) and would differ across processes for the same
    runner class.  ``nearest_profile`` replaces raw input shapes so that
    shapes that fall in the same tuning bucket share a single cache entry.
    """

    custom_op: str
    runner_class_name: str
    runner_hash: int
    nearest_profile: tuple[tuple[int, ...], ...]
    extras: tuple[Any, ...]

    @property
    def file_key(self) -> str:
        """Stable string key suitable for on-disk serialisation."""
        return str(
            (self.custom_op, self.runner_class_name, self.nearest_profile, self.extras)
        )


class _ValueAwareInputArena:
    """Reuse cold-L2 lane storage across value-aware token-shape profiles.

    Each lane owns distinct tensor storage, preserving the cold-L2 rotation.
    Dynamic profile tensors are allocated once at their maximum shape and
    narrower profiles bind views into the same lane-local storage.

    Example:
        arena = _ValueAwareInputArena.create(tuner, profiles, inputs, config)
        for input_batch in arena.batches_for(profile):
            measure(input_batch)
    """

    def __init__(
        self,
        input_batches: List[List[Optional[torch.Tensor]]],
        dynamic_input_indices: Set[int],
    ) -> None:
        self.input_batches = input_batches
        self.dynamic_input_indices = dynamic_input_indices
        self._arena_bytes = sum(
            tensor.numel() * tensor.element_size()
            for batch in input_batches
            for tensor in batch
            if isinstance(tensor, torch.Tensor)
        )

    @classmethod
    def create(
        cls,
        tuner: "AutoTuner",
        profiles: List[OptimizationProfile],
        inputs: List[Optional[torch.Tensor]],
        tuning_config: Optional[TuningConfig] = None,
        *,
        lane_count: Optional[int] = None,
    ) -> "_ValueAwareInputArena":
        """Allocate maximum-shape storage and return reusable cold-L2 lanes."""
        if not profiles:
            raise ValueError("value-aware input arena requires at least one profile")

        dynamic_input_indices = {
            input_idx
            for input_idx in range(len(inputs))
            if any(
                any(isinstance(dim, DynamicDim) for dim in profile.shapes[input_idx])
                for profile in profiles
            )
        }
        if not dynamic_input_indices:
            raise ValueError("value-aware input arena requires dynamic tensor inputs")

        max_shapes = [
            tuple(
                max(
                    profile.get_opt_shapes()[input_idx][dim_idx] for profile in profiles
                )
                for dim_idx in range(len(profiles[0].shapes[input_idx]))
            )
            for input_idx in range(len(inputs))
        ]
        max_inputs: List[Optional[torch.Tensor]] = []
        default_initializer = lambda shapes, dtype, device: (
            torch.rand(shapes, device=device) * 10 - 5
        ).to(dtype)
        for input_idx, input_tensor in enumerate(inputs):
            if input_tensor is None:
                max_inputs.append(None)
            elif input_idx not in dynamic_input_indices:
                max_inputs.append(input_tensor)
            else:
                initializer = (
                    profiles[0].tensor_initializers[input_idx] or default_initializer
                )
                max_inputs.append(
                    tuner._create_tensor_like(
                        input_tensor,
                        [StaticDim(size) for size in max_shapes[input_idx]],
                        initializer,
                    )
                )

        if lane_count is None:
            max_buffer_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in max_inputs
                if isinstance(tensor, torch.Tensor)
            )
            lane_count = tuner._get_cold_l2_buffer_count(
                max_buffer_bytes,
                True if tuning_config is None else tuning_config.use_cold_l2_cache,
            )

        input_batches = [max_inputs]
        for _ in range(lane_count - 1):
            input_batches.append(
                [
                    tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
                    for tensor in max_inputs
                ]
            )
        return cls(input_batches, dynamic_input_indices)

    @staticmethod
    def _view_for_shape(tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        if tuple(tensor.shape) == shape:
            return tensor
        if tensor.ndim != len(shape) or any(
            size > tensor.shape[dim] for dim, size in enumerate(shape)
        ):
            raise ValueError(
                f"arena tensor shape {tuple(tensor.shape)} cannot serve profile shape {shape}"
            )
        return tensor[tuple(slice(0, size) for size in shape)]

    def batches_for(
        self, profile: OptimizationProfile
    ) -> List[List[Optional[torch.Tensor]]]:
        """Return profile-shaped views of every cold-L2 lane."""
        profile_shapes = profile.get_opt_shapes()
        return [
            [
                (
                    self._view_for_shape(tensor, profile_shapes[input_idx])
                    if input_idx in self.dynamic_input_indices
                    and isinstance(tensor, torch.Tensor)
                    else tensor
                )
                for input_idx, tensor in enumerate(batch)
            ]
            for batch in self.input_batches
        ]

    def diagnostics(self) -> Dict[str, Any]:
        """Describe arena ownership for per-candidate profiling records."""
        return {
            "input_arena_reused": True,
            "cold_l2_lane_count": len(self.input_batches),
            "cold_l2_arena_bytes": self._arena_bytes,
            "mutable_route_h2d_copies": 0,
            "arena_fallback_reason": "",
        }


class _GraphProfileSession:
    """Own one CUDA graph and stable input storage for a tactic's value samples."""

    def __init__(
        self,
        tuner: "AutoTuner",
        runner: TunableRunner,
        tactic: Any,
        tuning_config: TuningConfig,
        input_batches: List[List[torch.Tensor]],
        mutable_input_indices: Tuple[int, ...],
        graph: torch.cuda.CUDAGraph,
        graph_pool: object,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        stream: torch.cuda.Stream,
        setup_host_time_s: float,
        arena_diagnostics: Optional[Dict[str, Any]] = None,
        value_stagers: Optional[Dict[int, ValueSampleStager]] = None,
        value_stager_states: Optional[Dict[int, Any]] = None,
        **kwargs,
    ) -> None:
        self.tuner = tuner
        self.runner = runner
        self.tactic = tactic
        self.tuning_config = tuning_config
        self.input_batches = input_batches
        self.mutable_input_indices = tuple(dict.fromkeys(mutable_input_indices))
        self.graph = graph
        self.graph_pool = graph_pool
        self.start = start
        self.end = end
        self.stream = stream
        self.setup_host_time_s = setup_host_time_s
        self.kwargs = kwargs
        self.replay_count = 0
        self.staging_copy_count = 0
        self.arena_diagnostics = arena_diagnostics or {
            "input_arena_reused": False,
            "cold_l2_lane_count": len(input_batches),
            "cold_l2_arena_bytes": 0,
            "mutable_route_h2d_copies": 0,
            "arena_fallback_reason": "",
        }
        self.value_stagers = value_stagers or {}
        self.value_stager_states = value_stager_states or {}
        self.mutable_route_h2d_copy_count = 0

    @classmethod
    def capture(
        cls,
        tuner: "AutoTuner",
        runner: TunableRunner,
        inputs: List[torch.Tensor],
        initial_value_inputs: List[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        mutable_input_indices: Tuple[int, ...],
        input_batches: Optional[List[List[Optional[torch.Tensor]]]] = None,
        arena_diagnostics: Optional[Dict[str, Any]] = None,
        value_stagers: Optional[Dict[int, ValueSampleStager]] = None,
        **kwargs,
    ) -> "_GraphProfileSession":
        """Allocate, warm up, and capture one tactic-local graph session."""
        setup_start = time.perf_counter()
        if input_batches is None:
            input_batches = tuner._prepare_input_tensors_with_batches(
                inputs, tuning_config
            )
        value_stager_states = {
            input_idx: stager.prepare(input_batches, input_idx)
            for input_idx, stager in (value_stagers or {}).items()
        }
        stream = torch.cuda.current_stream()
        graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        session = cls(
            tuner,
            runner,
            tactic,
            tuning_config,
            input_batches,
            mutable_input_indices,
            graph,
            graph_pool,
            start,
            end,
            stream,
            0.0,
            arena_diagnostics,
            value_stagers,
            value_stager_states,
            **kwargs,
        )

        with _profile_measurement_scope(), torch.cuda.stream(stream):
            with _autotune_nvtx_range("graph-session:stage-initial"):
                session._stage_values(initial_value_inputs)
            with _autotune_nvtx_range(f"graph-session:warmup, calls={tuner.warmup}"):
                for _ in range(tuner.warmup):
                    runner(input_batches[-1], tactic=tactic, **kwargs)
            stream.synchronize()
            with (
                _autotune_nvtx_range(
                    f"graph-session:capture, replays-per-graph={tuner.repeat}"
                ),
                torch.cuda.graph(graph, pool=graph_pool),
            ):
                session._run_kernels()
            stream.synchronize()

        session.setup_host_time_s = time.perf_counter() - setup_start
        return session

    def _stage_values(self, value_inputs: List[torch.Tensor]) -> None:
        """Copy mutable value-spec tensors into every stable cold-L2 batch."""
        for input_idx in self.mutable_input_indices:
            source = value_inputs[input_idx]
            if not isinstance(source, torch.Tensor):
                continue
            stager = self.value_stagers.get(input_idx)
            if stager is not None:
                copies, h2d_copies = stager.stage(
                    source,
                    self.input_batches,
                    input_idx,
                    self.value_stager_states[input_idx],
                )
                self.staging_copy_count += copies
                self.mutable_route_h2d_copy_count += h2d_copies
                continue
            for batch in self.input_batches:
                destination = batch[input_idx]
                if not isinstance(destination, torch.Tensor):
                    raise TypeError(
                        f"value input {input_idx} is not a tensor in a graph profile batch"
                    )
                destination.copy_(source)
                self.staging_copy_count += 1

    def _run_kernels(self) -> None:
        for repeat_idx in range(self.tuner.repeat):
            self.runner(
                self.input_batches[repeat_idx % len(self.input_batches)],
                tactic=self.tactic,
                **self.kwargs,
            )

    def measure(self, value_inputs: List[torch.Tensor]) -> float:
        """Stage one realization before timing and replay the captured graph."""
        with _profile_measurement_scope(), torch.cuda.stream(self.stream):
            with _autotune_nvtx_range("graph-session:stage-sample"):
                self._stage_values(value_inputs)
            with _autotune_nvtx_range(
                f"graph-session:thermal-cooldown, us={self.tuner._CUDA_GRAPH_DELAY_MICRO_SECS}"
            ):
                delay_kernel(self.tuner._CUDA_GRAPH_DELAY_MICRO_SECS)
            with _autotune_nvtx_range("graph-session:graph-replay"):
                self.start.record()
                self.graph.replay()
                self.end.record()
            self.stream.synchronize()
        self.replay_count += 1
        return self.start.elapsed_time(self.end) / self.tuner.repeat

    def diagnostics(self) -> Dict[str, Any]:
        """Return optional dump fields without retaining the session past a tactic."""
        diagnostics = {
            "value_sample_count": self.replay_count,
            "graph_captures": 1,
            "graph_replays": self.replay_count,
            "staging_copies": self.staging_copy_count,
            "staging_before_timing": True,
            "graph_setup_host_time_ms": self.setup_host_time_s * 1000.0,
            **self.arena_diagnostics,
        }
        diagnostics["mutable_route_h2d_copies"] = self.mutable_route_h2d_copy_count
        return diagnostics


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
    cache_miss_config_collection: dict[str, set[OptimizationProfile]] = field(
        default_factory=dict[str, set[OptimizationProfile]]
    )
    failed_profiling_count: dict[str, set[ProfilingCacheKey]] = field(
        default_factory=dict[str, set[ProfilingCacheKey]]
    )
    tuned_op_total_configs: dict[str, int] = field(default_factory=dict[str, int])
    tuned_op_successful_configs: dict[str, int] = field(default_factory=dict[str, int])
    # Maps "custom_op::RunnerClass" to sets of tactic values that failed.
    # Used by the offline blocklist generator to extract per-tactic pass/fail.
    failed_tactics: dict[str, set[Any]] = field(default_factory=dict[str, set[Any]])

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


@functools.lru_cache(maxsize=16384)
def load_from_file(file_key: str) -> tuple[bool, int, int, None]:
    module_name = get_config_path(is_module=True)
    try:
        module = importlib.import_module(module_name)
        best_configs = module.best_configs
    except (ImportError, AttributeError):
        best_configs = None
    if best_configs is not None:
        if file_key in best_configs:
            logger.info(f"[Autotuner]: Loading configs for {file_key} from file.")
            return True, best_configs[file_key][0], best_configs[file_key][1], None
    logger.info(
        f"[Autotuner]: Loading configs for {file_key} from file failed; Using default configs instead."
    )
    return False, 0, -1, None


# A single entry pushed by the autotune() context manager onto the per-thread
# override stack.  First element is the sorted tuple of tuning bucket values
# (None means "inherit from enclosing context / use default"), second element
# is the round_up flag that controls whether runtime shapes are rounded up to
# the nearest bucket.
Override: TypeAlias = tuple[tuple[int, ...] | None, bool]

# Per-thread stack of Override entries.  The top of the stack (index -1) is
# the currently active override; inner autotune() contexts push, outer ones pop.
OverrideStack: TypeAlias = list[Override]


class _OverrideLocal(threading.local):
    stack: OverrideStack


class _SkipOpsLocal(threading.local):
    stack: list[frozenset[str]]


class AutoTuner:
    """AutoTuner for optimizing TensorRT-LLM operations.

    This class handles automatic performance tuning of tensor operations by profiling
    different implementations and caching the best performing configurations.

    Args:
        warmup (int): Number of warmup iterations before profiling (default: 3)
        repeat (int): Number of profiling iterations for averaging (default: 10)
        stream_delay_micro_secs (int): Delay on CUDA stream before the profiled kernel runs in microseconds (default: 5000)
    """

    _CUDA_GRAPH_DELAY_MICRO_SECS = 100
    _instance = None
    _class_lock = threading.Lock()

    def __init__(
        self, warmup: int = 3, repeat: int = 10, stream_delay_micro_secs: int = 5000
    ):
        # Allow env-var override so frameworks (e.g. vLLM) can do a fast
        # verification-only autotune without code changes. FLASHINFER_AUTOTUNE_WARMUP=1
        # FLASHINFER_AUTOTUNE_REPEAT=1 makes profiling roughly 6x faster at the
        # cost of noisier measurements.
        warmup = int(os.environ.get("FLASHINFER_AUTOTUNE_WARMUP", warmup))
        repeat = int(os.environ.get("FLASHINFER_AUTOTUNE_REPEAT", repeat))
        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs
        self.profiling_cache: dict[
            ProfilingCacheKey, tuple[int, Any, OptimizationProfile]
        ] = {}
        self.profiling_time_cache: dict[ProfilingCacheKey, float] = {}
        # Retain every measured tactic, not only the winner. Post-selection
        # policies can compare fixed tactics on identical value profiles
        # without launching a second profiling sweep.
        self.profiling_tactic_time_cache: dict[
            tuple[ProfilingCacheKey, Any], float
        ] = {}
        self.last_selection: Optional[dict[str, Any]] = None
        self.is_tuning_mode = False
        self._active_tuning_contexts = 0
        self._post_autotune_callbacks: list = []

        # Reentrant lock protecting all mutable state on this instance.
        # RLock is used because choose_one() calls search_cache() internally.
        self._lock = threading.RLock()

        # Add statistics tracking
        self.stats = AutoTunerStatistics()

        self.profiling_debug = True
        self._profiling_records: List[Dict[str, Any]] = []

        # Offline tactics blocklist (loaded via env var or explicit call).
        # Lazy import to avoid circular dependency (tactics_blocklist
        # imports _METADATA_KEY / _collect_metadata from this module).
        from flashinfer.tactics_blocklist import TacticsBlocklist

        self._blocklist = TacticsBlocklist()
        _bl_path = os.environ.get("FLASHINFER_TACTICS_BLOCKLIST")
        if _bl_path:
            if os.path.isfile(_bl_path):
                self._blocklist.load(_bl_path)
            else:
                logger.warning(
                    f"[Autotuner]: Tactics blocklist file not found at {_bl_path}"
                )

        # User-loaded configs from JSON files (populated by load_configs or autotune(cache=))
        self._file_configs: dict[str, tuple[str, Any]] = {}
        # Track which file config keys have been logged (to avoid per-call spam)
        self._logged_file_hits: set[tuple[str, str]] = set()
        # Track which (custom_op, profile-shape signature) pairs have already
        # produced an out-of-range cache-miss warning, so we warn at most
        # once per unique missing shape.
        self._logged_cache_miss_oor: set[tuple[str, tuple[tuple[int, ...], ...]]] = (
            set()
        )
        # Set when new profiling results are added; cleared on save.
        self._dirty = False
        self._dirty_seq = 0

        # Per-thread stack of (tuning_buckets, round_up) overrides set by
        # autotune() context manager.  Using threading.local ensures concurrent
        # autotune() contexts on different threads don't clobber each other.
        self._override_local: _OverrideLocal = _OverrideLocal()
        # Per-thread stack of frozenset[str] for skip_ops overrides.
        self._skip_ops_local: _SkipOpsLocal = _SkipOpsLocal()
        # Cache overridden TuningConfig objects to keep stable object identity
        # for _find_nearest_profile's LRU cache.
        # Two-level: WeakKeyDictionary[TuningConfig, Dict[(buckets, round_up), TuningConfig]]
        # keyed by identity so configs differing only in tensor_initializers
        # (whose __hash__ is the same) don't collide.
        self._override_config_cache: weakref.WeakKeyDictionary[
            TuningConfig, dict[tuple[tuple[int, ...] | None, bool], TuningConfig]
        ] = weakref.WeakKeyDictionary()

    def _dump_profiling_records(self, custom_op: str):
        """Dump collected profiling records to CSV when FLASHINFER_AUTOTUNE_DUMP is set."""
        if not _AUTOTUNE_DUMP_PATH or not self._profiling_records:
            return
        import csv

        path = _AUTOTUNE_DUMP_PATH
        file_exists = os.path.exists(path)
        fieldnames = [
            "op",
            "runner",
            "num_tokens",
            "value_buckets",
            "tactic",
            "time_ms",
            "value_time_min_ms",
            "value_time_median_ms",
            "value_time_max_ms",
            "selection_stage",
            "is_best",
            "value_sample_count",
            "graph_captures",
            "graph_replays",
            "staging_copies",
            "staging_before_timing",
            "graph_setup_host_time_ms",
            "input_arena_reused",
            "cold_l2_lane_count",
            "cold_l2_arena_bytes",
            "mutable_route_h2d_copies",
            "arena_fallback_reason",
        ]
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._profiling_records)
        logger.info(
            f"[Autotuner]: Dumped {len(self._profiling_records)} profiling records to {path}"
        )
        self._profiling_records.clear()

    def _get_override_stack(self) -> OverrideStack:
        """Return the per-thread override stack, creating it on first access."""
        local = self._override_local
        if not hasattr(local, "stack"):
            local.stack = OverrideStack()
        return local.stack

    def _get_skip_ops_stack(self) -> list[frozenset[str]]:
        """Return the per-thread skip_ops stack, creating it on first access."""
        local = self._skip_ops_local
        if not hasattr(local, "stack"):
            local.stack = []
        return local.stack

    @property
    def _effective_skip_ops(self) -> frozenset[str]:
        """Cumulative union of all skip_ops from the current thread's stack."""
        stack = self._get_skip_ops_stack()
        return stack[-1] if stack else frozenset[str]()

    @property
    def _override_tuning_buckets(self) -> tuple[int, ...] | None:
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

    def get_effective_map_to_tuning_buckets(
        self,
        tuning_config: "TuningConfig",
        spec_idx: int = 0,
    ) -> Callable[[int], int]:
        """Return the currently effective ``map_to_tuning_buckets`` callable.

        Reflects any active ``autotune(tuning_buckets=..., round_up=...)``
        overrides on the current thread.  When no override is active,
        returns
        ``tuning_config.dynamic_tensor_specs[spec_idx].map_to_tuning_buckets``
        unchanged.

        Runners that have their own internal bucketing (e.g. cuDNN's
        override-shape ``cache_m``) should call this so their bucket
        function matches what the autotuner uses for cache lookup;
        otherwise a tactic profiled under one bucket scheme can be
        applied at runtime under a different one, with subtly wrong
        results.

        Args:
            tuning_config: The default ``TuningConfig`` the runner would
                pass to ``choose_one``.
            spec_idx: Index into ``tuning_config.dynamic_tensor_specs``
                whose mapper to return.  Defaults to 0 (single dynamic
                dimension, the common case).

        Returns:
            A callable ``int -> int`` mapping a runtime size to the
            currently effective bucket value.
        """
        with self._lock:
            if self._override_tuning_buckets is not None or self._override_round_up:
                tuning_config = self._apply_tuning_overrides(tuning_config)
        return tuning_config.dynamic_tensor_specs[spec_idx].map_to_tuning_buckets

    def search_cache(
        self,
        custom_op: str,
        runners: list[TunableRunner],
        input_shapes: tuple[tuple[int, ...], ...],
        tuning_config: TuningConfig,
        value_buckets: Tuple = (),
        inputs: list[torch.Tensor] | None = None,
    ) -> tuple[bool, int, int, OptimizationProfile | None]:
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
            value_buckets: Tuple of value bucket IDs for value-aware tuning
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
                    custom_op,
                    r,
                    input_shapes,
                    tuning_config,
                    extras,
                    value_buckets=value_buckets,
                )
                # 1. In-memory cache (from live tuning)
                if cache_key in self.profiling_cache:
                    return True, *self.profiling_cache[cache_key]

                # Build the hash-free file key used by both user configs and bundled configs.
                # Include extras (index 4) so that runner specific parameters
                # are not lost on disk.
                file_key = cache_key.file_key

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
                    output = load_from_file(cache_key.file_key)
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

        new_specs: list[DynamicTensorSpec] = []
        for spec in tuning_config.dynamic_tensor_specs:
            new_gen: tuple[int, ...] | Callable[[int], Iterable[int]]
            new_map: Callable[[int], int]
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

                    def _clamped_po2_mapper(x: int, gen_fn=gen_fn):
                        buckets = tuple(sorted(set(gen_fn(x))))
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
                    value_specs=spec.value_specs,
                )
            )

        new_config = TuningConfig(
            dynamic_tensor_specs=tuple(new_specs),
            constraint_specs=tuning_config.constraint_specs,
            use_cold_l2_cache=tuning_config.use_cold_l2_cache,
            use_cuda_graph=tuning_config.use_cuda_graph,
            value_sample_count=tuning_config.value_sample_count,
            default_value_buckets=tuning_config.default_value_buckets,
            inputs_pre_hook=tuning_config.inputs_pre_hook,
        )
        self._override_config_cache.setdefault(tuning_config, {})[cache_key] = (
            new_config
        )
        return new_config

    def choose_one(
        self,
        custom_op: str,
        runners: list[TunableRunner],
        tuning_config: TuningConfig,
        inputs: list[torch.Tensor],
        **kwargs,
    ) -> tuple[TunableRunner, int]:
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
        # Skip profiling for ops in the skip_ops set — return fallback
        # immediately.  The fallback runner (runners[0], tactic=-1) uses
        # the op's built-in heuristic, avoiding kernel compilation.
        # Checked before acquiring the lock since _effective_skip_ops is
        # thread-local and does not touch shared state.
        if custom_op in self._effective_skip_ops:
            logger.debug(
                f"[AutoTuner]: Skipping autotuning for '{custom_op}' "
                f"(in skip_ops). Using fallback tactic."
            )
            if not runners:
                raise ValueError(f"No runners provided for op '{custom_op}'")
            return runners[0], -1

        with self._lock:
            # Apply tuning bucket / rounding overrides from autotune() context.
            if self._override_tuning_buckets is not None or self._override_round_up:
                tuning_config = self._apply_tuning_overrides(tuning_config)

            input_shapes = tuple(self._get_input_sizes(inputs))

            def _serialize_tactic(tactic: Any) -> Any:
                if isinstance(tactic, tuple):
                    return tuple(
                        int(x) if isinstance(x, (int, float)) else x for x in tactic
                    )
                if isinstance(tactic, list):
                    return tuple(
                        int(x) if isinstance(x, (int, float)) else x for x in tactic
                    )
                return tactic

            def _tactic_key(tactic: Any) -> Any:
                return _json_to_tactic(_tactic_to_json(tactic))

            value_specs = AutoTuner._get_value_specs(tuning_config)
            if (
                _is_value_aware_autotune()
                and value_specs
                and not torch.cuda.is_current_stream_capturing()
            ):
                value_buckets = tuple(
                    self._map_dynamic_value_bucket(spec, inputs, kwargs)
                    for spec in value_specs
                )
            else:
                value_buckets = ()

            # Early return if it's not tuning, use cache found one or fallback one
            if not self.is_tuning_mode:
                is_cache_hit, runner_id, tactic, stored_profile = self.search_cache(
                    custom_op,
                    runners,
                    input_shapes,
                    tuning_config,
                    value_buckets=value_buckets,
                    inputs=inputs,
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
                        f"[AutoTuner]: Generated key{AutoTuner._get_cache_key(custom_op, runners[0], input_shapes, tuning_config, runners[0].get_cache_key_extras(inputs), value_buckets=value_buckets)}"
                    )

                    # If the user has loaded an autotune cache (via
                    # ``autotune(cache=...)``) or warmed up profiling
                    # results *for this specific custom_op* in the
                    # current process but this particular input shape
                    # still falls back, that almost always means the
                    # runtime input is *outside the tuned range* (e.g.
                    # ``max_num_tokens`` was 2048 during tuning,
                    # runtime sees 4000).  This is a silent perf
                    # regression: the fallback tactic is correct but
                    # not optimised for this shape.  Warn once per
                    # unique (op, profile-signature) pair so the user
                    # can extend ``tuning_buckets`` / re-tune.
                    #
                    # The has-tune-data check is intentionally scoped
                    # to ``custom_op`` -- unrelated ops with their own
                    # cache entries should not trigger a "tuned range
                    # exceeded" warning for an op that was never tuned
                    # in the first place.  ``profiling_cache`` keys are
                    # ``ProfilingCacheKey`` instances and ``_file_configs``
                    # keys are ``str((custom_op, runner_class_name, profile))``,
                    # so we filter by ``custom_op`` on each.
                    op_has_profiling = any(
                        k.custom_op == custom_op for k in self.profiling_cache
                    )
                    file_key_op_prefix = f"({repr(custom_op)}, "
                    op_has_file = any(
                        k.startswith(file_key_op_prefix) for k in self._file_configs
                    )
                    has_tune_data = op_has_profiling or op_has_file
                    if has_tune_data:
                        try:
                            signature = self._find_nearest_profile(
                                input_shapes, tuning_config, value_buckets
                            )
                        except Exception:
                            signature = tuple(tuple(s) for s in input_shapes)
                        warn_key = (custom_op, signature)
                        if warn_key not in self._logged_cache_miss_oor:
                            self._logged_cache_miss_oor.add(warn_key)
                            logger.warning(
                                f"[AutoTuner]: No tuned config covers "
                                f"{custom_op} input_shapes={input_shapes}; "
                                f"falling back to runner={runners[0].__class__.__name__} "
                                f"tactic=-1.  This shape is outside the tuning "
                                f"bucket range -- expand tuning_buckets / "
                                f"max_num_tokens during the next tuning "
                                f"pass to avoid this perf cliff."
                            )
                self.last_selection = {
                    "custom_op": custom_op,
                    "input_shapes": tuple(
                        tuple(int(v) for v in shape) for shape in input_shapes
                    ),
                    "value_buckets": tuple(value_buckets),
                    "runner_id": int(runner_id),
                    "runner_name": runner.__class__.__name__,
                    "tactic": _serialize_tactic(tactic),
                    "is_cache_hit": bool(is_cache_hit),
                    "is_tuning_mode": False,
                }
                return runner, tactic

            assert len(runners) > 0, "At least one runner is required"
            assert all([isinstance(r, TunableRunner) for r in runners]), (
                "All Given runners must be subclass of TunableRunner"
            )

            profiles = self._generate_optimization_profiles(tuning_config, inputs)
            # Record the total configs to try
            self.stats.tuned_op_total_configs[custom_op] = len(profiles)

            input_arena = None
            input_arena_attempted = False
            arena_fallback_reason = ""
            use_value_aware_arena = (
                _is_value_aware_autotune()
                and tuning_config.use_cuda_graph
                and bool(value_specs)
                and tuning_config.inputs_pre_hook is None
            )

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
                        value_buckets=p.value_buckets,
                        inputs=inputs,
                    )
                    if not is_cache_hit:
                        profile_batches = None
                        if use_value_aware_arena:
                            if not input_arena_attempted:
                                input_arena_attempted = True
                                try:
                                    input_arena = _ValueAwareInputArena.create(
                                        self, profiles, inputs, tuning_config
                                    )
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    arena_fallback_reason = "out_of_memory"
                                except (TypeError, ValueError) as error:
                                    arena_fallback_reason = type(error).__name__
                            if input_arena is not None:
                                profile_batches = input_arena.batches_for(p)
                                tensors = profile_batches[0]
                                self._apply_value_specs(
                                    p, tensors, inputs, tuning_config
                                )
                            else:
                                tensors = self._prepare_input_tensors(
                                    p, inputs, tuning_config
                                )
                        else:
                            # Synthesize inputs only on the profiling path.
                            tensors = self._prepare_input_tensors(
                                p, inputs, tuning_config
                            )
                        # Apply the optional inputs_pre_hook to inject a
                        # deterministic / realistic distribution before
                        # the per-tactic profile loop.
                        if tuning_config.inputs_pre_hook is not None:
                            tensors = list(tuning_config.inputs_pre_hook(tensors))
                        value_input_sets = None
                        if (
                            _is_value_aware_autotune()
                            and p.value_buckets
                            and value_specs
                        ):
                            value_input_sets = []
                            for sample_index in range(
                                self._value_sample_count(p, tuning_config)
                            ):
                                value_tensors = list(tensors)
                                for spec, bucket_id in zip(
                                    value_specs,
                                    p.value_buckets,
                                    strict=True,
                                ):
                                    generator = (
                                        spec.sample_value_generator
                                        or spec.tensor_value_generator
                                    )
                                    num_params = len(
                                        inspect.signature(generator).parameters
                                    )
                                    if num_params >= 5:
                                        value_tensors[spec.input_idx] = generator(
                                            bucket_id,
                                            tensors[spec.input_idx],
                                            inputs[spec.input_idx],
                                            inputs,
                                            sample_index,
                                        )
                                    elif num_params >= 4:
                                        value_tensors[spec.input_idx] = generator(
                                            bucket_id,
                                            tensors[spec.input_idx],
                                            inputs[spec.input_idx],
                                            inputs,
                                        )
                                    elif num_params >= 3:
                                        value_tensors[spec.input_idx] = generator(
                                            bucket_id,
                                            tensors[spec.input_idx],
                                            inputs[spec.input_idx],
                                        )
                                    else:
                                        value_tensors[spec.input_idx] = generator(
                                            bucket_id,
                                            tensors[spec.input_idx],
                                        )
                                value_input_sets.append(value_tensors)
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
                        profile_records: List[Dict[str, Any]] = []
                        opt_shapes = p.get_opt_shapes()
                        num_tokens = (
                            opt_shapes[0][0] if opt_shapes and opt_shapes[0] else "?"
                        )
                        skipped_count = 0
                        for r_id, r in enumerate(runners):
                            # TODO: use FakeTensor here.
                            tactic_groups = r.get_tactic_groups(tensors, p)
                            candidate_stages: Dict[Any, List[str]] = {}
                            if tactic_groups is None:
                                # Exhaustive runners profile every valid tactic directly.
                                valid_tactics = r.get_valid_tactics(tensors, p)
                            else:
                                # Factorized runners deduplicate overlapping group candidates.
                                valid_tactics = []
                                seen_tactics = set()
                                for group_id, group in enumerate(tactic_groups):
                                    for candidate in group:
                                        candidate_key = _tactic_key(candidate)
                                        candidate_stages.setdefault(
                                            candidate_key, []
                                        ).append(f"group_{group_id}")
                                        if candidate_key not in seen_tactics:
                                            seen_tactics.add(candidate_key)
                                            valid_tactics.append(candidate)
                            runner_profile_cache_key = AutoTuner._get_cache_key(
                                custom_op,
                                r,
                                p.get_opt_shapes(),
                                tuning_config,
                                r.get_cache_key_extras(tensors),
                                value_buckets=p.value_buckets,
                            )
                            # The explicit default-profile winner is the fixed
                            # NoDA fallback.  Factorized value profiles would
                            # not necessarily measure that exact composed
                            # tactic, so include it as a measurement-only
                            # candidate for the matching tile.  It is not part
                            # of either factorized group and therefore cannot
                            # change their winners or the composed DA tactic.
                            if (
                                tactic_groups is not None
                                and tuning_config.default_value_buckets is not None
                                and p.value_buckets
                                != tuning_config.default_value_buckets
                                and p.value_buckets
                            ):
                                shape_cache_key = AutoTuner._get_cache_key(
                                    custom_op,
                                    r,
                                    p.get_opt_shapes(),
                                    tuning_config,
                                    r.get_cache_key_extras(tensors),
                                )
                                default_entry = self.profiling_cache.get(
                                    shape_cache_key
                                )
                                if default_entry is not None:
                                    default_runner_id, default_tactic, _ = default_entry
                                    try:
                                        matches_profile_tile = int(
                                            default_tactic[0]
                                        ) == int(p.value_buckets[0])
                                    except (IndexError, TypeError, ValueError):
                                        matches_profile_tile = False
                                    default_key = _tactic_key(default_tactic)
                                    if (
                                        default_runner_id == r_id
                                        and matches_profile_tile
                                        and default_key not in seen_tactics
                                    ):
                                        seen_tactics.add(default_key)
                                        valid_tactics.append(default_tactic)
                                        candidate_stages.setdefault(
                                            default_key, []
                                        ).append("default_profile_baseline")
                            valid_tactics = self._blocklist.filter(
                                custom_op, r, valid_tactics
                            )
                            runner_arg_names = runner_arg_names_map[r]
                            if (
                                "do_preparation" in runner_arg_names
                                and len(valid_tactics) > 0
                            ):
                                r(tensors, tactic=-1, do_preparation=True, **kwargs)

                            def profile_tactic(tac, selection_stage):
                                nonlocal skipped_count
                                session_diagnostics = None
                                value_time_min_ms = None
                                value_time_median_ms = None
                                value_time_max_ms = None
                                try:
                                    if value_input_sets is not None:
                                        if tuning_config.use_cuda_graph:
                                            with _autotune_nvtx_range(
                                                "profile-tactic:"
                                                f"op={custom_op}, tokens={num_tokens}, "
                                                f"value-buckets={p.value_buckets}, "
                                                f"runner={r_id}, tactic={tac}"
                                            ):
                                                session = _GraphProfileSession.capture(
                                                    self,
                                                    r,
                                                    tensors,
                                                    value_input_sets[0],
                                                    tac,
                                                    tuning_config,
                                                    tuple(
                                                        spec.input_idx
                                                        for spec in value_specs
                                                    ),
                                                    input_batches=profile_batches,
                                                    arena_diagnostics=(
                                                        input_arena.diagnostics()
                                                        if input_arena is not None
                                                        else {
                                                            "input_arena_reused": False,
                                                            "cold_l2_lane_count": 0,
                                                            "cold_l2_arena_bytes": 0,
                                                            "mutable_route_h2d_copies": 0,
                                                            "arena_fallback_reason": arena_fallback_reason,
                                                        }
                                                    ),
                                                    value_stagers={
                                                        spec.input_idx: spec.sample_value_stager
                                                        for spec in value_specs
                                                        if spec.sample_value_stager
                                                        is not None
                                                    },
                                                    **kwargs,
                                                )
                                            value_times = [
                                                session.measure(value_tensors)
                                                for value_tensors in value_input_sets
                                            ]
                                            session_diagnostics = session.diagnostics()
                                        else:
                                            value_times = [
                                                self._profile_single_kernel(
                                                    r,
                                                    value_tensors,
                                                    tac,
                                                    tuning_config,
                                                    **kwargs,
                                                )
                                                for value_tensors in value_input_sets
                                            ]
                                            session_diagnostics = None
                                        value_time_min_ms = min(value_times)
                                        value_time_median_ms = statistics.median(
                                            value_times
                                        )
                                        value_time_max_ms = max(value_times)
                                        time_measured = value_time_median_ms
                                    else:
                                        time_measured = self._profile_single_kernel(
                                            r, tensors, tac, tuning_config, **kwargs
                                        )
                                        session_diagnostics = None
                                except torch.cuda.OutOfMemoryError:
                                    # Distributed autotuning: the per-tactic
                                    # all-reduce must run the same number of
                                    # times on every rank. Bubbling OOM up to
                                    # choose_one's outer handler early-returns
                                    # runners[0], -1 on this rank while peers
                                    # keep profiling -- the next tactic's
                                    # all_reduce then deadlocks. When a tune
                                    # group is set, treat OOM like any other
                                    # failed tactic (free memory, disqualify
                                    # with inf, keep looping in lockstep) so
                                    # cardinality is preserved. Without a group
                                    # the original early-return path is kept.
                                    if _tune_process_group is None:
                                        raise
                                    with contextlib.suppress(Exception):
                                        torch.cuda.empty_cache()
                                    skipped_count += 1
                                    time_measured = float("inf")
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

                                    # Record the failed tactic value for the
                                    # blocklist generator.
                                    self.stats.failed_tactics.setdefault(
                                        f"{custom_op}::{r.__class__.__name__}", set()
                                    ).add(_tactic_to_json_hashable(tac))

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
                                            value_buckets=p.value_buckets,
                                        )
                                    )

                                    # Set time_measured to inf to notify the failure of the tactic. This can happen when `get_valid_tactics` mistakenly return wrong tactics
                                    # or some runtime error occurs during profiling.
                                    time_measured = float("inf")
                                if _AUTOTUNE_DUMP_PATH:
                                    profile_records.append(
                                        {
                                            "op": custom_op,
                                            "runner": r.__class__.__name__,
                                            "num_tokens": num_tokens,
                                            "value_buckets": p.value_buckets
                                            if p.value_buckets
                                            else "",
                                            "tactic": _serialize_tactic(tac),
                                            "time_ms": time_measured,
                                            "value_time_min_ms": value_time_min_ms,
                                            "value_time_median_ms": value_time_median_ms,
                                            "value_time_max_ms": value_time_max_ms,
                                            "selection_stage": selection_stage,
                                            "is_best": False,
                                            **(session_diagnostics or {}),
                                        }
                                    )
                                tactic_time_key = (
                                    runner_profile_cache_key,
                                    _tactic_key(tac),
                                )
                                previous_time = self.profiling_tactic_time_cache.get(
                                    tactic_time_key, float("inf")
                                )
                                if time_measured < previous_time:
                                    self.profiling_tactic_time_cache[
                                        tactic_time_key
                                    ] = time_measured
                                return time_measured

                            measured_tactics = {}
                            # Profile every unique exhaustive or grouped candidate once.
                            for tac in valid_tactics:
                                if tactic_groups is None:
                                    time_measured = profile_tactic(tac, "exhaustive")
                                    if time_measured < min_time:
                                        min_time = time_measured
                                        runner_id, tactic = r_id, tac
                                    continue

                                tactic_key = _tactic_key(tac)
                                time_measured = profile_tactic(
                                    tac, "+".join(candidate_stages[tactic_key])
                                )
                                measured_tactics[tactic_key] = (time_measured, tac)

                            # Resolve factorized groups only after all shared candidates run.
                            if tactic_groups is not None:
                                factorized_selection_succeeded = False
                                group_winners: List[Any] = []
                                # Pick the fastest candidate from each anchored component sweep.
                                for group in tactic_groups:
                                    measured_group = [
                                        measured_tactics[_tactic_key(candidate)]
                                        for candidate in group
                                        if _tactic_key(candidate) in measured_tactics
                                        and measured_tactics[_tactic_key(candidate)][0]
                                        < float("inf")
                                    ]
                                    if not measured_group:
                                        group_winners = []
                                        break
                                    group_winners.append(
                                        min(
                                            measured_group, key=lambda result: result[0]
                                        )[1]
                                    )
                                # Compose the component winners and measure the executable tactic.
                                if group_winners:
                                    try:
                                        composed_tactic = r.compose_tactics(
                                            group_winners, tensors, p
                                        )
                                        composed_time = profile_tactic(
                                            composed_tactic, "composed"
                                        )
                                        if composed_time < min_time:
                                            min_time = composed_time
                                            runner_id, tactic = r_id, composed_tactic
                                        factorized_selection_succeeded = (
                                            composed_time < float("inf")
                                        )
                                    except Exception as e:
                                        skipped_count += 1
                                        logger.debug(
                                            "[Autotuner]: Failed to compose factorized "
                                            f"tactics for {r}: {e}"
                                        )

                                # Fall back to the complete search if composition cannot succeed.
                                if not factorized_selection_succeeded:
                                    logger.debug(
                                        "[Autotuner]: Factorized tactic selection failed for "
                                        f"{r}; falling back to exhaustive tactics"
                                    )
                                    for tac in r.get_valid_tactics(tensors, p):
                                        time_measured = profile_tactic(
                                            tac, "exhaustive_fallback"
                                        )
                                        if time_measured < min_time:
                                            min_time = time_measured
                                            runner_id, tactic = r_id, tac

                        if skipped_count > 0:
                            logger.info(
                                f"[Autotuner]: Skipped {skipped_count} unsupported tactic(s) for {custom_op} "
                                f"(enable debug logs to see details)"
                            )

                        if runner_id is not None:
                            if _AUTOTUNE_DUMP_PATH and profile_records:
                                for rec in reversed(profile_records):
                                    if (
                                        rec["tactic"] == _serialize_tactic(tactic)
                                        and rec["runner"]
                                        == runners[runner_id].__class__.__name__
                                    ):
                                        rec["is_best"] = True
                                        break
                                self._profiling_records.extend(profile_records)
                            # At least one valid (runner, tactic) pair is found
                            cache_key = AutoTuner._get_cache_key(
                                custom_op,
                                runners[runner_id],
                                p.get_opt_shapes(),
                                tuning_config,
                                runners[runner_id].get_cache_key_extras(tensors),
                                value_buckets=p.value_buckets,
                            )
                            # inspect call stack
                            self.profiling_cache[cache_key] = (runner_id, tactic, p)
                            self.profiling_time_cache[cache_key] = min_time
                            if (
                                tuning_config.default_value_buckets is not None
                                and p.value_buckets
                                == tuning_config.default_value_buckets
                            ):
                                default_profile = copy.deepcopy(p)
                                default_profile.value_buckets = ()
                                shape_cache_key = AutoTuner._get_cache_key(
                                    custom_op,
                                    runners[runner_id],
                                    p.get_opt_shapes(),
                                    tuning_config,
                                    runners[runner_id].get_cache_key_extras(tensors),
                                )
                                self.profiling_cache[shape_cache_key] = (
                                    runner_id,
                                    tactic,
                                    default_profile,
                                )
                                self.profiling_time_cache[shape_cache_key] = min_time
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
                custom_op,
                runners,
                input_shapes,
                tuning_config,
                value_buckets=value_buckets,
                inputs=inputs,
            )

            self._dump_profiling_records(custom_op)
            self.last_selection = {
                "custom_op": custom_op,
                "input_shapes": tuple(
                    tuple(int(v) for v in shape) for shape in input_shapes
                ),
                "value_buckets": tuple(value_buckets),
                "runner_id": int(runner_id),
                "runner_name": runners[runner_id].__class__.__name__,
                "tactic": _serialize_tactic(tactic),
                "is_cache_hit": True,
                "is_tuning_mode": True,
            }
            return runners[runner_id], tactic

    @staticmethod
    def _get_value_specs(
        tuning_config: TuningConfig,
    ) -> Tuple[DynamicValueSpec, ...]:
        specs: List[DynamicValueSpec] = []
        for tensor_spec in tuning_config.dynamic_tensor_specs:
            specs.extend(tensor_spec.value_specs)
        return tuple(specs)

    def _value_sample_count(
        self,
        profile: OptimizationProfile,
        tuning_config: TuningConfig,
    ) -> int:
        """Return the outer value-realization count for one profile."""
        callback = tuning_config.value_sample_count
        count = (
            self.repeat
            if callback is None
            else callback(profile.value_buckets, self.repeat)
        )
        if not isinstance(count, int) or count <= 0:
            raise ValueError(
                "value_sample_count must return a positive int, "
                f"got {count!r} for value_buckets={profile.value_buckets!r}"
            )
        return count

    @staticmethod
    def _map_dynamic_value_bucket(
        spec: DynamicValueSpec,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> int:
        tensor = inputs[spec.input_idx]
        num_params = len(inspect.signature(spec.map_to_value_bucket).parameters)
        if num_params >= 3:
            return spec.map_to_value_bucket(tensor, inputs, kwargs)
        return spec.map_to_value_bucket(tensor)

    def _get_input_sizes(self, inputs: list[Any]) -> tuple[tuple[int, ...], ...]:
        """Return ``torch.Size`` for each input, using ``(0,)`` for non-Tensor values."""
        return tuple(
            tuple(tensor.size()) if isinstance(tensor, torch.Tensor) else (0,)
            for tensor in inputs
        )

    def _profile_single_kernel(
        self,
        runner: TunableRunner,
        inputs: list[torch.Tensor],
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

            All runner invocations inside this method (warmup + measurement)
            execute under ``_profile_measurement_scope`` so that runners can
            consult ``is_in_profile_measurement()`` to apply consistent
            allocation behavior across tactics.  This is what unbiases the
            autotuner's tactic comparison for runners whose normal
            (non-profiling) path uses preallocated buffers sized for a
            specific tile_size: during the measurement window every tactic
            sees the same per-call allocation overhead, and the autotuner
            picks based on intrinsic kernel time.
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

        # Run the timing under ``_profile_measurement_scope`` (so runners
        # can consult ``is_in_profile_measurement()``), then — if a
        # cross-rank group is set — all-reduce the measured time so every
        # rank's argmin picks the same tactic. Local GPU timing noise
        # otherwise causes per-rank tactic divergence that can deadlock
        # collective allocation paths (e.g. NCCL_WIN_COLL_SYMMETRIC). See
        # ``set_autotune_process_group``.
        #
        # Collective cardinality invariant: every rank MUST reach the
        # all-reduce below exactly once per ``_profile_single_kernel``
        # call. If one rank raises inside the measurement window and exits
        # without reducing while peers are still waiting, the next
        # tactic's reduce deadlocks. We therefore catch failures here,
        # mark this rank's ``avg_time`` as ``inf`` (so the tactic is
        # disqualified everywhere after the SUM), run the reduce
        # unconditionally, and then re-raise so the outer error-handling
        # path in ``choose_one`` still runs (logging, stats, OOM
        # fallback).
        profile_exc: Optional[BaseException] = None
        try:
            with _profile_measurement_scope():
                # warm up, no timing
                for _ in range(self.warmup):
                    runner(input_tensor_batches[-1], tactic=tactic, **kwargs)

                avg_time = pure_profile(stream, self.repeat)
        except BaseException as e:  # noqa: BLE001
            # Catch everything (incl. KeyboardInterrupt / SystemExit): this
            # rank must still reach the all-reduce below or peers already
            # waiting on it deadlock. The original error is re-raised after.
            avg_time = float("inf")
            profile_exc = e

        try:
            if _tune_process_group is not None:
                import torch.distributed as dist

                # NCCL requires a CUDA tensor; a gloo (CPU) subgroup — the
                # recommended choice — uses a CPU tensor.
                backend = str(dist.get_backend(_tune_process_group)).lower()
                device = "cuda" if backend == "nccl" else "cpu"
                time_tensor = torch.tensor(
                    [avg_time], dtype=torch.float64, device=device
                )
                dist.all_reduce(
                    time_tensor, op=dist.ReduceOp.SUM, group=_tune_process_group
                )
                avg_time = time_tensor.item() / dist.get_world_size(_tune_process_group)
        finally:
            # Re-raise even if the collective itself failed, so the original
            # profiling error is never masked by a secondary reduce error.
            if profile_exc is not None:
                raise profile_exc

        shapes = self._get_input_sizes(inputs)
        logger.debug(
            f"[Autotuner]: profiling {runner} {tactic}, shapes={shapes}, avg_time {avg_time}"
        )

        return avg_time

    def _generate_optimization_profiles(
        self, tuning_config: TuningConfig, inputs: list[torch.Tensor]
    ) -> list[OptimizationProfile]:
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

        generated_profiles: list[OptimizationProfile] = []

        dynamic_dims: list[tuple[Any, ...]] = []

        for spec in tuning_config.dynamic_tensor_specs:
            assert callable(spec.gen_tuning_buckets) or isinstance(
                spec.gen_tuning_buckets, (list, tuple)
            ), (
                "The given dynamic dimension must provide a opt value generation function or a list of opt values"
            )
            assert len(spec.input_idx) == len(spec.dim_idx), (
                f"The number of input indices and dimension indices must be the same, got {len(spec.input_idx)} and {len(spec.dim_idx)}"
            )
            assert spec.tensor_initializers is not None
            assert len(spec.tensor_initializers) == len(spec.input_idx), (
                f"The number of tensor initializers and input indices must be the same, got {len(spec.tensor_initializers)} and {len(spec.input_idx)}"
            )
            for i, idx in enumerate(spec.input_idx):
                base_profile.tensor_initializers[idx] = spec.tensor_initializers[i]

            if callable(spec.gen_tuning_buckets):
                opt_shapes = spec.gen_tuning_buckets(
                    _get_opt(base_profile.shapes[spec.input_idx[0]][spec.dim_idx[0]])
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

        # If value-aware autotuning is enabled and there are value specs, do a
        # cartesian product with value buckets.
        value_specs = self._get_value_specs(tuning_config)
        if _is_value_aware_autotune() and value_specs:
            expanded_profiles = []
            for profile in generated_profiles:
                if tuning_config.default_value_buckets is not None:
                    if len(tuning_config.default_value_buckets) != len(value_specs):
                        raise ValueError(
                            "default_value_buckets must provide one marker for "
                            "each DynamicValueSpec"
                        )
                    default_profile = copy.deepcopy(profile)
                    default_profile.value_buckets = tuple(
                        tuning_config.default_value_buckets
                    )
                    expanded_profiles.append(default_profile)
                value_bucket_lists = []
                has_empty_value_spec = False
                for value_spec in value_specs:
                    generator = value_spec.gen_value_buckets
                    if callable(generator) and not isinstance(generator, tuple):
                        try:
                            inspect.signature(generator).bind(profile)
                        except (TypeError, ValueError):
                            buckets = generator()
                        else:
                            buckets = generator(profile)
                    else:
                        buckets = generator
                    buckets = tuple(buckets)
                    if not buckets:
                        has_empty_value_spec = True
                        break
                    value_bucket_lists.append(buckets)

                if has_empty_value_spec or not value_bucket_lists:
                    expanded_profiles.append(profile)
                    continue

                for value_buckets in itertools.product(*value_bucket_lists):
                    p = copy.deepcopy(profile)
                    p.value_buckets = tuple(value_buckets)
                    expanded_profiles.append(p)
                    logger.debug(
                        f"[Autotuner]: generated profile with value_buckets={p.value_buckets}: {p}"
                    )
            generated_profiles = expanded_profiles
        return generated_profiles

    @classmethod
    @functools.lru_cache(maxsize=16384)
    def _find_nearest_profile(
        cls,
        shapes: tuple[tuple[int, ...], ...],
        tuning_config: TuningConfig,
        value_buckets: Tuple = (),
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
        profile = tuple(tuple(shape) for shape in base_profile)
        if value_buckets:
            return (profile, value_buckets)
        return profile

    @classmethod
    def _get_cache_key(
        cls,
        custom_op: str,
        runner: TunableRunner,
        input_shapes: tuple[tuple[int, ...], ...],
        tuning_config: TuningConfig,
        extras: tuple[Any, ...] = (),
        value_buckets: Tuple = (),
    ) -> ProfilingCacheKey:
        return ProfilingCacheKey(
            custom_op=custom_op,
            runner_class_name=runner.__class__.__name__,
            runner_hash=hash(runner),
            nearest_profile=cls._find_nearest_profile(
                input_shapes, tuning_config, value_buckets
            ),
            extras=extras,
        )

    def _create_tensor_like(
        self,
        origin_tensor: torch.Tensor,
        dims: list[Dim],
        initializer: TensorInitializer,
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
        shapes = tuple(_get_opt(d) for d in dims)
        return initializer(shapes, dtype, device)

    def _prepare_input_tensors(
        self,
        profile: OptimizationProfile,
        inputs: list[torch.Tensor | None],
        tuning_config: Optional[TuningConfig] = None,
    ) -> list[torch.Tensor | None]:
        """Create tensors matching *profile* shapes; reuse static inputs as-is."""
        default_initializer = autotuner_initializer_rand_scaled
        tensors: list[torch.Tensor | None] = []
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
        self._apply_value_specs(profile, tensors, inputs, tuning_config)
        return tensors

    @staticmethod
    def _apply_value_specs(
        profile: OptimizationProfile,
        tensors: List[Optional[torch.Tensor]],
        inputs: List[Optional[torch.Tensor]],
        tuning_config: Optional[TuningConfig],
    ) -> None:
        """Apply value generators in place after profile storage is available."""
        # Apply value generators for value-aware profiling.
        if (
            _is_value_aware_autotune()
            and tuning_config is not None
            and profile.value_buckets
        ):
            value_specs = AutoTuner._get_value_specs(tuning_config)
            if len(value_specs) != len(profile.value_buckets):
                raise ValueError(
                    "value profile/spec mismatch: "
                    f"profile={profile.value_buckets!r}, num_specs={len(value_specs)}"
                )
            for spec, bucket_id in zip(value_specs, profile.value_buckets, strict=True):
                generator = spec.tensor_value_generator
                num_params = len(inspect.signature(generator).parameters)
                if num_params >= 4:
                    tensors[spec.input_idx] = generator(
                        bucket_id,
                        tensors[spec.input_idx],
                        inputs[spec.input_idx],
                        inputs,
                    )
                elif num_params >= 3:
                    tensors[spec.input_idx] = generator(
                        bucket_id,
                        tensors[spec.input_idx],
                        inputs[spec.input_idx],
                    )
                else:
                    tensors[spec.input_idx] = generator(
                        bucket_id, tensors[spec.input_idx]
                    )

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
            configs: dict[str, Any] = {}

            # Include previously loaded file configs as a base
            for file_key, (runner_name, tactic) in self._file_configs.items():
                configs[file_key] = [runner_name, _tactic_to_json(tactic)]

            num_previous = len(configs)

            # Overlay in-memory profiling results (take priority over loaded configs)
            for cache_key, cache_value in self.profiling_cache.items():
                _, tactic, _ = cache_value

                # Use hash-free key including extras so runner specific parameters
                # are preserved across save or load.
                file_key = cache_key.file_key

                # Store runner class name (not positional index) for robustness
                tactic_json = _tactic_to_json(tactic)
                configs[file_key] = [cache_key.runner_class_name, tactic_json]

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
        environment, the entire cache is **skipped** to avoid silently
        using invalid tactics.  Specifically the following metadata fields
        must all match (or be the wildcard ``"*"``):

            * ``flashinfer_version`` (writer bucketing/ordering changes)
            * ``cuda_version``
            * ``cublas_version``
            * ``cudnn_version`` (backend)
            * ``cudnn_frontend_version`` (Python wrapper -- can shuffle
              ``policy=ALL`` plan_index ordering independent of backend)
            * ``gpu`` (different SM may expose different engines)

        Caches saved by older flashinfer versions that predate the
        ``cudnn_frontend_version`` metadata field will fail this check
        (since saved metadata does not contain the field) and need to
        be regenerated.  See :func:`_collect_metadata` for the exact list.

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
        inputs: list[Any],
        tuning_config: TuningConfig,
    ) -> list[list[Any]]:
        """Create multiple input copies to flush the L2 cache between profiling iterations."""
        one_buffer_bytes = sum(
            input.numel() * input.element_size()
            if isinstance(input, torch.Tensor)
            else 0
            for input in inputs
        )
        num_buffers = self._get_cold_l2_buffer_count(
            one_buffer_bytes, tuning_config.use_cold_l2_cache
        )
        if num_buffers == 1:
            if one_buffer_bytes <= 0:
                logger.debug(
                    "[Autotuner] No tensor inputs or zero-sized tensors; falling back to single-batch profiling."
                )
            return [inputs]

        inputs_list = [inputs]
        for _ in range(num_buffers - 1):
            inputs_list.append(
                [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
            )

        logger.debug(
            f"[Autotuner] use_cold_l2_cache={tuning_config.use_cold_l2_cache}, use {num_buffers} different tensors for profiling"
        )
        return inputs_list

    def _get_cold_l2_buffer_count(
        self, one_buffer_bytes: int, use_cold_l2_cache: bool
    ) -> int:
        """Return the existing cold-L2 lane count without allocating tensors."""
        if not use_cold_l2_cache or one_buffer_bytes <= 0:
            return 1
        return min(
            self._get_l2_cache_size_in_bytes() * 3 // one_buffer_bytes + 1,
            self.repeat + 1,
        )

    def clear_cache(self) -> None:
        """Clear the profiling cache and user-loaded file configs."""
        with self._lock:
            self.profiling_cache.clear()
            self.profiling_time_cache.clear()
            self.profiling_tactic_time_cache.clear()
            self.last_selection = None
            self._profiling_records.clear()
            self._file_configs.clear()
            self._logged_file_hits.clear()
            self._logged_cache_miss_oor.clear()
            self._dirty = False
            self._dirty_seq = 0

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()

    def _get_l2_cache_size_in_bytes(self, device_id: int | None = None) -> int:
        """Return the L2 cache size in bytes for the given (or current) CUDA device."""
        if device_id is None:
            device_id = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device_id).L2_cache_size
