import contextlib
import copy
import importlib
import inspect
import itertools
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional

import torch

# from tensorrt_llm.bindings.internal.runtime import delay_kernel
# from tensorrt_llm.logger import logger
from flashinfer.tllm_utils import delay_kernel

from .jit.core import logger

# This version should be updated whenever the nvfp4_cutlass backend is changed,
# such as when new kernels or configs are added. In such cases, the tuning configs
# should also be updated. Currently, this process is manual, but it should be automated in the future.
_nvfp4_cutlass_version = "0.1"


def get_config_path(is_module: bool):
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
    """

    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: Tuple[ConstraintSpec, ...] = ()


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
            * in eager mode, w/o autotunning the custom op should have at least one kernel, which makes the autotuning
              process an optional process, such that user can opt out.

        We choose not to have a standalone can_implement function, the tactics returned by get_valid_tactics should return
        valid kernel for these given input tensors.
        """
        return [-1]

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
def autotune(tune_mode: bool = True):
    old_mode = AutoTuner.get().is_tuning_mode
    AutoTuner.get().is_tuning_mode = tune_mode
    autotune_enabled = tune_mode and not old_mode
    if autotune_enabled:
        logger.info("[Autotuner]: Autotuning process starts ...")
    try:
        yield
    finally:
        AutoTuner.get().is_tuning_mode = old_mode
        if autotune_enabled:
            logger.info("[Autotuner]: Autotuning process ends")


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

    _instance = None

    def __init__(self, warmup=3, repeat=10, stream_delay_micro_secs=1000):
        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs
        self.profiling_cache = {}
        self.is_tuning_mode = False

        # Add statistics tracking
        self.stats = AutoTunerStatistics()

        self.profiling_debug = True

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = AutoTuner()
        return cls._instance

    def search_cache(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
    ) -> Tuple[bool, int, int, OptimizationProfile]:
        """Search for cached profiling results matching the current configuration.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            profile (OptimizationProfile): Optimization profile

        Returns:
            A tuple containing:
            [is_cache_hit, runner_id, tactic, stored_profile]
        """
        for r in runners:
            cache_key = AutoTuner._get_cache_key(
                custom_op, r, input_shapes, tuning_config
            )
            if (
                os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0") == "1"
                and not self.is_tuning_mode
            ):
                output = load_from_file(cache_key)
                return output
            elif cache_key in self.profiling_cache:
                return True, *self.profiling_cache[cache_key]

        return False, 0, -1, None

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

        input_shapes = tuple(self._get_input_sizes(inputs))

        # Early return if it's not tuning, use cache found one or fallback one
        if not self.is_tuning_mode:
            is_cache_hit, runner_id, tactic, stored_profile = self.search_cache(
                custom_op, runners, input_shapes, tuning_config
            )
            runner = runners[runner_id]
            # TODO: check the stored runner and tactic can implement this shape here
            # Should not directly try (runner, tactic) here, or it will hurt a lot of inference perf.

            # Record the cache miss config.
            # Expect no cache miss in inference. Thus, any cache miss should be recorded.
            if not is_cache_hit:
                logger.debug(
                    f"[AutoTunner]: Using fallback tactic for {custom_op} with input shapes {input_shapes}"
                )
                logger.debug(
                    f"[AutoTunner]: Generated key{AutoTuner._get_cache_key(custom_op, runners[0], input_shapes, tuning_config)}"
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
                param.name for param in inspect.signature(r.forward).parameters.values()
            }

        for p in profiles:
            tensors = self._prepare_input_tensors(p, inputs)
            is_cache_hit, runner_id, tactic, _ = self.search_cache(
                custom_op, runners, p.get_opt_shapes(), tuning_config
            )
            if not is_cache_hit:
                min_time = float("inf")
                # Initialize runner and tactic as None in case of no valid tactic or runners are found
                runner_id, tactic = None, None
                for r_id, r in enumerate(runners):
                    # TODO: use FakeTensor here.
                    valid_tactics = r.get_valid_tactics(tensors, p)
                    runner_arg_names = runner_arg_names_map[r]
                    if "do_preparation" in runner_arg_names and len(valid_tactics) > 0:
                        r(tensors, tactic=-1, do_preparation=True, **kwargs)
                    for tac in valid_tactics:
                        try:
                            time_measured = self._profile_single_kernel(
                                r, tensors, tac, **kwargs
                            )
                        except Exception as e:
                            shapes = self._get_input_sizes(tensors)
                            logger.warning(
                                f"[Autotuner]: Skipping tactic {r} {tac}, due to failure while profiling: {e}"
                            )

                            # Log stacktrace as debug to not spam log
                            logger.debug(
                                f"[Autotuner]: Failed when profiling {r} {tac}, shapes={shapes}. Error occurred: {e}"
                            )

                            # Record the failed profiling combinations
                            if custom_op not in self.stats.failed_profiling_count:
                                self.stats.failed_profiling_count[custom_op] = set()
                            self.stats.failed_profiling_count[custom_op].add(
                                AutoTuner._get_cache_key(
                                    custom_op, r, p.get_opt_shapes(), tuning_config
                                )
                            )

                            # Set time_measured to inf to notify the failure of the tactic. This can happen when `get_valid_tactics` mistakenly return wrong tactics
                            # or some runtime error occurs during profiling.
                            time_measured = float("inf")
                        if time_measured < min_time:
                            min_time = time_measured
                            runner_id, tactic = r_id, tac
                if runner_id is not None:
                    # At least one valid (runner, tactic) pair is found
                    cache_key = AutoTuner._get_cache_key(
                        custom_op, runners[runner_id], p.get_opt_shapes(), tuning_config
                    )
                    # inspect call stack
                    self.profiling_cache[cache_key] = (runner_id, tactic, p)
                    self.stats.tuned_op_successful_configs[custom_op] = (
                        self.stats.tuned_op_successful_configs.get(custom_op, 0) + 1
                    )
                    logger.debug(
                        f"[Autotuner]: profiling chosen runner: {runners[runner_id]} {tactic} for {cache_key}"
                    )

        # Get the best runner and tactic from cache
        # If no valid tactic is found, the fallback runner and tactic will be used
        _, runner_id, tactic, _ = self.search_cache(
            custom_op, runners, input_shapes, tuning_config
        )

        return runners[runner_id], tactic

    def _get_input_sizes(self, inputs: List[torch.Tensor]) -> List[torch.Size]:
        # Handle None tensors for optional inputs and non-Tensor scalar values
        sizes = [
            input.size() if isinstance(input, torch.Tensor) else torch.Size((0,))
            for input in inputs
        ]

        return sizes

    def _profile_single_kernel(
        self, runner: TunableRunner, inputs: List[torch.Tensor], tactic: int, **kwargs
    ) -> float:
        """Profile a single kernel implementation for performance measurement.

        Args:
            runner (TunableRunner): The runner implementation to profile
            inputs (List[torch.Tensor]): Input tensors for the kernel
            tactic (int): Tactic ID to use for this profiling run

        Returns:
            Average execution time in milliseconds

        Note:
            The method performs warmup runs, then measures multiple iterations
            to get an average execution time. Stream synchronization and delays
            are used to ensure accurate timing.
        """
        stream = torch.cuda.current_stream()
        # warm up, no timing
        for _ in range(self.warmup):
            runner(inputs, tactic=tactic, **kwargs)
        stream.synchronize()

        # Delay the profiled kernel launch to eliminate affects of host time overhead in profiling.
        # TODO: This is build time sensitive, O(tactic_num * impl_num * num_profile * tunable_ops)
        # Consider apply a preprofiling to estimate the kernel execution time, then decide the necessity.
        if self.stream_delay_micro_secs > 0:
            delay_kernel(self.stream_delay_micro_secs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream=stream)
        for _ in range(self.repeat):
            runner(inputs, tactic=tactic, **kwargs)
        end.record(stream=stream)
        stream.synchronize()

        avg_time = start.elapsed_time(end) / self.repeat

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
            base_profile[spec.input_idx[0]][spec.dim_idx[0]] = (
                spec.map_to_tuning_buckets(
                    base_profile[spec.input_idx[0]][spec.dim_idx[0]]
                )
            )

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
    ) -> Tuple:
        return (
            custom_op,
            runner.__class__.__name__,
            hash(runner),
            cls._find_nearest_profile(input_shapes, tuning_config),
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
        self, profile: OptimizationProfile, inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        default_initializer = lambda shapes, dtype, device: (
            torch.rand(shapes, device=device) * 10 - 5
        ).to(dtype)
        tensors = []
        for i, p in enumerate(profile.shapes):
            if any(isinstance(d, DynamicDim) for d in p):
                tensor = self._create_tensor_like(
                    inputs[i],
                    p,
                    profile.tensor_initializers[i] or default_initializer,
                )
            else:
                tensor = inputs[i]
            tensors.append(tensor)
        return tensors

    def clear_cache(self) -> None:
        """Clear the profiling cache."""
        self.profiling_cache.clear()

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()
