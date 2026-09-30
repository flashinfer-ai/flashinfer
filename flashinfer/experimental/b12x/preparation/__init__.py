"""Declarative configuration and session-prepared plans."""
from .device import DetectedDevice, detect_device
from .session import PreparationJob, PreparationSession
from .session import prepare_default
from ._progress import PreparationDisplay
from .tuning import (
    BackendConfig, EligiblePlan, Knob, ParameterBinding, ParameterSpace,
    TuningConfiguration, TuningContract, make_fixed_contract,
)
from .types import (
    CollectiveRequirement, DeviceIdentity, FrozenMapping, MemoryRequirements,
    PersistentMemory, Plan, PreparationProgress, PreparationRequest,
    PreparationResult, PreparedCall, Selection, TuningCacheRequirement, TuningRequirement,
    current_plan, current_prepared_state, plan_from_handle, require_prepared,
)

__all__ = [
    "prepare_default",
    "BackendConfig", "CollectiveRequirement", "DetectedDevice", "DeviceIdentity",
    "EligiblePlan", "FrozenMapping", "Knob", "MemoryRequirements", "ParameterBinding",
    "ParameterSpace", "PersistentMemory", "Plan", "PreparationProgress",
    "PreparationJob", "PreparationSession", "PreparationDisplay",
    "PreparationRequest", "PreparationResult", "PreparedCall",
    "Selection", "TuningCacheRequirement", "TuningConfiguration", "TuningContract", "TuningRequirement",
    "detect_device", "make_fixed_contract",
    "current_plan", "current_prepared_state", "plan_from_handle", "require_prepared",
]
