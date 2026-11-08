"""Cache contracts and collective measurement for complete EP selection."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from statistics import median
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .tuning_measurement import measure_samples


NAMESPACE = "moe_ep.complete_forward.v1"
BACKEND_NAMESPACE = "moe_ep.complete_forward.backends.v1"


def plain(value):
    """Encode configuration values without process-local object identities."""
    from flashinfer.quantization.nvfp4_quantization_utils import _UNSET
    from ..algo_knobs import FleetAlgoKnobAllocator, HandleAlgoKnobUserStream

    if value is _UNSET:
        return {"nvfp4_4over6": "environment"}
    if isinstance(value, HandleAlgoKnobUserStream) or (
        isinstance(value, FleetAlgoKnobAllocator)
        and (value.alloc_fn or value.free_fn or value.context)
    ):
        raise ValueError(
            "EP automatic selection cannot cache process-local allocator callbacks "
            "or stream handles"
        )
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__qualname__, "value": value.value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "fields": {
                f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(
        f"EP selection requires serializable configuration fields; got {type(value).__name__}"
    )


def record_key(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def tuples(value):
    if isinstance(value, (list, tuple)):
        return tuple(tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: tuples(v) for k, v in value.items()}
    return value


def tensor_layout(value):
    return (
        None
        if value is None
        else [str(value.dtype), list(value.shape), list(value.stride())]
    )


class NoEligibleCandidate(RuntimeError):
    """No legal candidate remains before any candidate kernel is launched."""


class Collective:
    def __init__(self, bootstrap):
        from .bootstrap_utils import bootstrap_comm_group, bootstrap_ep_rank_world

        self.group = bootstrap_comm_group(bootstrap)
        self.rank, self.world = bootstrap_ep_rank_world(bootstrap)

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value, group=self.group)
        return values

    def agree(self, value, name):
        if any(v != value for v in self.gather(value)):
            raise ValueError(f"EP ranks disagree on {name}")

    def prepare(self, fn, *, optional=False):
        result, error, unsupported = None, None, False
        try:
            result = fn()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            unsupported = isinstance(
                exc, (ValueError, TypeError, NotImplementedError, NoEligibleCandidate)
            )
        errors = self.gather((error, unsupported))
        if any(e[0] for e in errors):
            if optional and all(e[0] is None or e[1] for e in errors):
                destroy = getattr(result, "destroy", None)
                if destroy is not None:
                    destroy()
                raise NoEligibleCandidate(f"EP candidate is unsupported: {errors}")
            raise RuntimeError(f"EP candidate preparation failed: {errors}")
        return result

    def lookup(self, context, key, compatible):
        records = self.gather(context.lookup(key))
        validity = self.gather([compatible(r) for r in records])
        return next(
            (r for i, r in enumerate(records) if all(v[i] for v in validity)), None
        )

    def publish(self, context, key, record):
        def write():
            if self.rank == 0:
                context.publish(key, record)

        self.prepare(write)

    def measure(self, fn, device, settings, *, warmup=None, warmups=3, repeats=10):
        for _ in range(warmups):
            (warmup or fn)()
        torch.cuda.synchronize()
        trial = SimpleNamespace(
            config=SimpleNamespace(world_size=self.world), device=device, launch=fn
        )
        samples = measure_samples(trial, (), repeats, settings, self.group)
        values = torch.tensor(samples, dtype=torch.float64, device=device)
        if self.world > 1:
            dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.group)
        score = median(values.tolist())
        if not math.isfinite(score) or score <= 0:
            raise RuntimeError("EP timing returned a nonpositive or nonfinite score")
        return score
