"""Immutable, versioned offline measurements consumed without GPU readback.

Load once before graph capture. Matching is deliberately exact in v1: an
unmeasured token capacity uses the normal selector, never a nearest neighbor.
"""
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import warnings

SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def implementation_fingerprint():
    root = Path(__file__).resolve().parent
    digest = sha256()
    for p in sorted(root.rglob('*')):
        # Built-in selection rules choose complete configs; changing that table
        # does not change the execution of an independently tuned config.
        if (p.suffix in ('.py', '.cu') and p.name != 'model_policy.py'
                and not p.name.startswith('test_')
                and '__pycache__' not in p.parts):
            digest.update(str(p.relative_to(root)).encode() + b'\0')
            digest.update(p.read_bytes())
    return digest.hexdigest()


def hardware_identity(device=None):
    """Call during setup on the rank's selected CUDA device, not in a graph."""
    import torch
    from importlib import metadata
    p = torch.cuda.get_device_properties(device)
    # Include the selected compiler and library distributions, not only Torch's
    # CUDA build version. A DSL upgrade can change the winning tile/stage config.
    names = {
        d.metadata['Name'] for d in metadata.distributions()
        if d.metadata.get('Name', '').lower().replace('_', '-').startswith(
            ('nvidia-cutlass-dsl', 'nvidia-nccl-cu13', 'nvidia-cuda-nvrtc-cu12', 'torch'))
    }
    toolchain = {name: metadata.version(name) for name in sorted(names)}
    return json.dumps(dict(name=p.name, capability=[p.major, p.minor],
                           num_sms=p.multi_processor_count,
                           memory_bytes=p.total_memory,
                           torch_version=torch.__version__, cuda_version=torch.version.cuda,
                           compiler_libraries=toolchain),
                      sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class TuningConfig:
    """Content, rather than file path, identifies a selection-cache entry."""
    payload: str
    sha256: str
    error: str = ''


def parse_tuning_config(value):
    """Validate structure. Invalid files cannot replace the normal selector."""
    try:
        if not isinstance(value, dict) or value.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported tuning schema')
        if set(value) != {'schema_version', 'implementation_sha256', 'kernel_cache_abi',
                          'hardware', 'entries', 'provenance'}:
            raise ValueError('unexpected or missing tuning document fields')
        if not isinstance(value['hardware'], str) or not value['hardware']:
            raise ValueError('missing hardware identity')
        if not isinstance(value['entries'], list):
            raise ValueError('entries must be a list')
        seen = set()
        for entry in value['entries']:
            if set(entry) != {'shape', 'numerical_contract', 'config', 'validation', 'evidence'}:
                raise ValueError('unexpected or missing entry fields')
            key = json.dumps([entry['shape'], entry['numerical_contract']], sort_keys=True)
            if key in seen:
                raise ValueError('duplicate/ambiguous shape entry')
            seen.add(key)
            if entry['numerical_contract'] not in ('owner_bf16', 'ordinary_topk'):
                raise ValueError('invalid numerical contract')
            if entry['validation'] != 'qualified':
                raise ValueError('unqualified entry')
            if not isinstance(entry['evidence'], dict) or not entry['evidence']:
                raise ValueError('missing measurement evidence')
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return TuningConfig(payload, sha256(payload.encode()).hexdigest())
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return TuningConfig('', '', str(exc))


def load_tuning_config(path):
    """Read once; malformed/unreadable files issue a warning and fall back."""
    try:
        result = parse_tuning_config(json.loads(Path(path).read_text()))
        if not result.error:
            from .api import KERNEL_CACHE_ABI
            value = json.loads(result.payload)
            # Warm the source fingerprint during explicit setup/loading, so
            # subsequent selector calls only consult immutable host objects.
            if value['implementation_sha256'] != implementation_fingerprint():
                result = TuningConfig(result.payload, result.sha256, 'implementation fingerprint mismatch')
            elif value['kernel_cache_abi'] != KERNEL_CACHE_ABI:
                result = TuningConfig(result.payload, result.sha256, 'kernel ABI mismatch')
    except (OSError, ValueError) as exc:
        result = TuningConfig('', '', str(exc))
    if result.error:
        warnings.warn('Offline tuning ignored: ' + result.error, RuntimeWarning, stacklevel=2)
    return result


def config_from_dict(values, shape, contract):
    from .heuristic import (MegaMoEKernelConfig, derive_features, legality_errors,
                            semantic_errors, correctness_qualification_errors)
    if not isinstance(values, dict) or set(values) != {f.name for f in fields(MegaMoEKernelConfig)}:
        raise ValueError('incomplete or unknown kernel config fields')
    # Enforce JSON types before arithmetic/legality validation. In particular,
    # Python otherwise accepts True as an integer and strings as truth values.
    from typing import get_type_hints, get_args
    hints = get_type_hints(MegaMoEKernelConfig)
    converted = dict(values)
    for name, val in converted.items():
        if name in ('k1_tile', 'k2_tile'):
            if not isinstance(val, (tuple, list)) or len(val) != 3 or any(type(x) is not int for x in val):
                raise ValueError('invalid tile: ' + name)
            converted[name] = tuple(val)
        elif name == 'expected_rows_per_expert':
            if type(val) not in (int, float) or not math.isfinite(val) or val != shape.expected_rows_per_expert:
                raise ValueError('density does not match shape')
        else:
            typ = hints[name]
            allowed = get_args(typ) or (typ,)
            if type(val) not in allowed:
                raise ValueError('invalid config type: ' + name)
    cfg = MegaMoEKernelConfig(**converted)
    if cfg.comm_backend != 'p2p_direct' or cfg.kernel_comm_backend != 'p2p_direct':
        raise ValueError('offline v1 supports P2P direct configurations only')
    if (cfg.sm_min_partition != shape.sm_min_partition
            or cfg.sm_partition_alignment != shape.sm_partition_alignment):
        raise ValueError('config partition constraints differ from hardware')
    if cfg.k2_min_blocks_per_sm not in (1, 2):
        raise ValueError('invalid K2 occupancy')
    errors = (legality_errors(shape, cfg) + semantic_errors(cfg, contract)
              + correctness_qualification_errors(shape, cfg))
    if errors:
        raise ValueError('; '.join(errors))
    f = derive_features(shape, cfg)
    if f.k1_smem_estimate > 99 * 1024 or f.k2_smem_estimate > 99 * 1024 / cfg.k2_min_blocks_per_sm:
        raise ValueError('estimated shared memory exceeds capacity')
    return cfg


def overrides_match(config, overrides):
    return overrides is None or all(
        value is None or getattr(config, key) == value
        for key, value in asdict(overrides).items())


@lru_cache(maxsize=256)
def resolve_tuning(config, shape, overrides, contract, hardware):
    info = dict(source='offline_tuning', matched=False, digest=config.sha256)
    def fallback(reason):
        return None, dict(info, reason=reason)
    if config.error:
        return fallback(config.error)
    from .api import KERNEL_CACHE_ABI
    try:
        value = json.loads(config.payload)
        if value['kernel_cache_abi'] != KERNEL_CACHE_ABI:
            return fallback('kernel ABI mismatch')
        if value['implementation_sha256'] != implementation_fingerprint():
            return fallback('implementation fingerprint mismatch')
        if not hardware or value['hardware'] != hardware:
            return fallback('hardware/toolchain identity missing or mismatched')
        for entry in value['entries']:
            if entry['shape'] != asdict(shape) or entry['numerical_contract'] != contract:
                continue
            candidate = config_from_dict(entry['config'], shape, contract)
            if not overrides_match(candidate, overrides):
                return fallback('explicit overrides conflict with tuned config')
            return candidate, dict(info, matched=True, reason='exact qualified shape',
                                   evidence=entry['evidence'])
        return fallback('no measured entry for this shape and numerical contract')
    except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError) as exc:
        return fallback('invalid tuning entry: ' + str(exc))


def make_document(hardware, entries, provenance):
    from .api import KERNEL_CACHE_ABI
    return dict(schema_version=SCHEMA_VERSION,
                implementation_sha256=implementation_fingerprint(),
                kernel_cache_abi=KERNEL_CACHE_ABI, hardware=hardware,
                entries=entries, provenance=provenance)
