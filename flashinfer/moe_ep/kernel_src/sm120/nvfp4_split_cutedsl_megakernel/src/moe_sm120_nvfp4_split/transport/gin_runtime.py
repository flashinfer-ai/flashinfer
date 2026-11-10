"""Version-locked GIN artifacts and explicit backend policy; no error fallback."""
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).with_name('gin_runtime')
def select_side(problem, world):
    if problem.num_tokens_per_rank <= 0:
        raise ValueError('num_tokens_per_rank must be positive')
    override = os.environ.get('MEGAMOE_NETWORK_BACKEND', 'auto')
    if override not in ('auto', 'gin'):
        raise ValueError('Only NCCL GIN is supported for network communication')
    diagnostic = os.environ.get('MEGA_GIN_SIDE')
    if diagnostic not in (None, 'both', 'auto'):
        raise ValueError('Split network backends are no longer supported')
    return 'both' if world > 4 else 'none'

def require_nccl_only_process():
    """Fail before allocating communication state in an incompatible framework."""
    modules = [name for name in sys.modules
               if name == 'nvshmem' or name.startswith('nvshmem.')]
    libraries = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6:
            name = Path(fields[-1]).name.lower()
            if name.startswith('lib') and 'nvshmem' in name:
                libraries.append(name)
    if modules or libraries:
        raise RuntimeError(
            'The NCCL-only MegaMoE runtime cannot use a process that already '
            'loads NVSHMEM. Activate release_changes/activate_gin_runtime.sh '
            'with the qualified clean Torch environment before importing Torch. '
            f'Loaded modules={modules}, libraries={sorted(set(libraries))}')

@lru_cache(maxsize=1)
def validate_artifacts():
    require_nccl_only_process()
    manifest = json.loads((ROOT/'build_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('GIN artifact/source hash mismatch: '+name)
    explicit = os.environ.get('MEGAMOE_NCCL_ROOT')
    if explicit:
        library = Path(explicit).expanduser().resolve()/'lib/libnccl.so.2'
    else:
        dist = importlib.metadata.distribution('nvidia-nccl-cu13')
        if dist.version != manifest['nccl_version']:
            raise RuntimeError('GIN requires pinned nvidia-nccl-cu13 '+manifest['nccl_version'])
        library = Path(dist.locate_file('nvidia/nccl/lib/libnccl.so.2')).resolve()
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest['nccl_library_sha256']:
        raise RuntimeError('GIN NCCL library differs from validated build; no backend fallback')
    return library

def device_bitcode():
    validate_artifacts()
    return str(ROOT/'gin_device.bc')
