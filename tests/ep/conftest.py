# tests/ep/conftest.py
#
# Pytest fixtures for FlashInfer Unified EP API tests.
# Launch with: torchrun --nproc_per_node=<N> -m pytest tests/ep/ -v
#
# Fixtures handle:
#   - torch.distributed init/teardown (session-scoped)
#   - EP process group creation
#   - Backend and layout parameterization
#   - Model config parameterization (Mixtral / medium / DeepSeek-V3)
#   - EpGroup factory with automatic cleanup
#
# Plain helper functions (make_tokens, identity_expert, get_gpu_arch, etc.)
# live in tests/ep/helpers.py and are imported directly by test modules.

import os

import pytest
import torch
import torch.distributed as dist

import flashinfer.ep as fep

import importlib.util, pathlib
_helpers_path = pathlib.Path(__file__).parent / "helpers.py"
_spec = importlib.util.spec_from_file_location("ep_helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
BACKENDS = _helpers.BACKENDS
BACKENDS_WITH_NIXL = _helpers.BACKENDS_WITH_NIXL
LL_BACKENDS = _helpers.LL_BACKENDS
LAYOUTS = _helpers.LAYOUTS


# ─── Multi-GPU setup ────────────────────────────────────────────────


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))


def get_rank():
    return int(os.environ.get("RANK", 0))


@pytest.fixture(scope="session")
def dist_env():
    """Initialize torch.distributed for multi-GPU tests.

    Launch with:
        torchrun --nproc_per_node=4 -m pytest tests/ep/
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    yield {"rank": rank, "world_size": dist.get_world_size()}
    dist.destroy_process_group()


@pytest.fixture(scope="session")
def ep_process_group(dist_env):
    """Create a dedicated process group for EP tests."""
    pg = dist.new_group(
        ranks=list(range(dist_env["world_size"])), backend="nccl"
    )
    yield pg


# ─── Parameterized backend fixture ──────────────────────────────────


@pytest.fixture(params=BACKENDS, ids=["deepep", "nccl_ep"])
def backend(request):
    return request.param


def _backend_id(b):
    """Generate test ID for a Backend enum value."""
    return b.value  # "deepep", "nccl_ep", "nixl_ep"


@pytest.fixture(params=BACKENDS_WITH_NIXL, ids=lambda b: _backend_id(b))
def backend_with_nixl(request):
    """Parameterized backend fixture including NIXL-EP (if available)."""
    return request.param


@pytest.fixture(params=LL_BACKENDS, ids=lambda b: _backend_id(b))
def ll_backend(request):
    """Parameterized backend fixture for LL-mode-only tests."""
    return request.param


@pytest.fixture(params=LAYOUTS, ids=["flat_2d", "expert_major_3d"])
def output_layout(request):
    return request.param


# ─── Standard model configs ─────────────────────────────────────────


@pytest.fixture(
    params=[
        {"num_experts": 8, "top_k": 2, "hidden_dim": 4096, "name": "mixtral"},
        {"num_experts": 64, "top_k": 8, "hidden_dim": 4096, "name": "medium"},
        {
            "num_experts": 256,
            "top_k": 8,
            "hidden_dim": 7168,
            "name": "deepseek_v3",
        },
    ],
    ids=lambda c: c["name"],
)
def model_config(request, dist_env):
    cfg = request.param.copy()
    cfg["num_local_experts"] = cfg["num_experts"] // dist_env["world_size"]
    return cfg


# ─── Lightweight config for quick smoke tests ───────────────────────


@pytest.fixture
def small_config(dist_env):
    """Minimal config for fast tests — 8 experts, top-2, small hidden."""
    return {
        "num_experts": 8,
        "top_k": 2,
        "hidden_dim": 1024,
        "num_local_experts": 8 // dist_env["world_size"],
        "name": "small",
    }


# ─── Group factory with automatic cleanup ───────────────────────────


@pytest.fixture
def make_group(ep_process_group):
    """Factory fixture that creates EpGroups and auto-destroys them."""
    groups = []

    def _make(backend, model_config, **kwargs):
        g = fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=model_config["num_experts"],
            num_local_experts=model_config["num_local_experts"],
            top_k=model_config["top_k"],
            hidden_dim=model_config["hidden_dim"],
            **kwargs,
        )
        groups.append(g)
        return g

    yield _make

    for g in groups:
        try:
            g.destroy()
        except Exception:
            pass  # Best-effort cleanup
