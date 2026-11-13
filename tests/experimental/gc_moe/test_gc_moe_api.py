# SPDX-License-Identifier: Apache-2.0
"""Public surface, validation and lifecycle of the experimental GC MoE API."""

import os
import subprocess
import sys

import pytest
import torch

import flashinfer

from .reference import (
    BACKEND_BY_CAPABILITY,
    GEOMETRY,
    checkpoint_weights,
    stage,
    stage_tokens,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
LAZY_NAMES = ("gc_moe", "GCMoEWorkspace", "transform_gc_moe_weights")
DISPATCH_ENV = {
    "sm90": "GC_MOE_SM90_DISPATCH_SMS",
    "sm100": "GC_MOE_DISPATCH_SMS",
    "sm120": "GC_MOE_SM120_DISPATCH_SMS",
}


def _run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def _make_workspace(group):
    return flashinfer.GCMoEWorkspace(
        group,
        GEOMETRY["num_experts"],
        GEOMETRY["capacity"],
        GEOMETRY["num_topk"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
    )


def test_lazy_exports_keep_the_backend_unimported():
    result = _run_python(
        "import sys, flashinfer\n"
        f"assert all(name in dir(flashinfer) for name in {LAZY_NAMES!r})\n"
        "assert 'flashinfer.experimental.gc_moe' not in sys.modules\n"
        f"assert all(getattr(flashinfer, n).is_experimental for n in {LAZY_NAMES!r})\n"
        "assert 'flashinfer.experimental.gc_moe' not in sys.modules\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.cpu_only
def test_upstream_version_is_recorded():
    from flashinfer.experimental.gc_moe.kernel_src.gc_moe import (
        UPSTREAM_COMMIT,
        UPSTREAM_VERSION,
    )

    drop = os.path.join(
        REPO_ROOT, "flashinfer", "experimental", "gc_moe", "kernel_src", "gc_moe"
    )
    with open(os.path.join(drop, "src", "__init__.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert f"__version__ = '{UPSTREAM_VERSION}'" in source
    with open(os.path.join(drop, "VENDOR.md"), encoding="utf-8") as handle:
        vendor = handle.read()
    assert f"`{UPSTREAM_COMMIT}`" in vendor


@pytest.mark.cpu_only
def test_ptxas_resolution_reports_missing_assemblers(tmp_path):
    result = _run_python(
        "from flashinfer.experimental.gc_moe.kernel_src.gc_moe import resolve_ptxas\n"
        "print(resolve_ptxas(None))\n"
    )
    assert result.returncode == 0, result.stderr
    resolved = result.stdout.strip().splitlines()[-1]
    assert os.path.isfile(resolved) and os.access(resolved, os.X_OK), resolved
    requested = os.environ.get("GC_MOE_PTXAS")
    if requested:
        assert os.path.realpath(resolved) == os.path.realpath(requested), resolved

    missing = tmp_path / "ptxas"
    result = _run_python(
        "from flashinfer.experimental.gc_moe.kernel_src.gc_moe import resolve_ptxas\n"
        f"resolve_ptxas({str(missing)!r})\n"
    )
    assert result.returncode != 0
    assert "ptxas" in result.stderr


def test_workspace_reports_the_device_backend(gc_group):
    capability = tuple(torch.cuda.get_device_capability())
    with _make_workspace(gc_group) as workspace:
        assert workspace.backend == BACKEND_BY_CAPABILITY[capability]


def test_workspace_staging_views(workspace):
    rows = workspace.x.shape[0]
    assert rows >= GEOMETRY["capacity"]
    assert workspace.x.shape[1] == GEOMETRY["hidden"]
    assert workspace.x.dtype == torch.float8_e4m3fn
    assert workspace.x_sf.shape[0] == rows
    assert workspace.x_sf.dtype in (torch.int32, torch.float32)
    assert workspace.topk_idx.shape == (rows, GEOMETRY["num_topk"])
    assert workspace.topk_idx.dtype == torch.int64
    assert workspace.topk_weights.shape == (rows, GEOMETRY["num_topk"])
    assert workspace.topk_weights.dtype == torch.float32


def test_workspace_lifecycle(gc_group):
    created = _make_workspace(gc_group)
    assert created.x is not None
    created.destroy()
    with pytest.raises(AttributeError):
        _ = created.x
    created.destroy()  # idempotent


def test_workspace_destroy_releases_cached_weights(gc_group, backend):
    import gc
    import weakref

    device = torch.device("cuda", torch.cuda.current_device())
    l1_checkpoint, l2_checkpoint = checkpoint_weights(
        backend,
        GEOMETRY["num_experts"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
        device,
    )
    l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(
        l1_checkpoint, l2_checkpoint
    )
    refs = [weakref.ref(tensor) for pair in (l1_weights, l2_weights) for tensor in pair]
    workspace = _make_workspace(gc_group)
    x, topk_idx, topk_weights = stage_tokens(
        1,
        GEOMETRY["hidden"],
        GEOMETRY["num_experts"],
        GEOMETRY["num_topk"],
        device,
    )
    stage(workspace, x, topk_idx, topk_weights)
    y = torch.empty(1, GEOMETRY["hidden"], dtype=torch.bfloat16, device=device)
    flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
    torch.cuda.synchronize()
    assert all(ref() is not None for ref in refs)

    workspace.destroy()
    del l1_weights, l2_weights, l1_checkpoint, l2_checkpoint
    del x, topk_idx, topk_weights, y
    gc.collect()
    assert [ref() for ref in refs] == [None] * len(refs)


def test_rejects_foreign_workspace():
    with pytest.raises(TypeError, match="workspace must come from"):
        flashinfer.gc_moe(None, None, None, object())


def test_rejects_destroyed_workspace(gc_group):
    created = _make_workspace(gc_group)
    created.destroy()
    with pytest.raises(ValueError, match="destroyed"):
        flashinfer.gc_moe(None, None, None, created)


def test_rejects_backend_mismatch(workspace):
    other = {"sm90": "sm120", "sm100": "sm90", "sm120": "sm100"}[workspace.backend]
    with pytest.raises(ValueError, match="does not match the workspace backend"):
        flashinfer.gc_moe(None, None, None, workspace, backend=other)


@pytest.mark.parametrize("activation", ["gelu", "silu", "swiglu_oai"])
def test_rejects_unsupported_activation(workspace, activation):
    with pytest.raises(ValueError, match="swiglu"):
        flashinfer.gc_moe(None, None, None, workspace, activation=activation)


def test_rejects_half_a_shared_pair(workspace):
    with pytest.raises(ValueError, match="must be given together"):
        flashinfer.gc_moe(None, None, None, workspace, shared_l1_weights=object())


def test_rejects_arch_specific_switches(workspace, backend):
    with pytest.raises(ValueError, match="activation_alpha"):
        flashinfer.gc_moe(None, None, None, workspace, activation_alpha=1.0)
    if backend == "sm100":
        with pytest.raises(ValueError, match="num_tokens_bound"):
            flashinfer.gc_moe(None, None, None, workspace, num_tokens_bound=8)


def test_transform_rejects_foreign_arguments(backend):
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = (
        GEOMETRY["num_experts"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
    )
    l1_checkpoint, l2_checkpoint = checkpoint_weights(
        backend, experts, hidden, intermediate, device
    )
    if backend == "sm100":
        with pytest.raises(ValueError, match="shared_recipe"):
            flashinfer.transform_gc_moe_weights(
                l1_checkpoint, l2_checkpoint, shared_recipe=(1, 1, 32)
            )
    elif backend == "sm120":
        with pytest.raises(ValueError, match="scale_2"):
            flashinfer.transform_gc_moe_weights(
                l1_checkpoint,
                l2_checkpoint,
                l1_scale_2=torch.ones(experts, device=device),
                l2_scale_2=torch.ones(experts, device=device),
            )
    with pytest.raises(ValueError, match="swiglu"):
        flashinfer.transform_gc_moe_weights(
            l1_checkpoint, l2_checkpoint, activation="gelu"
        )


def test_shared_expert_count_is_derived_from_the_weight_pair():
    from flashinfer.experimental.gc_moe.dispatch import _call_shared_experts

    assert _call_shared_experts((torch.empty(4, 64, 128), None), 256) == 2
    assert _call_shared_experts((torch.empty(512, 256), None), 256) == 2
    assert _call_shared_experts((torch.empty(3, 64, 128), None), 256) is None
    assert _call_shared_experts((object(), None), 256) is None


def test_plan_rejects_unknown_backend():
    from flashinfer.experimental.gc_moe import GcMoePlan

    with (
        pytest.raises(ValueError, match="unknown GC MoE backend"),
        GcMoePlan(dispatch_sms=4).applied("sm99"),
    ):
        pass


def test_experimental_package_exports_the_runtime_close():
    import flashinfer.experimental.gc_moe as gc_moe_experimental

    assert callable(gc_moe_experimental.close_runtime)
    assert "close_runtime" in gc_moe_experimental.__all__


def test_runtime_switch_after_launch_is_rejected(gc_group, workspace, backend):
    device = torch.device("cuda", torch.cuda.current_device())
    l1_checkpoint, l2_checkpoint = checkpoint_weights(
        backend,
        GEOMETRY["num_experts"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
        device,
    )
    l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(
        l1_checkpoint, l2_checkpoint
    )
    x, topk_idx, topk_weights = stage_tokens(
        1,
        GEOMETRY["hidden"],
        GEOMETRY["num_experts"],
        GEOMETRY["num_topk"],
        device,
    )
    stage(workspace, x, topk_idx, topk_weights)
    y = torch.empty(1, GEOMETRY["hidden"], dtype=torch.bfloat16, device=device)
    flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
    torch.cuda.synchronize()

    name = DISPATCH_ENV[backend]
    from flashinfer.experimental.gc_moe import plan

    before = plan.environment_fingerprint(backend)
    previous = os.environ.get(name)
    try:
        for candidate in ("4", "8", "16", "32"):
            os.environ[name] = candidate
            if plan.environment_fingerprint(backend) != before:
                break
        else:
            pytest.skip("no dispatch-switch value changes the fingerprint")
        with pytest.raises(RuntimeError, match="runtime switches changed"):
            flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.mark.cpu_only
def test_context_keyed_backends_latch_switches_across_workspaces():
    import threading

    from flashinfer.experimental.gc_moe.dispatch import _check_launch_latch

    outcome = {}

    def body():
        _check_launch_latch("sm90", 0, ("8", None, None))
        _check_launch_latch("sm90", 0, ("8", None, None))
        _check_launch_latch("sm120", 0, ("6", None, None))
        # Another device is another CUDA context, so it keeps its own switches.
        _check_launch_latch("sm90", 1, ("16", None, None))
        # SM100 keys its pair cache by the dispatch request, so a changed
        # request selects another pair instead of reusing the first one.
        _check_launch_latch("sm100", 0, ("8", None, None))
        _check_launch_latch("sm100", 0, ("16", None, None))
        for backend, changed in (
            ("sm90", ("16", None, None)),
            ("sm120", ("8", None, None)),
        ):
            try:
                _check_launch_latch(backend, 0, changed)
            except RuntimeError as error:
                outcome[backend] = str(error)
            else:
                outcome[backend] = ""

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    assert "sm90" in outcome.get("sm90", "")
    assert "sm120" in outcome.get("sm120", "")
    assert "thread" in outcome["sm90"]


def test_plan_context_restores_the_environment():
    from flashinfer.experimental.gc_moe import GcMoePlan

    name = "GC_MOE_DISPATCH_SMS"
    before = os.environ.get(name)
    with GcMoePlan(dispatch_sms=12).applied("sm100"):
        assert os.environ[name] == "12"
    assert os.environ.get(name) == before


def test_primary_context_mode_is_pinned(workspace):
    assert os.environ.get("GC_MOE_PRIMARY_CTX") == "1"


def test_pinned_and_inert_switches_do_not_trip_the_guard(workspace, backend):
    device = torch.device("cuda", torch.cuda.current_device())
    l1_checkpoint, l2_checkpoint = checkpoint_weights(
        backend,
        GEOMETRY["num_experts"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
        device,
    )
    l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(
        l1_checkpoint, l2_checkpoint
    )
    x, topk_idx, topk_weights = stage_tokens(
        1,
        GEOMETRY["hidden"],
        GEOMETRY["num_experts"],
        GEOMETRY["num_topk"],
        device,
    )
    y = torch.empty(1, GEOMETRY["hidden"], dtype=torch.bfloat16, device=device)

    def launch():
        stage(workspace, x, topk_idx, topk_weights)
        flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)

    launch()
    torch.cuda.synchronize()
    first = y.clone()
    names = (
        "GC_MOE_PRIMARY_CTX",
        "GC_MOE_BALANCED_WORKQUEUES",
        "GC_MOE_SM90_BALANCED_WORKQUEUES",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "0"
        launch()
        torch.cuda.synchronize()
        assert torch.equal(y, first)
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def test_block_m_tracks_the_routing_load():
    from flashinfer.experimental.gc_moe import block_m

    assert block_m(1, 4, 64, 1, 2) == 16
    assert block_m(8, 8, 256, 256, 8) >= 64
