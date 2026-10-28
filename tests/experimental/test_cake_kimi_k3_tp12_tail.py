"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Kimi-K3 TP12 fused LatentMoE tail (Cake backend).

The CPU tests check the host runtime (partition, route selection, generated
program inventory) on any machine.  The single-GPU test builds and loads every
registered program for the device's architecture (SM100 / SM103, no process
group).  The numerical GPU test needs the operator's real platform, a
twelve-rank multi-node NVLink domain (three GB200 / GB300 NVL72 compute
trays); launch it with torchrun over the three nodes::

    torchrun --nnodes 3 --nproc-per-node 4 --node-rank <n> --master-addr <host> \
        -m pytest tests/experimental/test_cake_kimi_k3_tp12_tail.py -k twelve_ranks

Single-process runs skip it.
"""

import os

import pytest
import torch

from flashinfer.experimental.kimi_k3_tp12_tail import cake_backend as cb
from flashinfer.experimental.kimi_k3_tp12_tail import cake_jit

ATOL = RTOL = 1e-2
SEED = 620
# One-launch rows (M <= 8: the four-token program at 1 / 4, the eight-token program at 6 / 8), the one-shot ESS
# rows, and the rows on both sides of every route-table boundary.
GPU_ROWS = (1, 4, 6, 8, 16, 32, 128, 256, 300, 319, 320, 960, 961, 962)
GPU_MAX_TOKENS = 1024


def test_partition_covers_hidden_in_128_column_blocks():
    assert len(cb.PARTITION) == cb.WORLD_SIZE == 12
    assert sum(cb.PARTITION) == cb.HIDDEN == 7168
    assert all(width % 128 == 0 for width in cb.PARTITION)
    assert cb.COL_BEGIN[0] == 0 and cb.COL_BEGIN[-1] == cb.HIDDEN
    assert cb.col_begin(cb.PARTITION) == cb.COL_BEGIN
    with pytest.raises(ValueError):
        cb.col_begin((640,) * 12)


def test_route_selection_by_token_count():
    # one-launch regime: the ESS one-shot K1 and the fused slice GEMM / tail as two CTA roles of one grid, program by
    # the accumulator capacity ladder (the four-token program for M <= 4, the eight-token program for 5 <= M <= 8)
    assert cb.K23_MAX_TOKENS == 8 and cb.K23_ROWS == 8 and cb.K23_LADDER == (4, 8)
    for M in (1, 2, 3, 4, 5, 6, 7, 8):
        capacity = 4 if M <= 4 else 8
        assert cb.k23_capacity_for(M) == capacity
        for rank in range(cb.WORLD_SIZE):
            # one program per capacity on every rank: the rank is a launch argument, the column width a grid size
            assert cb.route_kernel_keys(M, rank) == (f"k123:c{capacity}",)
            assert cb.k1_kernel_key(M, rank) == f"k123:c{capacity}"
        assert cb.k3_form_for(M) == "k123"
        # no tail launch of its own
        with pytest.raises(ValueError):
            cb.k3_kernel_key(M, 0)
        with pytest.raises(ValueError):
            cb.k3_grid(M, 152, 0)
    assert cb.up_proj_form_for(8) == "k23" and cb.up_proj_form_for(9) == "cublas"
    with pytest.raises(ValueError):
        cb.k23_capacity_for(9)
    # the token CTAs, then one K23 CTA per K23_ROWS output columns of the rank (80 for 640, 64 for 512)
    assert cb.k123_grid(1, 0) == (81, 1, 1)
    assert cb.k123_grid(4, 11) == (68, 1, 1)
    assert cb.k123_grid(8, 0) == (88, 1, 1)
    with pytest.raises(ValueError):
        cb.k123_grid(9, 0)
    # cuBLAS + K3-ESS regime: the ESS K1 scattered the shared partial, one CTA per token and column half
    for M in (9, 12, 16):
        for rank in range(cb.WORLD_SIZE):
            assert cb.route_kernel_keys(M, rank) == ("k1_oneshot_ess", "k3_ess:grouped")
    # route table above the one-shot rows: (first M, form)
    assert cb.ROUTE_TABLE == (
        (17, "ess"),
        (320, "reduce"),
        (961, "persist"),
    )
    assert cb.GROUPED_MAX_TOKENS == 960
    assert cb.K3_PERSIST_MIN_TOKENS == cb.BULK_MIN_TOKENS == 961
    expected = {
        17: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        32: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        128: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        255: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        256: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        257: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        319: ("k1_twoshot_ess:grouped", "k3_ess:grouped"),
        320: ("k1_twoshot_ess:grouped", "k3_reduce:grouped"),
        512: ("k1_twoshot_ess:grouped", "k3_reduce:grouped"),
        960: ("k1_twoshot_ess:grouped", "k3_reduce:grouped"),
        961: ("k1_twoshot:pinned", "k3_persist_bulk:pinned"),
        962: ("k1_twoshot:pinned", "k3_persist_bulk:pinned"),
        1024: ("k1_twoshot:pinned", "k3_persist_bulk:pinned"),
        2048: ("k1_twoshot:pinned", "k3_persist_bulk:pinned"),
        4096: ("k1_twoshot:pinned", "k3_persist_bulk:pinned"),
    }
    for M, keys in expected.items():
        for rank in (0, 3, 11):
            assert cb.route_kernel_keys(M, rank) == keys, M
    for M in (1, 4, 5, 8):
        assert len(cb.route_kernel_keys(M, 0)) == 1
    assert cb.k3_form_for(8) == "k123" and cb.k3_form_for(9) == "ess"
    for M, keys in expected.items():
        tail = keys[-1].split(":")[0]
        assert cb.k3_form_for(M) == {"k3_ess": "ess", "k3_reduce": "reduce"}.get(
            tail, "persist_bulk"
        ), M
    for M in (9, 16):
        with pytest.raises(ValueError):
            cb.route_form_for(M)
    # the plain grouped two-shot K1 and the plain persistent K3 stay registered, but no token count selects them
    selected = {key for M in range(1, 4097) for key in cb.route_kernel_keys(M, 0)}
    assert selected == set(cake_jit.required_kernel_keys()) - {
        "k1_twoshot:grouped",
        "k3_persist:grouped",
    }
    # K3-ESS: one CTA per token and column half; persistent forms (reduce mode, bulk): min(M, SM count)
    assert cb.k3_grid(9, 152, 0) == (9, 2, 1)
    assert cb.k3_grid(17, 148, 3) == (17, 2, 1)
    assert cb.k3_grid(17, 152, 3) == (17, 2, 1)
    assert cb.k3_grid(100, 152, 3) == (100, 2, 1)
    assert cb.k3_grid(255, 148, 3) == (255, 2, 1)
    assert cb.k3_grid(255, 152, 3) == (255, 2, 1)
    assert cb.k3_grid(319, 148, 3) == (319, 2, 1)
    assert cb.k3_grid(319, 152, 3) == (319, 2, 1)
    assert cb.k3_grid(320, 148, 3) == (148, 2, 1)
    assert cb.k3_grid(320, 152, 3) == (152, 2, 1)
    assert cb.k3_grid(960, 148, 3) == (148, 2, 1)
    assert cb.k3_grid(960, 152, 3) == (152, 2, 1)
    assert cb.k3_grid(961, 148, 3) == (148, 2, 1)
    assert cb.k3_grid(961, 152, 3) == (152, 2, 1)
    assert cb.k3_grid(4096, 148, 3) == (148, 2, 1)
    assert cb.k3_grid(4096, 152, 3) == (152, 2, 1)
    with pytest.raises(ValueError):
        cb.k3_grid(4096, 0, 3)
    assert len(cake_jit.required_kernel_keys()) == 10
    assert set(cake_jit.required_kernel_keys()) == {
        "k1_oneshot_ess",
        "k1_twoshot_ess:grouped",
        "k1_twoshot:grouped",
        "k1_twoshot:pinned",
        "k123:c4",
        "k123:c8",
        "k3_ess:grouped",
        "k3_reduce:grouped",
        "k3_persist:grouped",
        "k3_persist_bulk:pinned",
    }


def test_workspace_sizing():
    sizes = cb.workspace_buffer_bytes(4096)
    assert sizes["k1_oneshot"] == 16 * 3584 * 12 * 2
    assert sizes["k1_twoshot"] == 2 * 4104 * 3584 * 2
    assert sizes["k3"] == 2 * 4096 * 12 * 640 * 2
    assert cb.workspace_buffer_bytes(1)["k1_oneshot"] == 3584 * 12 * 2
    with pytest.raises(ValueError):
        cb.workspace_buffer_bytes(0)


def test_operand_validation_is_shape_and_dtype_strict():
    good = torch.empty((4, cb.LATENT), dtype=torch.bfloat16)
    cb._check(good, (4, cb.LATENT), "routed_partial")
    with pytest.raises(ValueError):
        cb._check(good.float(), (4, cb.LATENT), "routed_partial")
    with pytest.raises(ValueError):
        cb._check(good[:, ::2], (4, cb.LATENT // 2), "routed_partial")
    with pytest.raises(ValueError):
        cb._check(good, (4, cb.HIDDEN), "shared_partial")
    with pytest.raises(TypeError):
        cb._check(None, (4, cb.LATENT), "routed_partial")


def test_generated_module_inventory():
    if not cake_jit.MODULES:
        pytest.skip(
            "no generated Kimi-K3 TP12 tail program is registered in this checkout"
        )
    required = set(cake_jit.required_kernel_keys())
    # one program per logical kernel, each built for both architectures, launched with PDL
    assert set(cake_jit.KERNELS) == required
    assert len(set(cake_jit.KERNELS.values())) == len(cake_jit.MODULES)
    for key, name in cake_jit.KERNELS.items():
        record = cake_jit.MODULES[name]
        assert set(record["arches"]) == set(cake_jit.ARCHES), key
        assert record["ffi_entry"] == "run"
        assert len(record["sources"]) == 2
        assert {kind for kind, _ in record["arg_plan"]} <= {
            "buffer",
            "raw_pointer",
            "parameter",
            "grid",
        }
        assert record["launch"]["use_pdl"] is True
        # the one-launch program's CTAs wait for CTAs of the same grid: it launches cooperatively, the others do not
        assert record["launch"]["cooperative"] is key.startswith("k123:"), key
        assert tuple(record["launch"]["block"]) == (cb.THREADS, 1, 1)
    for arch in cake_jit.ARCHES:
        assert cake_jit.route_available(arch, tuple(required))
        for key in required:
            assert cake_jit.kernel_module_name(arch, key) == cake_jit.KERNELS[key]


class _FakeCommBackend:
    """A host-only stand-in for the comm backend's ``allgather``: every rank reports ``counts``."""

    def __init__(self, counts):
        self.counts = tuple(counts)
        self.sent = []

    def allgather(self, data):
        self.sent.append(data)
        return self.counts


def test_sm_count_is_the_group_minimum():
    # the persistent tails map token t to CTA t mod P on every rank: P is the group's smallest SM count
    backend = _FakeCommBackend([152] * 11 + [148])
    assert cb.group_sm_count(backend, 152) == 148
    assert backend.sent == [152]
    assert cb.group_sm_count(_FakeCommBackend([152] * 12), 152) == 152
    assert cb.group_sm_count(_FakeCommBackend([148] * 12), 148) == 148
    with pytest.raises(ValueError):
        cb.group_sm_count(_FakeCommBackend([152] * 12), 0)
    with pytest.raises(ValueError):
        cb.group_sm_count(_FakeCommBackend([152] * 11 + [0]), 152)


def test_refused_cooperative_launch_raises_cleanly(monkeypatch):
    """A refused cooperative launch of the one-launch program raises a RuntimeError that names the SM need, after
    the runtime's last error is cleared; other launch errors pass through unchanged (CPU only)."""
    cleared = []
    monkeypatch.setattr(cb, "_clear_cuda_last_error", lambda: cleared.append(True))

    def launch_of(message):
        def entry():
            raise RuntimeError(f"cudaLaunchKernelExC failed: {message}")

        return cb._Launch(
            stage="k1", key="k123:c4", module="m", kwargs={}, entry=entry, arguments=()
        )

    need = f"needs {cb.k123_grid(1, 0)[0]} SMs resident at once"
    with pytest.raises(RuntimeError, match=need) as info:
        cb._launch_one_launch_program(
            launch_of("too many blocks in cooperative launch"), 1, 0
        )
    assert cleared == [True]
    assert info.value.__cause__ is None and info.value.__suppress_context__
    with pytest.raises(RuntimeError, match="invalid argument"):
        cb._launch_one_launch_program(launch_of("invalid argument"), 1, 0)
    assert cleared == [True]


class _LaunchKwargsWorkspace:
    """A host-only stand-in for ``KimiK3Tp12TailWorkspace``: every buffer-set accessor the launch-argument
    builders use, with placeholder values (the arguments are never launched)."""

    sm_count = 148
    my_col_begin = 0
    my_cols = cb.HIDDEN // cb.WORLD_SIZE
    peer_ptrs = {"k1_oneshot": None, "k1_twoshot": None, "k3": None}

    @staticmethod
    def multicast_ptr(name):
        return name

    @staticmethod
    def local_unicast_ptr(name):
        return name

    @staticmethod
    def flags(name):
        return name


def test_launch_kwargs_cover_every_generated_argument_plan():
    """Every route key's host-built launch arguments satisfy the generated program's argument plan (CPU only).

    Guards the key -> kwargs dispatch against the registered ``arg_plan`` (the one-shot ESS key carries no
    poll-schedule suffix, so an exact-key test is required there)."""
    if not cake_jit.MODULES:
        pytest.skip(
            "no generated Kimi-K3 TP12 tail program is registered in this checkout"
        )
    workspace = _LaunchKwargsWorkspace()
    cases = [
        (M, rank, cb.route_kernel_keys(M, rank))
        for M in (1, 4, 5, 8, 9, 16, 17, 128, 255, 256, 300, 319, 320, 960, 961, 1024)
        for rank in (0, 5, 11)
    ]
    # registered programs no token count selects: the plain grouped two-shot K1 and the plain persistent K3
    cases += [
        (300, rank, ("k1_twoshot:grouped", "k3_persist:grouped")) for rank in (0, 5, 11)
    ]
    seen: set[str] = set()
    for M, rank, keys in cases:
        k1_key = keys[0]
        name, k1_kwargs = cb._k1_launch_kwargs(
            k1_key,
            M=M,
            rank=rank,
            workspace=workspace,
            routed_partial=None,
            shared_partial=None,
            y=None,
            norm_weight=None,
            up_weight_slice=None,
            out=None,
        )
        assert name == ("k1_oneshot" if M <= cb.ONESHOT_MAX_TOKENS else "k1_twoshot")
        launches = [(k1_key, k1_kwargs)]
        if M <= cb.K23_MAX_TOKENS:
            # the one-launch program: the token CTAs, then the rank's K23 CTAs
            assert len(keys) == 1
            assert k1_kwargs["grid"] == (
                M + workspace.my_cols // cb.K23_ROWS,
                1,
                1,
            )
        else:
            (k3_key,) = keys[1:]
            k3_kwargs = cb._k3_launch_kwargs(
                k3_key,
                M=M,
                rank=rank,
                workspace=workspace,
                grid=cb.k3_grid(M, workspace.sm_count, rank),
                shared_partial=None,
                gemm=None,
                out=None,
            )
            launches.append((k3_key, k3_kwargs))
        for key, kwargs in launches:
            record = cake_jit.MODULES[cake_jit.KERNELS[key]]
            expected = {name for kind, name in record["arg_plan"] if kind != "grid"}
            assert expected <= set(kwargs), (key, sorted(expected - set(kwargs)))
            assert len(kwargs["grid"]) == 3, key
            seen.add(key)
    assert seen == set(cake_jit.required_kernel_keys())


def _single_gpu_platform() -> str:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability(0) not in cb.SUPPORTED_COMPUTE_CAPABILITIES:
        return "generated kernels target Blackwell SM100 / SM103"
    return ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_clear_cuda_last_error_reaches_the_runtime_torch_reads():
    """The clear resets the last error of the libcudart that torch's launch checks read: an error planted there by
    a failing runtime call no longer surfaces in the next torch op (one GPU)."""
    import ctypes

    torch.ones(1, device="cuda").sum().item()
    major = torch.version.cuda.split(".")[0]
    cudart = ctypes.CDLL(f"libcudart.so.{major}", mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    cudart.cudaFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cudart.cudaFuncSetAttribute.restype = ctypes.c_int
    cudart.cudaPeekAtLastError.restype = ctypes.c_int
    # a null function: cudaErrorInvalidDeviceFunction, which is not sticky
    assert cudart.cudaFuncSetAttribute(None, 8, 0) != 0
    assert cudart.cudaPeekAtLastError() != 0
    cb._clear_cuda_last_error()
    assert cudart.cudaPeekAtLastError() == 0
    assert (torch.ones(1024, device="cuda") + 1).sum().item() == 2048


@pytest.mark.skipif(bool(_single_gpu_platform()), reason=_single_gpu_platform() or "")
def test_registered_programs_compile_on_this_device():
    """Every registered program builds and loads for this device's architecture (one GPU, no process group)."""
    if not cake_jit.MODULES:
        pytest.skip(
            "no generated Kimi-K3 TP12 tail program is registered in this checkout"
        )
    arch = cb.SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(0)]
    names = sorted(
        {
            cake_jit.kernel_module_name(arch, key)
            for key in cake_jit.required_kernel_keys()
        }
    )
    assert set(names) == set(cake_jit.MODULES)
    for name in names:
        module = cake_jit.load_cake_kimi_k3_tp12_tail_module(name)
        assert callable(getattr(module, cake_jit.MODULES[name]["ffi_entry"]))


# ---------------------------------------------------------------------------
# Twelve-rank GPU test (torchrun over three NVL72 compute trays)
# ---------------------------------------------------------------------------


def _twelve_rank_platform() -> str:
    if os.environ.get("WORLD_SIZE") != "12" or "RANK" not in os.environ:
        return "needs a torchrun launch with WORLD_SIZE=12 (three GB200 / GB300 NVL72 trays)"
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability(0) not in cb.SUPPORTED_COMPUTE_CAPABILITIES:
        return "generated kernels target Blackwell SM100 / SM103"
    return ""


def make_inputs(M: int, rank: int, device: torch.device, seed: int = SEED) -> dict:
    """Model-scale inputs: weights identical on every rank, per-rank partials."""
    g = torch.Generator(device=device).manual_seed(seed * 1000 + M)
    norm_w = (1.0 + 0.1 * torch.randn((cb.LATENT,), generator=g, device=device)).to(
        torch.bfloat16
    )
    up_w = (0.01 * torch.randn((cb.HIDDEN, cb.LATENT), generator=g, device=device)).to(
        torch.bfloat16
    )
    gr = torch.Generator(device=device).manual_seed(seed * 1000 + M + 17 * (rank + 1))
    routed = (0.3 * torch.randn((M, cb.LATENT), generator=gr, device=device)).to(
        torch.bfloat16
    )
    shared = (0.1 * torch.randn((M, cb.HIDDEN), generator=gr, device=device)).to(
        torch.bfloat16
    )
    out = torch.empty((M, cb.HIDDEN), dtype=torch.bfloat16, device=device)
    return dict(norm_w=norm_w, up_w=up_w, routed=routed, shared=shared, out=out)


def reference(inp: dict, world: int) -> torch.Tensor:
    """FP32 partial sums, one BF16 rounding of the latent, KimiRMSNorm round points, one final BF16 rounding."""
    import torch.distributed as dist

    rs = [torch.empty_like(inp["routed"]) for _ in range(world)]
    ss = [torch.empty_like(inp["shared"]) for _ in range(world)]
    dist.all_gather(rs, inp["routed"])
    dist.all_gather(ss, inp["shared"])
    routed_sum = torch.stack([t.float() for t in rs]).sum(0).to(torch.bfloat16)
    shared_sum = torch.stack([t.float() for t in ss]).sum(0)
    xf = routed_sum.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cb.RMS_EPS)
    y = inp["norm_w"] * xf.to(torch.bfloat16)
    gemm = y.float() @ inp["up_w"].float().t()
    return (gemm + shared_sum).to(torch.bfloat16)


def _rank_invariant(out: torch.Tensor, world: int) -> bool:
    import torch.distributed as dist

    outs = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(outs, out)
    return all(torch.equal(outs[0], t) for t in outs[1:])


@pytest.fixture(scope="module")
def twelve_rank_group():
    """The process group of the twelve-rank tests, initialised once for the module.

    A process group initialised again after ``destroy_process_group`` in the same torchrun process can read the
    previous group's rendezvous keys from the launcher's store, so the tests share one group.
    """
    import torch.distributed as dist

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local)
    yield rank, world, torch.device("cuda", local)
    if owns_group:
        dist.destroy_process_group()


@pytest.mark.skipif(bool(_twelve_rank_platform()), reason=_twelve_rank_platform() or "")
def test_tail_matches_reference_on_twelve_ranks(twelve_rank_group):
    import torch.distributed as dist

    from flashinfer.kimi_k3_tp12_tail import (
        create_kimi_k3_tp12_tail_workspace,
        prepare_kimi_k3_tp12_tail,
    )

    rank, world, device = twelve_rank_group
    assert world == cb.WORLD_SIZE
    if not cb.generated_program_available(device):
        pytest.skip("no generated program registered for this architecture")
    workspace = create_kimi_k3_tp12_tail_workspace(rank=rank, max_tokens=GPU_MAX_TOKENS)
    try:
        for M in GPU_ROWS:
            inp = make_inputs(M, rank, device)
            expected = reference(inp, world)
            runner = prepare_kimi_k3_tp12_tail(
                inp["routed"],
                inp["shared"],
                inp["norm_w"],
                inp["up_w"],
                inp["out"],
                workspace=workspace,
            )
            assert runner.kernel_keys == cb.route_kernel_keys(M, rank)
            assert runner.cublas == (M > cb.K23_MAX_TOKENS)
            if M <= cb.K23_MAX_TOKENS:
                # one launch: the one-launch program, no tail launch
                assert runner.k3 is None and runner.launch_count == 1
                assert runner.k1.kwargs["grid"] == cb.k123_grid(M, rank)
            else:
                assert runner.launch_count == 3
                assert runner.k3.kwargs["grid"] == cb.k3_grid(
                    M, workspace.sm_count, rank
                )
            inp["out"].fill_(float("nan"))
            runner()
            torch.cuda.synchronize()
            first = inp["out"].clone()
            assert torch.isfinite(first.float()).all(), f"M={M}: non-finite output"
            torch.testing.assert_close(
                first.float(), expected.float(), atol=ATOL, rtol=RTOL
            )
            assert _rank_invariant(first, world), f"M={M}: output differs between ranks"
            # idempotent second launch (Lamport buffers rotate and clear themselves)
            inp["out"].fill_(float("nan"))
            runner()
            torch.cuda.synchronize()
            assert torch.equal(inp["out"], first), f"M={M}: second launch differs"
            # CUDA-graph replay is bitwise identical to the eager launch
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                runner()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                runner()
            inp["out"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(inp["out"], first), f"M={M}: graph replay differs"
            del graph
        with pytest.raises(ValueError):
            inp = make_inputs(GPU_MAX_TOKENS + 1, rank, device)
            prepare_kimi_k3_tp12_tail(
                inp["routed"],
                inp["shared"],
                inp["norm_w"],
                inp["up_w"],
                inp["out"],
                workspace=workspace,
            )
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        workspace.destroy()


# One-launch rows of both programs and one row of every tail route.
STREAM_ROWS = (1, 4, 8, 64, 320, 961)
STREAM_CALLS = 24


@pytest.mark.skipif(bool(_twelve_rank_platform()), reason=_twelve_rank_platform() or "")
def test_alternating_streams_are_ordered_on_twelve_ranks(twelve_rank_group):
    """Calls that alternate between two streams with no caller synchronisation run one at a time (the workspace
    orders a launch from another stream after the previous one): every call is bitwise the single-stream result."""
    import torch.distributed as dist

    from flashinfer.kimi_k3_tp12_tail import (
        create_kimi_k3_tp12_tail_workspace,
        prepare_kimi_k3_tp12_tail,
    )

    rank, world, device = twelve_rank_group
    assert world == cb.WORLD_SIZE
    if not cb.generated_program_available(device):
        pytest.skip("no generated program registered for this architecture")
    workspace = create_kimi_k3_tp12_tail_workspace(rank=rank, max_tokens=GPU_MAX_TOKENS)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    try:
        for M in STREAM_ROWS:
            inp = make_inputs(M, rank, device)
            # one runner (and output) per stream over the one workspace: each stream touches only its own output
            outs = (inp["out"], torch.empty_like(inp["out"]))
            runners = [
                prepare_kimi_k3_tp12_tail(
                    inp["routed"],
                    inp["shared"],
                    inp["norm_w"],
                    inp["up_w"],
                    out,
                    workspace=workspace,
                )
                for out in outs
            ]
            runners[0]()
            torch.cuda.synchronize()
            first = outs[0].clone()
            copies = torch.empty(
                (STREAM_CALLS, *first.shape), dtype=first.dtype, device=device
            )
            for i in range(STREAM_CALLS):
                with torch.cuda.stream(streams[i % 2]):
                    outs[i % 2].fill_(float("nan"))
                    runners[i % 2]()
                    copies[i].copy_(outs[i % 2])
            torch.cuda.synchronize()
            for i in range(STREAM_CALLS):
                assert torch.equal(copies[i], first), f"M={M}: call {i} differs"
            assert _rank_invariant(first, world), f"M={M}: output differs between ranks"
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        workspace.destroy()


@pytest.mark.skipif(bool(_twelve_rank_platform()), reason=_twelve_rank_platform() or "")
def test_partition_below_the_grid_refuses_cleanly_on_twelve_ranks(twelve_rank_group):
    """On a green-context stream with fewer SMs than the one-launch program's grid, every rank's call raises a
    RuntimeError before any device work and leaves no CUDA error behind; the same runner then runs on the full
    device with its earlier result."""
    import torch.distributed as dist

    from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
    from flashinfer.kimi_k3_tp12_tail import (
        create_kimi_k3_tp12_tail_workspace,
        prepare_kimi_k3_tp12_tail,
    )

    rank, world, device = twelve_rank_group
    assert world == cb.WORLD_SIZE
    if not cb.generated_program_available(device):
        pytest.skip("no generated program registered for this architecture")
    workspace = create_kimi_k3_tp12_tail_workspace(rank=rank, max_tokens=GPU_MAX_TOKENS)
    try:
        streams, resources = split_device_green_ctx_by_sm_count(device, [32])
        partition_sms = int(resources[0].sm.smCount)
        for M in (1, 8):
            assert partition_sms < cb.k123_grid(M, rank)[0]
            inp = make_inputs(M, rank, device)
            runner = prepare_kimi_k3_tp12_tail(
                inp["routed"],
                inp["shared"],
                inp["norm_w"],
                inp["up_w"],
                inp["out"],
                workspace=workspace,
            )
            runner()
            torch.cuda.synchronize()
            first = inp["out"].clone()
            with (
                torch.cuda.stream(streams[0]),
                pytest.raises(RuntimeError, match="cooperative launch was refused"),
            ):
                runner()
            # no stale error: the next torch op launches and checks cleanly
            assert (torch.ones(1024, device=device) + 1).sum().item() == 2048
            inp["out"].fill_(float("nan"))
            runner()
            torch.cuda.synchronize()
            assert torch.equal(inp["out"], first), f"M={M}: full-device call differs"
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        workspace.destroy()
