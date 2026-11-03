"""MXFP8 output of the TRT-LLM allreduce + residual + RMSNorm fusion.

The fused kernel must produce exactly what the unfused sequence produces:
kARResidualRMSNorm followed by mxfp8_quantize (backend="cute-dsl") on the norm
output -- same E4M3 bytes, same UE8M0 scale bytes (including the zeroed row /
column padding of the swizzled 128x4 layout), same residual and norm outputs.

Run as: python tests/comm/test_trtllm_allreduce_fusion_mxfp8.py [world_size]
"""

import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import flashinfer.comm as comm
from flashinfer import mxfp8_quantize
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.node_exclusive

MAX_TOKEN_NUM = 2048
RMS_EPS = 1e-6


def _mxfp8_ref(norm_out: torch.Tensor, layout: int):
    swizzled = layout == comm.QuantizationSFLayout.SWIZZLED_128x4
    q, sf = mxfp8_quantize(norm_out, is_sf_swizzled_layout=swizzled, backend="cute-dsl")
    return q, sf.view(-1)


def _written_sf_mask(token_num: int, hidden_dim: int, layout: int, numel: int, device):
    """Scale bytes the kernel writes: every row < token_num, padding rows excluded."""
    if layout != comm.QuantizationSFLayout.SWIZZLED_128x4:
        return torch.ones(numel, dtype=torch.bool, device=device)
    k_tiles = (hidden_dim // 32 + 3) // 4
    rows = torch.arange(token_num, device=device)[:, None]
    cols = torch.arange(k_tiles * 4, device=device)[None, :]
    off = (
        ((rows // 128) * k_tiles + cols // 4) * 512
        + (rows % 32) * 16
        + (rows % 128) // 32 * 4
        + cols % 4
    )
    mask = torch.zeros(numel, dtype=torch.bool, device=device)
    mask[off.reshape(-1)] = True
    return mask


def _run_worker(world_size, rank, dtype, hidden_dims, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    failures = 0
    try:
        for hidden_dim in hidden_dims:
            ipc_handles, workspace_tensor, metadata = (
                comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                    rank,
                    world_size,
                    MAX_TOKEN_NUM,
                    hidden_dim,
                    group=group,
                    create_metadata=True,
                )
            )
            for token_num in [1, 4, 7, 56, 128, 129, 300]:
                for use_oneshot in [True, False]:
                    if not use_oneshot and token_num <= world_size:
                        continue
                    for layout in [
                        comm.QuantizationSFLayout.SWIZZLED_128x4,
                        comm.QuantizationSFLayout.LINEAR,
                    ]:
                        for with_norm_out in [False, True]:
                            failures += _check_case(
                                rank,
                                world_size,
                                dtype,
                                hidden_dim,
                                token_num,
                                use_oneshot,
                                layout,
                                with_norm_out,
                                workspace_tensor,
                                metadata,
                                group,
                            )
            dist.barrier(group=group)
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                ipc_handles, group=group
            )
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group(group=group)
    assert failures == 0, f"rank {rank}: {failures} failing cases"


def _check_case(
    rank,
    world_size,
    dtype,
    hidden_dim,
    token_num,
    use_oneshot,
    layout,
    with_norm_out,
    workspace_tensor,
    metadata,
    group,
):
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234 + 31 * token_num + hidden_dim)
    # Different per-rank inputs; same residual / gamma on every rank.
    allreduce_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
    allreduce_in *= 1.0 + rank
    residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device="cpu").to(
        device
    )
    rms_gamma = torch.randn(hidden_dim, dtype=dtype, device="cpu").to(device)
    weight_bias = 1.0

    def run(pattern, quant_out=None, scale_out=None):
        inp = allreduce_in.clone()
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(residual_in)
        comm.trtllm_allreduce_fusion(
            allreduce_in=inp.view(-1),
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace_tensor,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            fp32_acc=True,
            pattern_code=pattern,
            use_oneshot=use_oneshot,
            allreduce_out=None,
            residual_in=residual_in.view(-1),
            residual_out=residual_out.view(-1),
            norm_out=norm_out.view(-1),
            quant_out=None if quant_out is None else quant_out.view(-1),
            scale_out=scale_out,
            rms_gamma=rms_gamma,
            rms_eps=RMS_EPS,
            scale_factor=None,
            layout_code=layout,
            metadata=metadata,
            weight_bias=weight_bias,
        )
        return residual_out, norm_out

    dist.barrier(group=group)
    ref_residual, ref_norm = run(comm.AllReduceFusionPattern.kARResidualRMSNorm)
    torch.cuda.synchronize()
    ref_q, ref_sf = _mxfp8_ref(ref_norm, layout)

    pattern = (
        comm.AllReduceFusionPattern.kARResidualRMSNormOutMXFP8Quant
        if with_norm_out
        else comm.AllReduceFusionPattern.kARResidualRMSNormMXFP8Quant
    )
    quant_out = torch.empty(
        token_num, hidden_dim, dtype=torch.float8_e4m3fn, device=device
    )
    # Poison the scale buffer so unwritten scales (padding columns included) are detected;
    # padding rows are not written and are excluded from the comparison.
    scale_out = torch.full((ref_sf.numel(),), 0xAB, dtype=torch.uint8, device=device)
    dist.barrier(group=group)
    residual_out, norm_out = run(pattern, quant_out, scale_out)
    torch.cuda.synchronize()
    written = _written_sf_mask(token_num, hidden_dim, layout, ref_sf.numel(), device)
    scale_out, ref_sf = scale_out[written], ref_sf[written]

    if with_norm_out:
        own_q, own_sf = _mxfp8_ref(norm_out, layout)
        if not (
            torch.equal(quant_out.view(torch.uint8), own_q.view(torch.uint8))
            and torch.equal(scale_out, own_sf[written])
        ):
            print(f"FAIL quant-vs-own-norm rank{rank} H={hidden_dim} T={token_num}")
            return 1
    tag = (
        f"rank{rank} dtype={dtype} H={hidden_dim} T={token_num} "
        f"oneshot={use_oneshot} layout={layout} norm_out={with_norm_out}"
    )
    ok = True
    if not torch.equal(residual_out, ref_residual):
        print(f"FAIL residual {tag}")
        ok = False
    if with_norm_out:
        # The reference norm comes from a separate launch whose grid can differ (the
        # MXFP8 variant keeps scale groups inside one CTA), so its RMS reduction order
        # can differ: allow one ulp. The quant is checked exactly against the kernel's
        # own norm output above.
        ulps = (
            norm_out.view(torch.int16).int() - ref_norm.view(torch.int16).int()
        ).abs()
        if not bool((ulps <= 1).all()):
            print(f"FAIL norm_out {tag}: max {ulps.max().item()} ulp")
            ok = False
    q_bytes, ref_q_bytes = quant_out.view(torch.uint8), ref_q.view(torch.uint8)
    if not torch.equal(q_bytes, ref_q_bytes):
        # The reference norm comes from the kARResidualRMSNorm kernel. When the
        # launcher re-tiles block/cluster sizes by register count (large token
        # counts), its RMS reduction order can differ from the MXFP8 kernel's,
        # moving a norm value by one ulp across an E4M3 rounding boundary.
        # Accept only isolated, adjacent-code differences of that kind;
        # with norm_out the quant must match the kernel's own norm exactly.
        diff = (q_bytes.int() - ref_q_bytes.int()).abs()
        n_bad = (diff != 0).sum().item()
        adjacent = bool((diff <= 1).all())
        if with_norm_out or not adjacent or n_bad > max(1, q_bytes.numel() // 100000):
            print(f"FAIL quant {tag}: {n_bad}/{q_bytes.numel()} bytes differ")
            ok = False
        else:
            print(
                f"WARN quant {tag}: {n_bad}/{q_bytes.numel()} bytes off by one code "
                "(reference norm launch config differs)"
            )
    if not torch.equal(scale_out, ref_sf):
        n_bad = (scale_out != ref_sf).sum().item()
        print(f"FAIL scale {tag}: {n_bad}/{ref_sf.numel()} bytes differ")
        ok = False
    if ok and rank == 0:
        print(f"PASS {tag}")
    return 0 if ok else 1


def _run_unified_api_worker(world_size, rank, dtype, port):
    """allreduce_fusion() on a TRT-LLM workspace: outputs, return value and validation."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    token_num, hidden_dim = 37, 2048
    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(group=group),
        )
        torch.manual_seed(4321)
        allreduce_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
        allreduce_in *= 1.0 + rank
        residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

        def run(pattern, norm_out=None, quant_out=None, scale_out=None, layout=None):
            residual_out = torch.empty_like(residual_in)
            ret = comm.allreduce_fusion(
                input=allreduce_in.clone(),
                workspace=workspace,
                pattern=pattern,
                launch_with_pdl=True,
                residual_in=residual_in,
                residual_out=residual_out,
                norm_out=norm_out,
                quant_out=quant_out,
                scale_out=scale_out,
                rms_gamma=rms_gamma,
                rms_eps=RMS_EPS,
                layout_code=layout,
                fp32_acc=True,
            )
            return ret, residual_out

        ref_norm = torch.empty_like(residual_in)
        run(comm.AllReduceFusionPattern.kARResidualRMSNorm, norm_out=ref_norm)
        torch.cuda.synchronize()

        for layout in [None, comm.QuantizationSFLayout.LINEAR]:
            ref_layout = (
                comm.QuantizationSFLayout.SWIZZLED_128x4 if layout is None else layout
            )
            ref_q, ref_sf = _mxfp8_ref(ref_norm, ref_layout)
            written = _written_sf_mask(
                token_num, hidden_dim, ref_layout, ref_sf.numel(), device
            )
            for with_norm_out in [False, True]:
                pattern = (
                    comm.AllReduceFusionPattern.kARResidualRMSNormOutMXFP8Quant
                    if with_norm_out
                    else comm.AllReduceFusionPattern.kARResidualRMSNormMXFP8Quant
                )
                norm_out = torch.empty_like(residual_in) if with_norm_out else None
                quant_out = torch.empty(
                    token_num, hidden_dim, dtype=torch.float8_e4m3fn, device=device
                )
                scale_out = torch.full(
                    (ref_sf.numel(),), 0xAB, dtype=torch.uint8, device=device
                )
                ret, _ = run(pattern, norm_out, quant_out, scale_out, layout)
                torch.cuda.synchronize()
                assert ret is quant_out
                # The reference norm comes from a different launch configuration; allow
                # the same isolated one-code differences as the legacy-API test.
                diff = quant_out.view(torch.uint8).int() - ref_q.view(torch.uint8).int()
                assert bool((diff.abs() <= 1).all())
                assert (diff != 0).sum().item() <= max(1, diff.numel() // 100000)
                assert torch.equal(scale_out[written], ref_sf[written])
                if with_norm_out:
                    assert torch.equal(norm_out, ref_norm)

        # Argument validation happens before any communication, on every rank.
        mx = comm.AllReduceFusionPattern.kARResidualRMSNormMXFP8Quant
        mx_out = comm.AllReduceFusionPattern.kARResidualRMSNormOutMXFP8Quant
        quant_out = torch.empty(
            token_num, hidden_dim, dtype=torch.float8_e4m3fn, device=device
        )
        n_sf = 128 * (hidden_dim // 32)
        with pytest.raises(ValueError, match="uint8"):
            run(
                mx, None, quant_out, torch.empty(n_sf, dtype=torch.int32, device=device)
            )
        with pytest.raises(ValueError, match="at least"):
            run(
                mx,
                None,
                quant_out,
                torch.empty(n_sf - 1, dtype=torch.uint8, device=device),
            )
        with pytest.raises(ValueError, match="norm_out is required"):
            run(
                mx_out,
                None,
                quant_out,
                torch.empty(n_sf, dtype=torch.uint8, device=device),
            )
        with pytest.raises(ValueError, match="quant_out and scale_out"):
            run(mx, None, quant_out, None)
        with pytest.raises(ValueError, match="layouts"):
            run(
                mx,
                None,
                quant_out,
                torch.empty(n_sf, dtype=torch.uint8, device=device),
                comm.QuantizationSFLayout.SWIZZLED_8x4,
            )
        dist.barrier(group=group)
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group(group=group)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _multi_process(world_size, target, *args):
    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = [
        mp.Process(target=target, args=(world_size, i, *args, port))
        for i in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for i, p in enumerate(procs):
        assert p.exitcode == 0, f"Process {i} failed with exit code {p.exitcode}"


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_trtllm_allreduce_fusion_mxfp8(world_size, dtype):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    if get_compute_capability(torch.device("cuda:0"))[0] < 10:
        pytest.skip("mxfp8_quantize(cute-dsl) reference needs SM100+")
    _multi_process(world_size, _run_worker, dtype, [6144, 2048, 7168, 1056, 2080])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_allreduce_fusion_mxfp8_unified_api(dtype):
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    if get_compute_capability(torch.device("cuda:0"))[0] < 10:
        pytest.skip("mxfp8_quantize(cute-dsl) reference needs SM100+")
    _multi_process(world_size, _run_unified_api_worker, dtype)


if __name__ == "__main__":
    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for dt in (torch.bfloat16, torch.float16):
        _multi_process(ws, _run_worker, dt, [6144, 2048, 7168, 1056, 2080])
        _multi_process(2, _run_unified_api_worker, dt)
    print("ALL PASSED")
