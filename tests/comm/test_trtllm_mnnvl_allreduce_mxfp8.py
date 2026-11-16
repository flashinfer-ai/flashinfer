"""MXFP8 output of the MNNVL allreduce + residual + RMSNorm fusion.

The fused kernel must produce what the unfused sequence produces: the MNNVL
kARResidualRMSNorm pattern followed by mxfp8_quantize (backend="cute-dsl") on
the norm output -- same E4M3 bytes and UE8M0 scale bytes (swizzled 128x4 column
padding included), same residual and norm outputs -- for the one-shot kernel
and the two-shot RMSNorm kernel.

Runs single-node with one spawned process per GPU (no MPI launcher needed):
python tests/comm/test_trtllm_mnnvl_allreduce_mxfp8.py [world_size]
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

MAX_TOKEN_NUM = 300
TOKEN_NUMS = [1, 4, 37, 128, 300]
HIDDEN_DIMS = [1056, 2048, 3072, 7168]


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


def _check_case(rank, workspace, dtype, hidden_dim, token_num, use_oneshot, group):
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234 + 31 * token_num + hidden_dim)
    # Different per-rank inputs; same residual / gamma on every rank.
    allreduce_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
    allreduce_in *= 1.0 + rank
    residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device="cpu").to(
        device
    )
    rms_gamma = torch.randn(hidden_dim, dtype=dtype, device="cpu").to(device)
    eps = 1e-6

    def run(pattern, norm_out=None, quant_out=None, scale_out=None, layout=None):
        residual_out = torch.empty_like(residual_in)
        dist.barrier(group=group)
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
            rms_eps=eps,
            layout_code=layout,
            use_oneshot=use_oneshot,
            weight_bias=1.0,
        )
        torch.cuda.synchronize()
        return ret, residual_out

    ref_norm = torch.empty_like(residual_in)
    _, ref_residual = run(comm.AllReduceFusionPattern.kARResidualRMSNorm, ref_norm)

    failures = 0
    for layout in [
        comm.QuantizationSFLayout.SWIZZLED_128x4,
        comm.QuantizationSFLayout.LINEAR,
    ]:
        ref_q, ref_sf = _mxfp8_ref(ref_norm, layout)
        written = _written_sf_mask(
            token_num, hidden_dim, layout, ref_sf.numel(), device
        )
        for with_norm_out in [False, True]:
            pattern = (
                comm.AllReduceFusionPattern.kARResidualRMSNormOutMXFP8Quant
                if with_norm_out
                else comm.AllReduceFusionPattern.kARResidualRMSNormMXFP8Quant
            )
            tag = (
                f"rank{rank} dtype={dtype} H={hidden_dim} T={token_num} "
                f"oneshot={use_oneshot} layout={layout} norm_out={with_norm_out}"
            )
            norm_out = torch.empty_like(residual_in) if with_norm_out else None
            quant_out = torch.empty(
                token_num, hidden_dim, dtype=torch.float8_e4m3fn, device=device
            )
            # Poison the scale buffer so unwritten scales (padding columns included) are
            # detected; padding rows are not written and are excluded from the comparison.
            scale_out = torch.full(
                (ref_sf.numel(),), 0xAB, dtype=torch.uint8, device=device
            )
            ret, residual_out = run(pattern, norm_out, quant_out, scale_out, layout)
            ok = ret is quant_out
            if not ok:
                print(f"FAIL return value {tag}")
            if not torch.equal(residual_out, ref_residual):
                print(f"FAIL residual {tag}")
                ok = False
            q_bytes = quant_out.view(torch.uint8)
            if with_norm_out:
                # The reference norm comes from a separate launch whose grid can differ
                # (the MXFP8 variant keeps scale groups inside one CTA), so its RMS
                # reduction order can differ: allow one ulp. The quant is then checked
                # exactly against the kernel's own norm output.
                ulps = (
                    norm_out.view(torch.int16).int() - ref_norm.view(torch.int16).int()
                ).abs()
                if not bool((ulps <= 1).all()):
                    print(f"FAIL norm_out {tag}: max {ulps.max().item()} ulp")
                    ok = False
                own_q, own_sf = _mxfp8_ref(norm_out, layout)
                if not (
                    torch.equal(q_bytes, own_q.view(torch.uint8))
                    and torch.equal(scale_out[written], own_sf[written])
                ):
                    print(f"FAIL quant-vs-own-norm {tag}")
                    ok = False
            else:
                # The reference norm comes from a separate launch whose grid can differ
                # (the MXFP8 variant keeps scale groups inside one CTA), which can move a
                # norm value by one ulp across an E4M3 rounding boundary. Accept only
                # isolated, adjacent-code differences of that kind.
                diff = (q_bytes.int() - ref_q.view(torch.uint8).int()).abs()
                n_bad = (diff != 0).sum().item()
                if not bool((diff <= 1).all()) or n_bad > max(
                    1, diff.numel() // 100000
                ):
                    print(f"FAIL quant {tag}: {n_bad}/{diff.numel()} bytes differ")
                    ok = False
                if not torch.equal(scale_out[written], ref_sf[written]):
                    n_bad = (scale_out[written] != ref_sf[written]).sum().item()
                    print(f"FAIL scale {tag}: {n_bad} bytes differ")
                    ok = False
            if ok and rank == 0:
                print(f"PASS {tag}")
            failures += 0 if ok else 1
    return failures


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
            workspace = comm.create_allreduce_fusion_workspace(
                backend="mnnvl",
                world_size=world_size,
                rank=rank,
                max_token_num=MAX_TOKEN_NUM,
                hidden_dim=hidden_dim,
                dtype=dtype,
                gpus_per_node=torch.cuda.device_count(),
                comm_backend=TorchDistBackend(group=group),
                force_oneshot_support=True,
            )
            try:
                for token_num in TOKEN_NUMS:
                    for use_oneshot in [True, False]:
                        failures += _check_case(
                            rank,
                            workspace,
                            dtype,
                            hidden_dim,
                            token_num,
                            use_oneshot,
                            group,
                        )
                # Argument validation runs before any communication, on every rank.
                quant_out = torch.empty(
                    4, hidden_dim, dtype=torch.float8_e4m3fn, device=device
                )
                x = torch.randn(4, hidden_dim, dtype=dtype, device=device)
                for kwargs, match in [
                    (
                        dict(
                            scale_out=torch.empty(
                                4096, dtype=torch.int32, device=device
                            )
                        ),
                        "uint8",
                    ),
                    (
                        dict(
                            scale_out=torch.empty(1, dtype=torch.uint8, device=device)
                        ),
                        "at least",
                    ),
                    (dict(scale_out=None), "quant_out and scale_out"),
                ]:
                    with pytest.raises(ValueError, match=match):
                        comm.allreduce_fusion(
                            input=x,
                            workspace=workspace,
                            pattern=comm.AllReduceFusionPattern.kARResidualRMSNormMXFP8Quant,
                            launch_with_pdl=True,
                            residual_in=x,
                            residual_out=torch.empty_like(x),
                            quant_out=quant_out,
                            rms_gamma=x[0],
                            **kwargs,
                        )
                dist.barrier(group=group)
            finally:
                workspace.destroy()
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group(group=group)
    assert failures == 0, f"rank {rank}: {failures} failing cases"


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _multi_process(world_size, dtype, hidden_dims):
    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = [
        mp.Process(target=_run_worker, args=(world_size, i, dtype, hidden_dims, port))
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
def test_mnnvl_allreduce_fusion_mxfp8(world_size, dtype):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    if get_compute_capability(torch.device("cuda:0"))[0] < 10:
        pytest.skip("mxfp8_quantize(cute-dsl) reference needs SM100+")
    _multi_process(world_size, dtype, HIDDEN_DIMS)


if __name__ == "__main__":
    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for dt in (torch.bfloat16, torch.float16):
        _multi_process(ws, dt, HIDDEN_DIMS)
    print("ALL PASSED")
