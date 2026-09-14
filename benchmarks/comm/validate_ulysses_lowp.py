"""User-run collective and Sage acceptance gate on either SM90 or SM120.

torchrun --standalone --nproc-per-node=4 benchmarks/comm/validate_ulysses_lowp.py --output result.json
Requires this source checkout's tests and pytest; does not modify Sage/SGLang.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

import flashinfer.comm.ulysses_lowp as lowp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "comm"))
from test_ulysses_lowp_boundary import check_receiver, inputs  # noqa: E402


def metrics(output, reference):
    a, b = output.float(), reference.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("non-finite attention output/reference")
    delta = a - b
    return dict(
        cos=torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
        rel_l1=(delta.abs().mean() / b.abs().mean().clamp_min(1e-12)).item(),
        rel_l2=(delta.norm() / b.norm().clamp_min(1e-12)).item(),
        max_abs=delta.abs().max().item(),
    )


def mean_metrics(value, reference):
    def ordered_bits(x):
        bits = x.float().contiguous().view(torch.int32).to(torch.int64)
        return torch.where(
            bits < 0, 0x80000000 - (bits & 0x7FFFFFFF), 0x80000000 + bits
        )

    return dict(
        max_abs=(value.float() - reference.float()).abs().max().item(),
        max_ulp=(ordered_bits(value) - ordered_bits(reference)).abs().max().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--local-sequence", type=int, default=129)
    parser.add_argument("--used", type=int)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--zero-v", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if Path(lowp.__file__).resolve() != repo / "flashinfer/comm/ulysses_lowp.py":
        raise RuntimeError(f"FlashInfer import is not this checkout: {lowp.__file__}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        cap = lowp.capability("cuda")
        if cap["device_capability"] not in ((9, 0), (12, 0)) or not cap["supported"]:
            raise RuntimeError(cap)
        layout = getattr(lowp, cap["layout_class"])(head_dim=args.head_dim)
        length, total = args.local_sequence, args.local_sequence * world
        used = total if args.used is None else args.used
        if world not in (2, 4, 8) or not 0 < used <= total or args.heads % world:
            raise ValueError("requires P=2/4/8, 0<U<=P*L, and H divisible by P")
        q, k, v = inputs(
            args.batch,
            total,
            args.heads,
            getattr(torch, args.dtype),
            used,
            head_dim=args.head_dim,
        )
        if args.zero_v:
            v.zero_()
        shard = tuple(x[:, rank * length : (rank + 1) * length] for x in (q, k, v))
        send, ctx = layout.local_stats(
            *shard, rank=rank, world_size=world, used_sequence=used, enable_pdl=False
        )
        assert ctx.stats_protocol == lowp.BOUNDARY_MERGE
        gathered = torch.empty(
            world * send.numel(), device=send.device, dtype=send.dtype
        )
        dist.all_gather_into_tensor(gathered, send)
        stats = layout.finalize_stats(gathered, ctx, shard[1], enable_pdl=False)
        payload = layout.quant_and_pack(*shard, stats, enable_pdl=False)
        recv = torch.empty_like(payload)
        dist.all_to_all_single(recv, payload)
        quant_report = {}
        qi, ki, vp, qs, ks = check_receiver(
            layout, q, k, v, world, used, rank, stats, recv, report=quant_report
        )
        # Check actual exchanged bytes against each sender's destination chunk.
        all_payloads = [torch.empty_like(payload) for _ in range(world)]
        dist.all_gather(all_payloads, payload)
        assert torch.equal(recv, torch.stack([p[rank] for p in all_payloads]))
        assert lowp.verify_duplicate_scale_slots(
            recv,
            batch_size=args.batch,
            local_sequence=length,
            local_heads=args.heads // world,
            head_dim=args.head_dim,
            world_size=world,
            q_group=layout.Q_GROUP,
            k_group=layout.K_GROUP,
            spec=layout.payload_spec(
                batch_size=args.batch,
                local_sequence=length,
                num_heads=args.heads,
                world_size=world,
            ),
        )
        heads = slice(rank * (args.heads // world), (rank + 1) * (args.heads // world))
        q0, k0, v0 = (x[:, :used, heads].contiguous() for x in (q, k, v))
        with sdpa_kernel(SDPBackend.MATH):
            ref = torch.nn.functional.scaled_dot_product_attention(
                q0.float().transpose(1, 2),
                k0.float().transpose(1, 2),
                v0.float().transpose(1, 2),
                scale=args.head_dim**-0.5,
            ).transpose(1, 2)
        from sageattention import core

        if layout.Q_GROUP == 16:
            from sageattention import _qattn_sm90

            attn = _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
            extension_file = _qattn_sm90.__file__
            stock = core.sageattn_qk_int8_pv_fp8_cuda_sm90
            accum = "fp32+fp32"
        else:
            from sageattention import _qattn_sm89

            attn = _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
            extension_file = _qattn_sm89.__file__
            stock = core.sageattn_qk_int8_pv_fp8_cuda
            accum = "fp32+fp16"
        vs = stats.v_scale_global[:, heads].contiguous()
        sentinel = torch.full(
            (args.batch, total + 1, args.heads // world, args.head_dim),
            17,
            device=q.device,
            dtype=q.dtype,
        )

        def run(qscale):
            attn(
                qi[:, :used],
                ki[:, :used],
                vp,
                sentinel[:, :used],
                qscale,
                ks,
                vs,
                0,
                0,
                2,
                args.head_dim**-0.5,
                0,
            )
            assert torch.all(sentinel[:, used:] == 17)
            return sentinel[:, :used].clone()

        output = run(qs)
        for value in (0.0, 1.0, 2.0):
            alternate = qs.clone()
            alternate[..., (used + layout.Q_GROUP - 1) // layout.Q_GROUP :] = value
            assert torch.equal(output, run(alternate)), (
                "extra Q scales affect live output"
            )
        low_metrics = metrics(output, ref)
        if args.zero_v:
            assert torch.count_nonzero(output) == 0
            stock_metrics = None  # stock's zero-amax behavior is not our oracle
        else:
            sage = stock(
                q0,
                k0,
                v0,
                tensor_layout="NHD",
                is_causal=False,
                qk_quant_gran="per_warp",
                sm_scale=args.head_dim**-0.5,
                pv_accum_dtype=accum,
                smooth_k=True,
            )
            # Stock can emit NaNs in the two synthetic zero channels. Exclude
            # these from its comparison; our complete output must be finite.
            stock_metrics = metrics(sage[..., 2:], ref[..., 2:])
            low_metrics_nonzero = metrics(output[..., 2:], ref[..., 2:])
            assert low_metrics_nonzero["cos"] >= 0.999
            assert low_metrics_nonzero["rel_l1"] <= 1.10 * stock_metrics["rel_l1"]
        reference_mean32 = k[:, :used].float().mean(1)
        distributed_mean32 = (
            gathered.view(world, -1)[:, : ctx.stats_numel]
            .sum(0)
            .view(args.batch, args.heads, args.head_dim)
            / used
        )
        row = dict(
            rank=rank,
            lowp=low_metrics,
            stock=stock_metrics,
            mean_before_rounding=mean_metrics(distributed_mean32, reference_mean32),
            mean_after_rounding=mean_metrics(
                stats.k_mean_global, reference_mean32.to(k.dtype)
            ),
            quantization=quant_report,
            stats_bytes=send.numel() * 4,
            payload_bytes=payload.numel(),
            payload_dtype=str(payload.dtype),
            protocol=ctx.stats_protocol,
        )
        rows = [None] * world
        dist.all_gather_object(rows, row)
        if rank == 0:
            report = dict(
                args=vars(args),
                gpu=torch.cuda.get_device_name(),
                cap=cap,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                scale_max=2.25,
                flashinfer_file=lowp.__file__,
                sage_file=core.__file__,
                sage_extension_file=extension_file,
                rows=rows,
                sha=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                diff=subprocess.check_output(
                    ["git", "diff", "--binary"], cwd=repo, text=True
                ),
            )
            with open(args.output, "x") as f:
                json.dump(report, f, indent=2)
            print(f"PASS: {args.output}")
    except Exception as error:
        failure = Path(args.output).with_suffix(f".rank{rank}.failure.json")
        with failure.open("x") as f:
            json.dump(dict(rank=rank, args=vars(args), error=str(error)), f, indent=2)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
