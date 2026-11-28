"""Compare prequantized CuTe/Cake outputs with an FP32 dequantized reference.

Run once on the old checkout with --label before, then on the new checkout with
--label after --baseline before.pt. Input hashes verify identical aligned cases.
Known unsupported/unsafe old tail paths are never launched.
"""

import argparse
import hashlib
import json
import math

import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd


def make_case(sq, sk, topk, kind, seed):
    torch.manual_seed(seed)
    q, k, v = [
        torch.randn(1, s, 4, 128, device="cuda").bfloat16().float()
        for s in (sq, sk, sk)
    ]
    qs = q.abs().amax(-1).clamp_min(1e-6) / 448
    kp = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, (-sk) % 16))
    ks = kp.reshape(1, -1, 16, 4, 128).abs().amax((2, 4)).clamp_min(1e-6) / 448
    vs = v.abs().amax((0, 1)).clamp_min(1e-6) / 448
    q = (q / qs[..., None]).to(torch.float8_e4m3fn)
    k = (k / ks.repeat_interleave(16, 1)[:, :sk, :, None]).to(torch.float8_e4m3fn)
    v = (v / vs).to(torch.float8_e4m3fn)
    nq, nk = (sq + 63) // 64, (sk + 63) // 64
    index = torch.rand(1, 4, nq, nk, device="cuda").argsort(-1)[..., :topk]
    index = index.to(torch.int32).contiguous()
    kwargs = dict(
        q_scale=qs.transpose(1, 2).contiguous(),
        k_scale=ks.transpose(1, 2).contiguous(),
        v_scale=vs,
    )
    if kind == "variable":
        counts = torch.arange(4 * nq, device="cuda", dtype=torch.int32)
        kwargs["q2k_block_nums"] = (counts % (topk + 1)).reshape(1, 4, nq)
    if kind in ("variable", "padding"):
        sizes = torch.full((nk,), 64, device="cuda", dtype=torch.int32)
        sizes[0] = 17
        kwargs["block_sizes"] = sizes
    mask = torch.zeros(1, 4, sq, sk, device="cuda", dtype=torch.bool)
    idx = index.cpu()
    counts = kwargs.get("q2k_block_nums")
    counts = counts.cpu() if counts is not None else None
    for h in range(4):
        for row in range(nq):
            count = int(counts[0, h, row]) if counts is not None else topk
            for block in idx[0, h, row, :count].tolist():
                size = 17 if kind in ("variable", "padding") and block == 0 else 64
                mask[
                    0,
                    h,
                    row * 64 : min((row + 1) * 64, sq),
                    block * 64 : min(block * 64 + size, sk),
                ] = True
    qd = q.float() * qs[..., None]
    kd = k.float() * ks.repeat_interleave(16, 1)[:, :sk, :, None]
    vd = v.float() * vs
    scores = qd.transpose(1, 2) @ kd.transpose(1, 2).transpose(-1, -2)
    scores = (scores / math.sqrt(128)).masked_fill(~mask, -torch.inf)
    expected_lse = torch.logsumexp(scores, -1)
    expected = (scores.softmax(-1).nan_to_num() @ vd.transpose(1, 2)).transpose(1, 2)
    digest = hashlib.sha256()
    for t in (q, k, v, index, *kwargs.values()):
        digest.update(
            t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        )
    return q, k, v, index, kwargs, expected, expected_lse, digest.hexdigest()


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    diff = a - b
    return dict(
        max_abs=diff.abs().max().item(),
        rmse=diff.square().mean().sqrt().item(),
        relative_l2=(diff.norm() / b.norm().clamp_min(1e-30)).item(),
        cosine=torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
    )


def reproduce_tail(label):
    """Expose #5090's logical padding error using fully allocated backing storage."""
    from flashinfer.sparse import BlockSparseAttentionWrapper

    q = torch.zeros(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    k_backing = torch.zeros_like(q)
    v_backing = torch.zeros_like(q)
    v_backing[:, :65] = 1
    k, v = k_backing[:, :65], v_backing[:, :65]
    index = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    index = index.expand(1, 4, 2, 2).contiguous()
    actual, _ = bsa_attn_sm100_blk64_fwd(
        q,
        k,
        v,
        index,
        2,
        backend="cute",
        use_clc=False,
    )
    # With Q=K=0, attention must average the 65 valid V=1 rows to exactly 1.
    row = dict(
        label=label,
        sq=128,
        sk=65,
        expected_value=1.0,
        actual_min=actual.min().item(),
        actual_max=actual.max().item(),
        reference_error=errors(actual, torch.ones_like(actual)),
    )
    workspace = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm100_blk64")
    try:
        wrapper.plan(
            torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
            torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32),
            128,
            65,
            64,
            64,
            4,
            4,
            128,
            q_data_type=torch.bfloat16,
            use_clc=False,
        )
        row["wrapper_plan"] = "accepted"
        output = wrapper.run(q[0], k[0], v[0])
        row["wrapper_max_abs"] = (output.float() - 1).abs().max().item()
    except ValueError as exc:
        row["wrapper_plan"] = str(exc)
    if label == "after":
        assert row["reference_error"]["max_abs"] == 0
        assert row["wrapper_plan"] == "accepted" and row["wrapper_max_abs"] == 0
    print(json.dumps(row), flush=True)
    return row, actual.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("before", "after"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--reproduce-tail", action="store_true")
    args = parser.parse_args()
    if args.reproduce_tail:
        row, output = reproduce_tail(args.label)
        with open(args.output, "w") as f:
            json.dump(row, f, indent=2)
        torch.save(output, args.tensor_output)
        raise SystemExit(0)
    torch.set_float32_matmul_precision("highest")
    baseline = torch.load(args.baseline, weights_only=True) if args.baseline else {}
    snapshots, rows = {}, []
    cases = [
        ("aligned_dense", 128, 256, 4, "aligned", 1),
        ("aligned_sparse", 512, 1024, 8, "aligned", 1),
        ("multiwave", 4160, 1024, 16, "aligned", 1),
        ("short_q_auto", 64, 16384, 256, "aligned", "auto"),
        ("kv_tail", 65, 127, 2, "tail", 1),
        ("short_q_tail", 1, 65, 2, "tail", 1),
        ("padding", 128, 128, 2, "padding", 1),
        ("variable_tail", 193, 193, 4, "variable", 4),
    ]
    for i, (name, sq, sk, topk, kind, splits) in enumerate(cases):
        if args.label == "before" and kind != "aligned":
            rows.append(dict(case=name, status="not_run_old_tail_or_metadata_path"))
            continue
        q, k, v, index, kwargs, expected, expected_lse, digest = make_case(
            sq, sk, topk, kind, 42 + i
        )
        for backend in ["cute"] if args.label == "before" else ["cute", "cake"]:
            actual, lse = bsa_attn_sm100_blk64_fwd(
                q,
                k,
                v,
                index,
                topk,
                **kwargs,
                backend=backend,
                kv_splits=splits,
                return_lse=True,
            )
            torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.04)
            torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)
            empty = torch.isneginf(expected_lse)
            assert (actual.transpose(1, 2)[empty] == 0).all()
            finite = torch.isfinite(expected_lse)
            row = dict(
                case=name,
                backend=backend,
                status="pass",
                input_sha256=digest,
                reference_error=errors(actual, expected),
                lse_max_abs=(lse[finite] - expected_lse[finite]).abs().max().item(),
            )
            snapshot = dict(output=actual.cpu(), lse=lse.cpu(), input_sha256=digest)
            if name in baseline:
                old = baseline[name]["cute"]
                assert old["input_sha256"] == digest
                row["old_cute_error"] = errors(actual.cpu(), old["output"])
                row["old_cute_bitwise_equal"] = torch.equal(
                    snapshot["output"], old["output"]
                )
            if backend == "cake":
                row["new_cute_error"] = errors(
                    actual.cpu(), snapshots[name]["cute"]["output"]
                )
            snapshots.setdefault(name, {})[backend] = snapshot
            rows.append(row)
            print(json.dumps(row), flush=True)
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        torch.save(snapshots, args.tensor_output)
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)
