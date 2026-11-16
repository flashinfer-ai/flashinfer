"""Opt-in, small GPU/MPS check for flashinfer-ai/flashinfer#9484.

From the checkout root, in an ALREADY configured MPS environment, run separate
processes (the client percentage is fixed before CUDA initialization):

    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 python -m \
        scripts.repro_gdn_tensormap_workspace --run --save /tmp/gdn-full.pt
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python -m \
        scripts.repro_gdn_tensormap_workspace --run --compare /tmp/gdn-full.pt

Requires torch, CuPy and FlashInfer's CUDA/CuTe dependencies. Does not start MPS
or provision resources. Use --q-heads 16 --v-heads 48 for the issue's head counts.
Run with compute-sanitizer for additional memory-access evidence. A probe that
misses high SM IDs does NOT rule out the precondition. CPU tests do not validate
this script or establish GPU compilation/concurrency correctness.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="explicitly opt into GPU work"
    )
    parser.add_argument("--save", help="save CPU outputs for a separate MPS client")
    parser.add_argument("--compare", help="compare with outputs from --save")
    parser.add_argument("--q-heads", type=int, default=2)
    parser.add_argument("--v-heads", type=int, default=4)
    args = parser.parse_args()
    if not args.run:
        parser.error("GPU work requires --run; --help does not import CUDA packages")
    if args.q_heads <= 0 or args.v_heads <= 0 or args.v_heads % args.q_heads:
        parser.error("v-heads must be a positive multiple of positive q-heads")

    import cupy as cp
    import torch
    import flashinfer
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from tests.gdn.reference_delta_rule import blockwise_delta_rule, exclusive_cumsum

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9 and (major, minor) not in ((12, 0), (12, 1)):
        raise RuntimeError("This check targets the SM90 and SM12x DSL wrappers only")

    probe = cp.RawKernel(
        r"""
        extern "C" __global__ void probe(unsigned int* ids, unsigned int* bounds) {
            unsigned int sm, ns;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
            asm volatile("mov.u32 %0, %%nsmid;" : "=r"(ns));
            unsigned long long start = clock64();
            while (clock64() - start < 4096) {}
            ids[blockIdx.x] = sm;
            bounds[blockIdx.x] = ns;
        }
        """,
        "probe",
    )
    with cp.cuda.Device(device.index):
        ids = cp.empty(4096, dtype=cp.uint32)
        bounds = cp.empty_like(ids)
        probe((4096,), (1,), (ids, bounds))
        observed = cp.unique(ids).get().tolist()
        nsmid = cp.unique(bounds).get().tolist()
    visible = torch.cuda.get_device_properties(device).multi_processor_count
    metadata = dict(
        device=torch.cuda.get_device_name(device),
        capability=[major, minor],
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        flashinfer=flashinfer.__version__,
        flashinfer_source=flashinfer.__file__,
        mps_percentage=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset"),
        visible_sms=visible,
        observed_smid=observed,
        nsmid=nsmid,
        observed_original_oob_precondition=max(observed) >= visible,
    )
    print(json.dumps(metadata, indent=2), flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    for lengths in ([2] * 32, [65, 2, 128, 17]):
        torch.manual_seed(9484)
        total = sum(lengths)
        q = torch.randn(total, args.q_heads, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(total, args.v_heads, 128, device=device, dtype=torch.bfloat16)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        g = 0.9 + 0.1 * torch.rand(total, args.v_heads, device=device)
        beta = torch.rand_like(g)
        cu = torch.tensor(exclusive_cumsum(lengths), device=device, dtype=torch.int64)
        initial_state = 0.1 * torch.randn(
            len(lengths), args.v_heads, 128, 128, device=device
        )
        cases.append(
            dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                cu_seqlens=cu,
                initial_state=initial_state,
            )
        )

    def run(inputs, use_cp):
        lengths = [2] * 32 if inputs is cases[0] else [65, 2, 128, 17]
        return chunk_gated_delta_rule(
            **inputs,
            scale=128**-0.5,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_cp=use_cp,
            max_seqlen=max(lengths),
            _cp_chunk_len=64 if use_cp else None,
        )

    def check(actual, expected, *, exact=False):
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                a, b, atol=0 if exact else 1e-2, rtol=0 if exact else 1e-2
            )

    outputs = {}
    with torch.inference_mode():
        for i, inputs in enumerate(cases):
            lengths = [2] * 32 if i == 0 else [65, 2, 128, 17]
            ref_o, ref_s = blockwise_delta_rule(
                inputs["q"].float(),
                inputs["k"].float(),
                inputs["v"].float(),
                lengths,
                scale_factor=128**-0.5,
                alpha=inputs["g"],
                beta=inputs["beta"],
                state_dtype=torch.float32,
                initial_state=inputs["initial_state"].transpose(-1, -2),
            )
            for use_cp in (False, True):
                result = run(inputs, use_cp)
                check(result, (ref_o.to(inputs["q"].dtype), ref_s.transpose(-1, -2)))
                outputs[f"case{i}-cp{use_cp}"] = [t.cpu() for t in result]

                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    run(inputs, use_cp)
                stream.synchronize()
                original_g = inputs["g"].clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run(inputs, use_cp)
                graph.replay()
                torch.cuda.synchronize(device)
                check(captured, result, exact=True)
                inputs["g"].mul_(0.99)
                graph.replay()
                torch.cuda.synchronize(device)
                check(captured, run(inputs, use_cp), exact=True)
                inputs["g"].copy_(original_g)
                torch.cuda.synchronize(device)

    if args.compare:
        saved = torch.load(args.compare, map_location="cpu", weights_only=True)
        assert outputs.keys() == saved["outputs"].keys()
        for name, result in outputs.items():
            check(result, saved["outputs"][name])
    if args.save:
        torch.save(dict(metadata=metadata, outputs=outputs), args.save)
    print("PASS: reference, eager, changed-input graphs")


if __name__ == "__main__":
    main()
