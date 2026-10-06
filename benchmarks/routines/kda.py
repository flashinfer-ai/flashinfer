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
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_compute_capability,
    get_device,
    is_close_stats,
    print_perf_metrics,
)

HEAD_DIM = 128

#: Backends this routine can time. All of them run the same shapes, the same
#: timing method, the same stream and the same pre-allocated output and
#: final-state buffers -- which is the only way the numbers can be divided by
#: one another afterwards.
#:
#: ``flashinfer`` is the public entry point with its own variant policy;
#: ``flashinfer-decomp`` and ``flashinfer-fused`` pin one variant each, so a
#: report can show what the policy chose *and* what it declined.
#:
#: ``cutekda`` and ``flash-kda`` are external and optional. They are imported
#: only when asked for and only if already installed, they are never a
#: dependency of this file, and a missing one is an explicit skip rather than a
#: silent omission -- a comparison table with a quietly absent baseline is
#: worse than one with a gap in it.
_FLASHINFER_BACKENDS = ("flashinfer", "flashinfer-decomp", "flashinfer-fused")
_EXTERNAL_BACKENDS = ("cutekda", "flash-kda")

#: ``fused_kda_decode`` backends, passed straight through as its ``backend``
#: argument. ``cake`` and ``auto`` are given ``state_indices_mode=
#: "positive_unique"``, which the benchmark's slot assignment satisfies.
_DECODE_BACKENDS = ("cute-dsl", "cake", "auto")
_DECODE_NUM_HEADS = (8, 12, 24, 32, 48, 96)
_DECODE_CONV_HISTORY = 3


def run_kda_test(args):
    """Route a KDA benchmark case to its routine."""
    if args.routine == "recurrent_kda_prefill":
        return testRecurrentKDAPrefill(args)
    if args.routine == "fused_kda_decode":
        return testFusedKDADecode(args)
    raise ValueError(f"Unsupported routine: {args.routine}")


def parse_kda_args(line, parser):
    """Parse the KDA-specific arguments.

    Args:
        line: Command line arguments
        parser: ArgumentParser already populated with the shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Number of sequences. fused_kda_decode: number of decode rows "
        "(one token each).",
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        default=None,
        help="Tokens per sequence (required for recurrent_kda_prefill; "
        "fused_kda_decode is single-token). With --packed the sequences are "
        "packed into one [1, B * T, H, 128] activation and given explicit "
        "offsets; without it they are a fixed [B, T, H, 128] batch.",
    )
    parser.add_argument(
        "--num_q_heads",
        type=int,
        default=16,
        help="Number of heads. KDA is equal-head: H == HV.",
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=HEAD_DIM,
        help="Head dimension. Fixed at 128 for this backend.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16"],
        help="Data type for q/k/v/g/beta. The first published version is "
        "bfloat16 only.",
    )
    parser.add_argument(
        "--state_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Recurrent state dtype. recurrent_kda_prefill supports bfloat16 "
        "only; fused_kda_decode supports both.",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Use packed varlen input with explicit cu_seqlens.",
    )
    parser.add_argument(
        "--offsets_dtype",
        type=str,
        default="int32",
        choices=["int32", "int64"],
        help="cu_seqlens dtype for packed input. Both are accepted by the "
        "public API; the backend consumes a canonical int32 copy either way.",
    )
    parser.add_argument(
        "--has_initial_state",
        action="store_true",
        help="Pass a bfloat16 initial state, updated in place.",
    )
    parser.add_argument(
        "--lower_bound",
        type=float,
        default=-5.0,
        help="Gate lower bound. Must be in [-5.0, 0.0).",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=list(_FLASHINFER_BACKENDS)
        + list(_EXTERNAL_BACKENDS)
        + list(_DECODE_BACKENDS),
        help="Backends to benchmark. recurrent_kda_prefill (default "
        "'flashinfer'): 'flashinfer' is the public entry point with its own "
        "variant policy; '-decomp'/'-fused' pin one variant each; 'cutekda' "
        "and 'flash-kda' are external baselines, benchmarked only when already "
        "installed and never a dependency of this file. fused_kda_decode "
        "(default 'cute-dsl'): its backend argument, 'cute-dsl', 'cake' or "
        "'auto'.",
    )
    args = parser.parse_args(line)

    if args.routine == "fused_kda_decode":
        if args.backends is None:
            args.backends = ["cute-dsl"]
        if any(backend not in _DECODE_BACKENDS for backend in args.backends):
            raise ValueError(
                f"fused_kda_decode backends are {list(_DECODE_BACKENDS)}, "
                f"got {args.backends}"
            )
        if args.s_qo not in (None, 1):
            raise ValueError("fused_kda_decode is single-token: --s_qo must be 1")
        args.s_qo = 1
        if args.num_q_heads not in _DECODE_NUM_HEADS:
            raise ValueError(
                f"fused_kda_decode supports --num_q_heads in {_DECODE_NUM_HEADS}"
            )
        if args.packed or args.has_initial_state:
            raise ValueError(
                "--packed / --has_initial_state apply to recurrent_kda_prefill only"
            )
    else:
        if args.backends is None:
            args.backends = ["flashinfer"]
        if any(backend in _DECODE_BACKENDS for backend in args.backends):
            raise ValueError(
                f"{list(_DECODE_BACKENDS)} are fused_kda_decode backends; "
                f"{args.routine} supports "
                f"{list(_FLASHINFER_BACKENDS) + list(_EXTERNAL_BACKENDS)}"
            )
        if args.s_qo is None:
            raise ValueError(f"{args.routine} requires --s_qo")
        if args.state_dtype != "bfloat16":
            raise ValueError(f"{args.routine} supports --state_dtype bfloat16 only")
    return args


def kda_prefill_flops(total_tokens: int, num_heads: int, head_size: int) -> float:
    """Matmul FLOPs of one KDA prefill.

    Three [K, V] rank-updates and projections per token per head, each
    ``2 * K * V``: the state read for the prediction, the state update, and the
    output projection. The gate and the normalization are elementwise and do
    not move this number.
    """
    return 3.0 * 2.0 * total_tokens * num_heads * head_size * head_size


def kda_prefill_bytes(
    total_tokens: int,
    num_seqs: int,
    num_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    has_initial_state: bool,
    output_final_state: bool,
) -> float:
    """Compulsory DRAM traffic: activations in, output out, state either way.

    Compulsory, not achieved: the chunk factors the decomposed variant writes
    and reads back are not here, because whether they reach DRAM at all is a
    property of the schedule rather than of the problem, and counting them
    would make the two variants incomparable on a bytes-per-second column.
    """
    element = torch.finfo(input_dtype).bits // 8
    # q, k, v, g in; out out.
    activations = 5.0 * total_tokens * num_heads * head_size * element
    # beta in.
    activations += total_tokens * num_heads * element
    state_slabs = (1 if has_initial_state else 0) + (1 if output_final_state else 0)
    state = state_slabs * num_seqs * num_heads * head_size * head_size * element
    return activations + state


def _make_inputs(args, device):
    """Inputs matching the SM120 KDA prefill contract."""
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    num_seqs = args.batch_size
    seq_len = args.s_qo
    num_heads = args.num_q_heads
    head_size = args.head_size

    if args.packed:
        batch, tokens = 1, num_seqs * seq_len
        offsets_dtype = torch.int32 if args.offsets_dtype == "int32" else torch.int64
        cu_seqlens = torch.arange(
            0, tokens + 1, seq_len, dtype=offsets_dtype, device=device
        )
    else:
        batch, tokens = num_seqs, seq_len
        cu_seqlens = None

    shape = (batch, tokens, num_heads, head_size)

    def normalized():
        raw = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.nn.functional.normalize(raw, p=2.0, dim=-1).to(input_dtype)

    inputs = {
        "q": normalized(),
        "k": normalized(),
        "v": torch.randn(shape, dtype=torch.float32, device=device).to(input_dtype),
        "g": (0.1 * torch.randn(shape, dtype=torch.float32, device=device)).to(
            input_dtype
        ),
        "beta": torch.randn(
            (batch, tokens, num_heads), dtype=torch.float32, device=device
        ).to(input_dtype),
        "A_log": 0.1 * torch.randn(num_heads, dtype=torch.float32, device=device),
        "dt_bias": 0.1
        * torch.randn((num_heads, head_size), dtype=torch.float32, device=device),
        "cu_seqlens": cu_seqlens,
    }
    if args.has_initial_state:
        inputs["initial_state"] = (
            0.1
            * torch.randn(
                (num_seqs, num_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )
        ).to(torch.bfloat16)
    else:
        inputs["initial_state"] = None
    return inputs


def _reference_prefill(inputs, args, device):
    """Token-serial reference, for the optional refcheck.

    Deliberately the public-contract reference and not either kernel's own: two
    references that share a derivation cannot disagree, so a check against one
    of the implementations would confirm nothing.
    """
    q = inputs["q"]
    batch, tokens, num_heads, head_dim = q.shape
    scale = head_dim**-0.5
    normalize = torch.nn.functional.normalize

    q_flat = normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))

    decay = torch.exp(
        args.lower_bound
        * torch.sigmoid(
            torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
            * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
        )
    )

    if inputs["cu_seqlens"] is None:
        offsets = [index * tokens for index in range(batch + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]

    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        state = inputs["initial_state"].clone()

    out = torch.empty_like(q_flat)
    for sequence in range(len(offsets) - 1):
        for token in range(offsets[sequence], offsets[sequence + 1]):
            decayed = state[sequence].float() * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            out[token] = (
                scale
                * torch.einsum("hk,hvk->hv", q_flat[token], state[sequence].float())
            ).to(torch.bfloat16)
    return out.reshape_as(q), state


def _flashinfer_runner(inputs, args, output, final_state, variant):
    """A zero-argument callable running one FlashInfer backend.

    ``variant is None`` is the public entry point, policy and all. The other
    two go through the internal entry so a report can show what the policy
    chose and what it declined; they are not a second public API.
    """
    if variant is None:

        def call():
            flashinfer.recurrent_kda(
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                g=inputs["g"],
                beta=inputs["beta"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                beta_is_logit=True,
                lower_bound=args.lower_bound,
                cu_seqlens=inputs["cu_seqlens"],
                initial_state=inputs["initial_state"],
                output_final_state=final_state is not None,
                output=output,
            )

        return call

    # Note on what the row above measures against the ones below.  The public
    # entry takes `output_final_state: bool` and has no parameter for a state
    # buffer, so with no `initial_state` it allocates one per call while the
    # direct variants below write into a preallocated tensor.  That is a real
    # difference between the two APIs, not an artifact of this harness, and it
    # is left visible rather than papered over -- with `--has_initial_state`
    # both paths update the same buffer in place and the comparison is exact.

    from flashinfer.kda_kernels import sm120_prefill

    def call():
        sm120_prefill.run_kda_prefill_sm120(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=args.lower_bound,
            cu_seqlens=inputs["cu_seqlens"],
            initial_state=inputs["initial_state"],
            final_state=final_state,
            output=output,
            variant=variant,
        )

    return call


def _external_runner(backend, inputs, args, output, final_state):
    """A callable for an external baseline, or ``None`` if it is not installed.

    Both baselines write the caller's ``output`` and ``final_state`` in place,
    exactly as the FlashInfer path does. A baseline timed without the state
    store would not be measuring the same operation, and dividing by it would
    overstate the speedup by however much that store costs.
    """
    scale = args.head_size**-0.5
    if backend == "cutekda":
        try:
            import cute_kda
        except ImportError:
            return None

        def call():
            cute_kda.fwd(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                scale,
                output,
                inputs["A_log"],
                inputs["dt_bias"],
                args.lower_bound,
                inputs["initial_state"],
                final_state,
                inputs["cu_seqlens"],
            )

        return call

    if backend == "flash-kda":
        try:
            import flash_kda
        except ImportError:
            return None

        def call():
            flash_kda.fwd(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                scale,
                output,
                inputs["A_log"],
                inputs["dt_bias"],
                args.lower_bound,
                inputs["initial_state"],
                final_state,
                inputs["cu_seqlens"],
            )

        return call

    return None


def testRecurrentKDAPrefill(args):
    """Time SM120 ordinary multi-token KDA prefill.

    Every backend gets the same tensors, the same stream, the same
    pre-allocated ``output`` and ``final_state``, and the same timing method.
    Compilation, descriptor construction, metadata preparation and the first
    launch all happen before the timed region, so what is measured is the
    steady-state call a serving loop makes rather than the first one it ever
    made.

    Returns:
        list[dict]: one row per backend.
    """
    if args.verbose >= 1:
        print("[INFO] Running testRecurrentKDAPrefill")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: "
            f"{args.repro_command}"
        )

    backends = args.backends[:]
    num_seqs = args.batch_size
    seq_len = args.s_qo
    num_heads = args.num_q_heads
    head_size = args.head_size
    run_refcheck = args.refcheck
    res = []

    if head_size != HEAD_DIM:
        print(f"[ERROR] KDA prefill is fixed at head_size={HEAD_DIM}. Exiting.")
        return res

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    for backend in list(backends):
        if backend in _EXTERNAL_BACKENDS:
            if _external_runner(backend, {}, args, None, None) is None:
                # Deliberately not silent: a comparison table with a quietly
                # absent baseline reads as "we measured against it".
                print(
                    f"[WARNING] the {backend} baseline is not installed in this "
                    f"environment. Skipping it; the report must say so."
                )
                backends.remove(backend)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    total_tokens = num_seqs * seq_len

    inputs = _make_inputs(args, device)
    output = torch.empty_like(inputs["v"])
    output_final_state = True
    final_state = (
        inputs["initial_state"]
        if inputs["initial_state"] is not None
        else torch.empty(
            (num_seqs, num_heads, head_size, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
    )

    # Which variant the policy picks, and whether this device's thresholds were
    # measured or inherited. A time quoted under fallback thresholds is not a
    # time quoted under tuned ones, and nothing else in the output says which.
    selected_variant = ""
    policy_source = ""
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    try:
        from flashinfer.kda_kernels import sm120_prefill

        selected_variant = sm120_prefill.choose_variant(
            num_seqs, num_heads, seq_len, sm_count=sm_count
        )
        profile_key, _ = sm120_prefill.auto_profile(sm_count)
        policy_source = "tuned" if profile_key == sm_count else "fallback"
        if args.verbose >= 1:
            print(f"[INFO] {sm120_prefill.describe_variant_policy(sm_count)}")
    except Exception as exc:  # noqa: BLE001 -- reporting only
        if args.verbose >= 1:
            print(f"[WARNING] could not read the SM120 variant policy: {exc}")

    reference_output = None
    if run_refcheck:
        reference_output, _ = _reference_prefill(inputs, args, device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {inputs['q'].shape = }, {inputs['q'].dtype = }")
        print(f"[VVERBOSE] {inputs['cu_seqlens'] = }")

    backend_times = {backend: [] for backend in backends}
    outputs = {}

    # KDA updates the recurrent state in place: `final_state` *is*
    # `initial_state` whenever the caller supplies one.  Without a snapshot the
    # first backend's warmup rewrites the state every later backend starts
    # from, so the second one is not solving the same problem as the first --
    # which shows up as a performance difference that is not one, and as a
    # refcheck failure under --has_initial_state that blames the wrong backend.
    initial_state_snapshot = (
        inputs["initial_state"].clone() if inputs["initial_state"] is not None else None
    )

    for backend in backends:
        if backend in _FLASHINFER_BACKENDS:
            variant = {
                "flashinfer": None,
                "flashinfer-decomp": "decomp",
                "flashinfer-fused": "fused",
            }[backend]
            runner = _flashinfer_runner(inputs, args, output, final_state, variant)
        else:
            runner = _external_runner(backend, inputs, args, output, final_state)
        if runner is None:
            continue

        # Warm up outside the timed region: the first call compiles, builds
        # descriptors and reads the offsets on the host, none of which a
        # steady-state call repeats.
        try:
            if initial_state_snapshot is not None:
                inputs["initial_state"].copy_(initial_state_snapshot)
            runner()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 -- reported and skipped
            print(f"[ERROR] backend {backend} failed to run: {exc}")
            continue

        outputs[backend] = output.clone()
        if initial_state_snapshot is not None:
            inputs["initial_state"].copy_(initial_state_snapshot)
        backend_times[backend] = bench_gpu_time(
            fn=runner,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            # CUDA-event timing, not graph-replay timing, and not because graphs
            # do not work here -- they do, and the test suite captures every
            # published combination. They need a protocol this harness cannot
            # follow: a caller-owned RecurrentKDAPrefillWorkspace, warmed
            # eagerly on the *capture* stream with the exact tensors, then a
            # sync before capture. ``bench_gpu_time`` owns the stream and warms
            # on its own, so a capture from here starts cold on a stream the
            # backend has never seen, and the first thing it needs is an
            # allocation. Timing the eager path is the honest measurement of
            # what this harness is actually able to set up.
            use_cuda_graph=False,
        )

    if run_refcheck and reference_output is not None:
        rtol, atol = 1e-2, 1e-2
        for backend, tested in outputs.items():
            (
                num_different_elements,
                num_elements,
                num_different_elements_percentage,
            ) = is_close_stats(
                reference_output.float(), tested.float(), rtol=rtol, atol=atol
            )
            mismatch_threshold_pct = 0.01
            if num_different_elements_percentage > mismatch_threshold_pct:
                print(
                    f"[ERROR] Output tensor mismatch from backend {backend}: "
                    f"{num_different_elements}/{num_elements} "
                    f"({num_different_elements_percentage:.4f}%) elements differ "
                    f"(threshold: {mismatch_threshold_pct}%)"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with "
                        f"{num_different_elements} elements"
                    )
            elif args.verbose >= 1:
                print(f"[REFCHECK] Backend {backend}: PASSED")

    problem_flops = kda_prefill_flops(total_tokens, num_heads, head_size)
    problem_bytes = kda_prefill_bytes(
        total_tokens,
        num_seqs,
        num_heads,
        head_size,
        input_dtype,
        inputs["initial_state"] is not None,
        output_final_state,
    )

    for backend in backends:
        if len(backend_times[backend]) == 0:
            continue
        median_time = np.median(backend_times[backend])
        std_time = np.std(backend_times[backend])
        tflops = problem_flops / (10**9 * median_time)
        tb_per_sec = problem_bytes / (10**9 * median_time)

        print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend
            cur_res["batch_size"] = num_seqs
            cur_res["s_qo"] = seq_len
            cur_res["num_q_heads"] = num_heads
            cur_res["head_size"] = head_size
            cur_res["input_dtype"] = str(input_dtype)
            cur_res["state_dtype"] = args.state_dtype
            cur_res["packed"] = args.packed
            cur_res["has_initial_state"] = args.has_initial_state
            cur_res["kda_variant"] = (
                selected_variant if backend == "flashinfer" else backend
            )
            cur_res["kda_variant_policy"] = policy_source
            cur_res["sm_count"] = sm_count
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def kda_decode_flops(num_rows: int, num_heads: int, head_size: int) -> float:
    """Matmul FLOPs of one fused KDA decode step, same convention as prefill."""
    return 3.0 * 2.0 * num_rows * num_heads * head_size * head_size


def kda_decode_bytes(num_rows: int, num_heads: int, state_dtype: torch.dtype) -> float:
    """Compulsory DRAM traffic of one fused KDA decode step.

    The recurrent state is read and written once per row; the convolution
    history (three taps of the packed QKV channels) likewise. Activations are
    the packed QKV row, the gate and beta, the output gate and the output.
    """
    hidden = num_heads * HEAD_DIM
    state = 2.0 * num_rows * hidden * HEAD_DIM * state_dtype.itemsize
    conv = 2.0 * num_rows * 3 * hidden * _DECODE_CONV_HISTORY * 2
    activations = num_rows * (3 * hidden + hidden + num_heads + 2 * hidden) * 2
    return state + conv + activations


def _make_decode_inputs(args, device):
    """Inputs matching the fused_kda_decode contract.

    The convolution cache and the recurrent state are laid out as one page per
    slot -- ``[conv history | recurrent state]`` -- the hybrid-cache layout a
    serving stack hands to this kernel, with slot zero reserved as the null
    slot.
    """
    num_heads = args.num_q_heads
    num_rows = args.batch_size
    num_slots = num_rows + 1
    hidden = num_heads * HEAD_DIM
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)

    conv_slot_bytes = 3 * hidden * _DECODE_CONV_HISTORY * 2
    state_slot_bytes = num_heads * HEAD_DIM * HEAD_DIM * state_dtype.itemsize
    page_bytes = conv_slot_bytes + state_slot_bytes

    def randn(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)

    conv_state = torch.empty_strided(
        (num_slots, 3 * hidden, _DECODE_CONV_HISTORY),
        (page_bytes // 2, 1, 3 * hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state.copy_(
        0.1 * randn((num_slots, 3 * hidden, _DECODE_CONV_HISTORY), torch.bfloat16)
    )
    state = torch.empty_strided(
        (num_slots, num_heads, HEAD_DIM, HEAD_DIM),
        (page_bytes // state_dtype.itemsize, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        dtype=state_dtype,
        device=device,
    )
    state.copy_(0.01 * randn((num_slots, num_heads, HEAD_DIM, HEAD_DIM)))
    return {
        "x": randn((num_rows, 3 * hidden), torch.bfloat16),
        "weight": 0.1 * randn((3, 4, hidden)),
        "conv_state": conv_state,
        "raw_gate": randn((1, num_rows, num_heads, HEAD_DIM), torch.bfloat16),
        "raw_beta": randn((1, num_rows, num_heads), torch.bfloat16),
        "A_log": 0.5 * randn((num_heads,)),
        "dt_bias": 0.1 * randn((hidden,)),
        "state_indices": torch.arange(
            num_rows, 0, -1, dtype=torch.int32, device=device
        ),
        "state": state,
        "output_gate": randn((num_rows, num_heads, HEAD_DIM), torch.bfloat16),
        "norm_weight": randn((HEAD_DIM,)),
    }


@torch.no_grad()
def _reference_decode(inputs, conv_state, state, lower_bound):
    """Token-serial torch reference of the fused decode, for the refcheck.

    Updates ``conv_state`` and ``state`` in place, as the kernel does.
    """
    num_rows = inputs["x"].shape[0]
    num_heads = inputs["A_log"].numel()
    hidden = num_heads * HEAD_DIM
    slots = inputs["state_indices"].long()

    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden, 4)
    history = conv_state.index_select(0, slots).float()
    window = torch.cat((history, inputs["x"].float().unsqueeze(-1)), dim=-1)
    conv_state.index_copy_(0, slots, window[:, :, 1:].to(torch.bfloat16))
    mixed = F.silu((window * taps).sum(-1)).to(torch.bfloat16).float()
    query, key, value = mixed.view(num_rows, 3, num_heads, HEAD_DIM).unbind(1)

    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    query = query * HEAD_DIM**-0.5
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    gate = inputs["raw_gate"][0].float() + inputs["dt_bias"].view(num_heads, HEAD_DIM)
    gate = lower_bound * torch.sigmoid(inputs["A_log"].exp()[None, :, None] * gate)

    selected = state.index_select(0, slots).float() * gate.exp().unsqueeze(-2)
    delta = value - torch.einsum("nhvk,nhk->nhv", selected, key)
    delta = delta * inputs["raw_beta"][0].float().sigmoid().unsqueeze(-1)
    selected = selected + delta.unsqueeze(-1) * key.unsqueeze(-2)
    state.index_copy_(0, slots, selected.to(state.dtype))

    output = torch.einsum("nhvk,nhk->nhv", selected, query).to(torch.bfloat16).float()
    inverse_rms = torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-5)
    output = (
        output
        * inverse_rms
        * inputs["norm_weight"]
        * inputs["output_gate"].float().sigmoid()
    )
    return output.unsqueeze(0).to(torch.bfloat16)


def _clone_strided(tensor):
    clone = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    clone.copy_(tensor)
    return clone


def testFusedKDADecode(args):
    """Time the fused Kimi KDA decode (``flashinfer.kda_decode.fused_kda_decode``).

    One decode token per row: width-four causal convolution with SiLU, one
    recurrent KDA update and gated RMSNorm, with the convolution cache and the
    recurrent state updated in place. Every backend starts from the same cache
    contents. A backend the device or configuration cannot serve -- e.g.
    ``cake``, whose exported kernels cover SM100a/SM103a only -- is reported
    and skipped.

    Returns:
        list[dict]: one row per backend.
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedKDADecode")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: "
            f"{args.repro_command}"
        )

    res = []
    if args.head_size != HEAD_DIM:
        print(f"[ERROR] fused KDA decode is fixed at head_size={HEAD_DIM}. Exiting.")
        return res
    backends = filter_backends_by_compute_capability(
        args.backends[:], args.routine, device
    )
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    from flashinfer.kda_decode import fused_kda_decode

    num_rows = args.batch_size
    num_heads = args.num_q_heads
    state_dtype = dtype_str_to_torch_dtype(args.state_dtype)
    inputs = _make_decode_inputs(args, device)
    output = torch.empty(
        (1, num_rows, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    conv_snapshot = _clone_strided(inputs["conv_state"])
    state_snapshot = _clone_strided(inputs["state"])

    def restore_caches():
        inputs["conv_state"].copy_(conv_snapshot)
        inputs["state"].copy_(state_snapshot)

    reference_output = None
    if args.refcheck:
        reference_output = _reference_decode(
            inputs,
            _clone_strided(conv_snapshot),
            _clone_strided(state_snapshot),
            args.lower_bound,
        )

    # The per-row activations are the timed call's arguments, so cold-L2
    # timing can rotate them. The caches stay fixed: they are strided views
    # into one page per slot, a layout a rotating-buffer clone would not keep.
    activations = (
        inputs["x"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["output_gate"],
        output,
    )

    def make_runner(backend):
        mode = None if backend == "cute-dsl" else "positive_unique"

        def call(x, raw_gate, raw_beta, output_gate, out):
            fused_kda_decode(
                x=x,
                weight=inputs["weight"],
                conv_state=inputs["conv_state"],
                raw_gate=raw_gate,
                raw_beta=raw_beta,
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                state_indices=inputs["state_indices"],
                state=inputs["state"],
                output_gate=output_gate,
                norm_weight=inputs["norm_weight"],
                lower_bound=args.lower_bound,
                output=out,
                backend=backend,
                state_indices_mode=mode,
            )

        return call

    backend_times = {}
    refcheck_results = {}
    for backend in backends:
        runner = make_runner(backend)
        # The first call compiles or loads the kernel; a backend with no route
        # for this device and configuration raises here.
        try:
            restore_caches()
            runner(*activations)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 -- reported and skipped
            print(
                f"[WARNING] fused_kda_decode backend {backend} cannot run this "
                f"case: {exc}. Skipping."
            )
            continue

        if reference_output is not None:
            (
                num_different_elements,
                num_elements,
                num_different_elements_percentage,
            ) = is_close_stats(
                reference_output.float(), output.float(), rtol=3e-2, atol=2e-2
            )
            mismatch_threshold_pct = 0.01
            passed = num_different_elements_percentage <= mismatch_threshold_pct
            refcheck_results[backend] = passed
            if not passed:
                print(
                    f"[ERROR] Output tensor mismatch from backend {backend}: "
                    f"{num_different_elements}/{num_elements} "
                    f"({num_different_elements_percentage:.4f}%) elements differ "
                    f"(threshold: {mismatch_threshold_pct}%)"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with "
                        f"{num_different_elements} elements"
                    )
            elif args.verbose >= 1:
                print(f"[REFCHECK] Backend {backend}: PASSED")

        restore_caches()
        backend_times[backend] = bench_gpu_time(
            fn=runner,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph,
            input_args=activations,
        )

    problem_flops = kda_decode_flops(num_rows, num_heads, HEAD_DIM)
    problem_bytes = kda_decode_bytes(num_rows, num_heads, state_dtype)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    for backend, times in backend_times.items():
        median_time = np.median(times)
        std_time = np.std(times)
        tflops = problem_flops / (10**9 * median_time)
        tb_per_sec = problem_bytes / (10**9 * median_time)

        print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend
            cur_res["batch_size"] = num_rows
            cur_res["s_qo"] = 1
            cur_res["num_q_heads"] = num_heads
            cur_res["head_size"] = HEAD_DIM
            cur_res["input_dtype"] = str(torch.bfloat16)
            cur_res["state_dtype"] = args.state_dtype
            cur_res["sm_count"] = sm_count
            cur_res["refcheck_passed"] = refcheck_results.get(backend, "")
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res
