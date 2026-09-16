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

import pytest
import torch

from flashinfer.gdn_decode import (
    gated_delta_rule_mtp,
    gated_delta_rule_replayssm_commit,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="ReplaySSM requires SM100/SM103",
)


def _inputs(batch=4, steps=8, heads=2, value_heads=8, dtype=torch.bfloat16):
    torch.manual_seed(42)
    device = "cuda"
    slots = batch + 3

    def rand(*shape, dtype=dtype):
        return torch.randn(*shape, dtype=dtype, device=device) * 0.1

    q = rand(batch, steps, heads, 128)
    k = rand(batch, steps, heads, 128)
    v = rand(batch, steps, value_heads, 128)
    state = rand(slots, value_heads, 128, 128, dtype=torch.float32)
    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device=device)
    if batch > 1:
        indices[-1] = -1
    args = dict(
        q=q,
        k=k,
        v=v,
        initial_state=state,
        initial_state_indices=indices,
        A_log=rand(value_heads, dtype=torch.float32),
        a=rand(batch, steps, value_heads),
        dt_bias=rand(value_heads, dtype=torch.float32),
        b=rand(batch, steps, value_heads),
        disable_state_update=True,
        use_qk_l2norm=True,
        scale=0.37,
    )
    cache = dict(
        replayssm_rawv=torch.full(
            (slots, value_heads, steps, 128), 7.0, dtype=torch.bfloat16, device=device
        ),
        replayssm_rawk=torch.full(
            (slots, heads, steps, 128), 7.0, dtype=torch.bfloat16, device=device
        ),
        replayssm_g=torch.full((slots, value_heads, steps), 7.0, device=device),
        replayssm_beta=torch.full((slots, value_heads, steps), 7.0, device=device),
    )
    return args, cache


def _reference(args):
    """FP64 recurrence independent of the CuTe layouts and TF32 contractions."""
    q, k, v = (args[n].to(torch.bfloat16).double() for n in ("q", "k", "v"))
    batch, steps, heads, _ = q.shape
    hv = v.shape[2]
    if args["use_qk_l2norm"]:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // heads, dim=2) * args["scale"]
    k = k.repeat_interleave(hv // heads, dim=2)
    a, b = (args[n].to(torch.bfloat16).double() for n in ("a", "b"))
    log_g = -args["A_log"].double().exp() * torch.nn.functional.softplus(
        a + args["dt_bias"].double()
    )
    beta = b.sigmoid()
    indices = args["initial_state_indices"].long().clamp_min(0)
    state = args["initial_state"][indices].double()
    outputs, states = [], []
    for t in range(steps):
        state = state * log_g[:, t, :, None, None].exp()
        prediction = (state * k[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - prediction) * beta[:, t, :, None]
        state = state + delta[..., None] * k[:, t, :, None, :]
        outputs.append((state * q[:, t, :, None, :]).sum(-1))
        states.append(state.clone())
    return torch.stack(outputs, 1), torch.stack(states, 1), log_g, beta


@pytest.mark.parametrize(
    "batch,steps,heads,hv,variant",
    [
        (1, 4, 2, 8, "default"),
        (1, 8, 2, 8, "default"),
        (2, 4, 2, 8, "default"),
        (4, 8, 2, 8, "default"),
        (17, 4, 2, 8, "default"),
        (256, 8, 1, 2, "default"),
        (257, 4, 1, 2, "default"),
        (2, 3, 2, 8, "default"),
        (17, 5, 2, 8, "default"),
        (2, 16, 2, 8, "default"),
        (2, 8, 2, 8, "unnormalized"),
        (17, 4, 2, 8, "unnormalized"),
        (2, 4, 2, 8, "int64"),
        (17, 8, 2, 8, "int64"),
        (2, 8, 3, 3, "default"),
        (17, 4, 2, 6, "default"),
        (2, 4, 2, 8, "strided"),
        (17, 8, 2, 8, "strided"),
        (2, 4, 2, 8, "state_strided"),
        (17, 8, 2, 8, "state_strided"),
        (2, 8, 2, 8, "fp16"),
        (17, 4, 2, 8, "fp16"),
        (2, 8, 2, 8, "zero_qk"),
        (2, 4, 2, 8, "extreme_gates"),
        (2, 8, 2, 8, "all_padding"),
        (2, 8, 2, 8, "bf16_bias"),
        (2, 8, 2, 8, "strided_bias"),
        (17, 5, 2, 8, "strided_bias"),
        (2, 4, 2, 8, "fp32_output"),
        (17, 8, 2, 8, "fp32_output"),
        (2, 8, 2, 8, "inner_strided"),
        (17, 5, 2, 8, "inner_strided"),
        (2, 8, 2, 8, "extreme_gates_fallback"),
        (17, 5, 2, 8, "extreme_gates_fallback"),
    ],
)
def test_replayssm_verify(batch, steps, heads, hv, variant):
    args, cache = _inputs(
        batch, steps, heads, hv, torch.float16 if variant == "fp16" else torch.bfloat16
    )
    if variant == "unnormalized":
        args["use_qk_l2norm"] = False
    if variant == "int64":
        args["initial_state_indices"] = args["initial_state_indices"].long()
    if variant == "strided":
        for name in ("q", "k", "v", "a", "b"):
            value = args[name]
            storage = torch.empty(
                (*value.shape[:-1], value.shape[-1] * 2),
                dtype=value.dtype,
                device="cuda",
            )
            args[name] = storage[..., : value.shape[-1]]
            args[name].copy_(value)
    if variant == "inner_strided":
        for name in ("q", "k", "v"):
            value = args[name]
            storage = torch.empty(
                (*value.shape[:-1], value.shape[-1] * 2),
                dtype=value.dtype,
                device="cuda",
            )
            args[name] = storage[..., ::2]
            args[name].copy_(value)
    if variant == "strided_bias":
        for name in ("A_log", "dt_bias"):
            value = args[name]
            storage = torch.empty(value.numel() * 2, dtype=value.dtype, device="cuda")
            args[name] = storage[::2]
            args[name].copy_(value)
    if variant == "bf16_bias":
        args["dt_bias"] = args["dt_bias"].bfloat16()
    if variant == "state_strided":
        state = args["initial_state"]
        storage = torch.empty((state.shape[0] * 2, *state.shape[1:]), device="cuda")
        args["initial_state"] = storage[::2]
        args["initial_state"].copy_(state)
    if variant == "zero_qk":
        args["q"].zero_()
        args["k"].zero_()
    if variant == "extreme_gates_fallback":
        args["initial_state_indices"] = args["initial_state_indices"].long()
    if variant in ("extreme_gates", "extreme_gates_fallback"):
        args["a"][:, 0].fill_(100.0)
        args["a"][:, 1].fill_(-30.0)
        args["b"][:, 0].fill_(-100.0)
        args["b"][:, 1].fill_(100.0)
    if variant == "all_padding":
        args["initial_state_indices"].fill_(-1)
    before = args["initial_state"].clone()
    expected, _, log_g, beta = _reference(args)
    out = torch.full_like(args["v"], 3.0)
    if variant == "fp32_output":
        out = out.float()
    output, state = gated_delta_rule_mtp(
        **args, output=out, cache_replayssm=True, **cache
    )
    assert output is out and state is args["initial_state"]
    live = args["initial_state_indices"] >= 0
    torch.testing.assert_close(out[live].double(), expected[live], atol=1e-4, rtol=1e-2)
    torch.testing.assert_close(
        out[~live], torch.full_like(out[~live], 3.0), atol=0, rtol=0
    )
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    slots = args["initial_state_indices"][live].long()
    for name, ref in (
        ("replayssm_rawv", args["v"].to(torch.bfloat16).permute(0, 2, 1, 3)),
        ("replayssm_rawk", args["k"].to(torch.bfloat16).permute(0, 2, 1, 3)),
        ("replayssm_g", log_g.permute(0, 2, 1)),
        ("replayssm_beta", beta.permute(0, 2, 1)),
    ):
        tol = 0 if "raw" in name else 2e-6
        torch.testing.assert_close(
            cache[name][slots].double(), ref[live].double(), atol=tol, rtol=tol
        )
        untouched = torch.ones(cache[name].shape[0], device="cuda", dtype=torch.bool)
        untouched[slots] = False
        torch.testing.assert_close(
            cache[name][untouched],
            torch.full_like(cache[name][untouched], 7.0),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("steps", [4, 8])
@pytest.mark.parametrize(
    "normalize,track", [(True, False), (True, True), (False, True)]
)
@pytest.mark.parametrize("tcgen", [False, True])
def test_replayssm_verify_commit_cycles(steps, normalize, track, tcgen):
    if tcgen and (steps != 8 or not normalize or track):
        pytest.skip("tcgen commit specializes normalized T=8 without tracking")
    batch = steps + 3
    args, cache = _inputs(batch, steps)
    args["use_qk_l2norm"] = normalize
    # Include every accepted length, a zero-accept live row and a null row.
    accepts = torch.tensor(
        list(range(steps + 1)) + [0, steps], device="cuda", dtype=torch.int32
    )
    layers = 2
    slots, hv, v, k = args["initial_state"].shape
    checkpoint = torch.randn(layers, slots * 2, hv, v, k, device="cuda") * 0.1
    windows = {
        name: torch.empty(
            (layers, slots * 2, *value.shape[1:]), device="cuda", dtype=value.dtype
        )
        for name, value in cache.items()
    }
    track_ids = torch.arange(slots, slots + batch, device="cuda", dtype=torch.int32)
    track_steps = torch.arange(batch, device="cuda", dtype=torch.int32) % steps
    track_ids[1] = -1
    for _ in range(2):
        before = checkpoint.clone()
        expected_state = checkpoint.clone()
        for layer in range(layers):
            args["initial_state"] = checkpoint[layer]
            args["v"] = torch.randn_like(args["v"]) * 0.1
            ref_out, ref_states, _, _ = _reference(args)
            output, _ = gated_delta_rule_mtp(
                **args,
                cache_replayssm=True,
                **{name: value[layer] for name, value in windows.items()},
            )
            torch.testing.assert_close(
                output[:-1].double(), ref_out[:-1], atol=1e-4, rtol=1e-2
            )
            for row, (slot, count) in enumerate(
                zip(
                    args["initial_state_indices"].tolist(),
                    accepts.tolist(),
                    strict=True,
                )
            ):
                if slot < 0 or count == 0:
                    continue
                expected_state[layer, slot] = ref_states[row, count - 1].float()
                if track and track_ids[row] >= 0 and track_steps[row] < count:
                    expected_state[layer, track_ids[row]] = ref_states[
                        row, track_steps[row]
                    ].float()
        torch.testing.assert_close(checkpoint, before, atol=0, rtol=0)
        # A rejected suffix must not influence the accepted checkpoint.
        for slot, count in zip(
            args["initial_state_indices"].tolist(), accepts.tolist(), strict=True
        ):
            if slot >= 0:
                for window in windows.values():
                    window[:, slot, :, count:] = float("nan")
        gated_delta_rule_replayssm_commit(
            checkpoint,
            windows["replayssm_rawv"],
            windows["replayssm_rawk"],
            windows["replayssm_g"],
            windows["replayssm_beta"],
            args["initial_state_indices"],
            accepts,
            use_qk_l2norm=normalize,
            track_state_indices=track_ids if track else None,
            track_steps=track_steps if track else None,
            backend="tcgen05" if tcgen else "simt",
        )
        torch.testing.assert_close(
            checkpoint,
            expected_state,
            atol=1e-4 if tcgen else 2e-6,
            rtol=1e-3 if tcgen else 1e-5,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "update",
        "snapshots",
        "scatter",
        "shape",
        "dtype",
        "strides",
        "disabled",
    ],
)
def test_replayssm_verify_validation(invalid):
    args, cache = _inputs()
    if invalid == "missing":
        cache.pop("replayssm_beta")
    if invalid == "update":
        args["disable_state_update"] = False
    if invalid == "snapshots":
        args["intermediate_states_buffer"] = torch.empty(
            4, 8, 8, 128, 128, device="cuda"
        )
    if invalid == "scatter":
        args["ssm_state_indices"] = torch.zeros(4, 8, device="cuda", dtype=torch.int32)
    if invalid == "shape":
        cache["replayssm_rawk"] = cache["replayssm_rawk"][:, :, :-1].contiguous()
    if invalid == "dtype":
        cache["replayssm_rawv"] = cache["replayssm_rawv"].float()
    if invalid == "strides":
        cache["replayssm_g"] = cache["replayssm_g"].transpose(1, 2)
    with pytest.raises((AssertionError, ValueError)):
        gated_delta_rule_mtp(**args, cache_replayssm=invalid != "disabled", **cache)


@pytest.mark.parametrize("backend", ["auto", "simt"])
def test_replayssm_cuda_graph(backend):
    args, cache = _inputs()
    output = torch.empty_like(args["v"])
    before = args["initial_state"].clone()
    accepts = torch.tensor([0, 1, 4, 8], device="cuda", dtype=torch.int32)

    def cycle():
        gated_delta_rule_mtp(**args, output=output, cache_replayssm=True, **cache)
        gated_delta_rule_replayssm_commit(
            args["initial_state"].unsqueeze(0),
            cache["replayssm_rawv"].unsqueeze(0),
            cache["replayssm_rawk"].unsqueeze(0),
            cache["replayssm_g"].unsqueeze(0),
            cache["replayssm_beta"].unsqueeze(0),
            args["initial_state_indices"],
            accepts,
            backend=backend,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cycle()
    torch.cuda.current_stream().wait_stream(stream)
    expected_output, expected_state = output.clone(), args["initial_state"].clone()
    args["initial_state"].copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cycle()
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:-1], expected_output[:-1], atol=0, rtol=0)
    torch.testing.assert_close(args["initial_state"], expected_state, atol=0, rtol=0)


@pytest.mark.parametrize(
    "invalid",
    [
        "backend",
        "tcgen_track",
        "tcgen_norm",
        "track_pair",
        "dtype",
        "shape",
        "index_dtype",
        "null_id",
        "cpu",
    ],
)
def test_replayssm_commit_validation(invalid):
    from flashinfer.trace.templates.gdn import gdn_replayssm_commit_trace

    args = gdn_replayssm_commit_trace.init()
    if invalid == "backend":
        args["backend"] = "invalid"
    if invalid == "tcgen_track":
        args.update(
            backend="tcgen05",
            track_state_indices=args["state_indices"],
            track_steps=args["accept_lens"],
        )
    if invalid == "tcgen_norm":
        args.update(backend="tcgen05", use_qk_l2norm=False)
    if invalid == "track_pair":
        args["track_steps"] = args["accept_lens"]
    if invalid == "dtype":
        args["checkpoint_state"] = args["checkpoint_state"].bfloat16()
    if invalid == "shape":
        args["rawv_cache"] = args["rawv_cache"][:, :, :, :-1].contiguous()
    if invalid == "index_dtype":
        args["state_indices"] = args["state_indices"].long()
    if invalid == "null_id":
        args["null_block_id"] = -2
    if invalid == "cpu":
        args["accept_lens"] = args["accept_lens"].cpu()
    with pytest.raises((ValueError, TypeError)):
        gated_delta_rule_replayssm_commit(**args)


@pytest.mark.parametrize("batch", [0, 1])
@pytest.mark.parametrize("backend", ["auto", "simt"])
def test_replayssm_commit_noop(batch, backend):
    from flashinfer.trace.templates.gdn import gdn_replayssm_commit_trace

    args = gdn_replayssm_commit_trace.init(batch_size=batch)
    before = args["checkpoint_state"].clone()
    args["accept_lens"].zero_()
    gated_delta_rule_replayssm_commit(**args, backend=backend)
    torch.testing.assert_close(args["checkpoint_state"], before, atol=0, rtol=0)


def test_replayssm_rejects_other_architectures(monkeypatch):
    from flashinfer.gdn_kernels import device_target

    args, cache = _inputs()
    target = device_target.gdn_device_target(args["q"].device)
    monkeypatch.setattr(
        device_target, "gdn_device_target", lambda device: target._replace(major=12)
    )
    with pytest.raises(ValueError, match="SM100/SM103"):
        gated_delta_rule_mtp(**args, cache_replayssm=True, **cache)
