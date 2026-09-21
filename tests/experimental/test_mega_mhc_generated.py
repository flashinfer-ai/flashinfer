"""Independent fused mHC values, scale bytes, current stream and graph epochs."""
import pytest
import torch
from flashinfer.mega_mhc import prepare_mega_mhc


def inputs(tokens, shifted):
    hidden = 5120
    token_scale = 1.0 + (torch.arange(tokens, device="cuda", dtype=torch.float32) % 2) * 0.5
    pattern = torch.tensor([1.0, 0.5, -1.0, -0.5], device="cuda")
    return dict(
        x=token_scale[:, None].expand(tokens, hidden).to(torch.bfloat16).contiguous(),
        residual=torch.tensor([0.25, 0.5, 0.75, 1.0], device="cuda", dtype=torch.bfloat16)
            .view(1, 4, 1).expand(tokens, 4, hidden).contiguous(),
        post_mix=torch.full((tokens, 4, 1), 0.5, device="cuda"),
        comb_res_mix=torch.eye(4, device="cuda").expand(tokens, 4, 4).contiguous(),
        shifted_prev_mix=torch.full((tokens, 4, 1), 0.25, device="cuda") if shifted else None,
        fn=((torch.arange(24, device="cuda", dtype=torch.float32) - 12) / 1024)
            .view(24, 1).expand(24, 4 * hidden).contiguous(),
        mix_scales=torch.tensor([1 / 16, 1 / 32, 1 / 64], device="cuda"),
        mix_bases=(torch.arange(24, device="cuda", dtype=torch.float32) - 12) / 128,
        rmsnorm_weight=pattern[(torch.arange(hidden, device="cuda") // 32) % 4].bfloat16(),
        hc_mult=4, hc_norm_eps=2e-5, hc_pre_eps=3e-4, hc_post_scale=1.25,
        sinkhorn_eps=2e-6, num_sinkhorn_iters=20, rmsnorm_eps=7e-6, rmsnorm_scale=1.25)


def reference(values, layout):
    tokens, hidden = values["x"].shape
    # comb_res_mix is the identity and every projection row is constant. This
    # closed evaluation avoids a second GPU GEMM implementation as the oracle.
    residual = (values["x"].float().unsqueeze(1) * values["post_mix"] + values["residual"].float()).bfloat16()
    mixes = residual.float().sum((1, 2))[:, None] * values["fn"][:, 0]
    mixes *= torch.rsqrt(residual.float().square().mean((1, 2)) + values["hc_norm_eps"]).unsqueeze(1)
    scales = torch.cat((values["mix_scales"][0].expand(4), values["mix_scales"][1].expand(4),
                        values["mix_scales"][2].expand(16)))
    mixes = mixes * scales + values["mix_bases"]
    prev_mix = mixes[:, :4].sigmoid().unsqueeze(2) + values["hc_pre_eps"]
    comb = mixes[:, 8:].reshape(tokens, 4, 4).softmax(-1) + values["sinkhorn_eps"]
    comb = comb / (comb.sum(-2, keepdim=True) + values["sinkhorn_eps"])
    for _ in range(1, values["num_sinkhorn_iters"]):
        comb = comb / (comb.sum(-1, keepdim=True) + values["sinkhorn_eps"])
        comb = comb / (comb.sum(-2, keepdim=True) + values["sinkhorn_eps"])
    pre = values["shifted_prev_mix"] if values["shifted_prev_mix"] is not None else prev_mix
    norm_input = (residual.float() * pre).sum(1).bfloat16().float()
    norm_scale = torch.rsqrt(norm_input.square().mean(1) + values["rmsnorm_eps"]) * values["rmsnorm_scale"]
    normalized = (norm_input * norm_scale[:, None] * values["rmsnorm_weight"].float()).bfloat16()
    result = dict(new_residual=residual, new_post_mix=(mixes[:, 4:8].sigmoid() * values["hc_post_scale"]).unsqueeze(2),
                  new_comb_res_mix=comb, y_bf16=normalized)
    if values["shifted_prev_mix"] is not None:
        result["new_prev_mix"] = prev_mix
    groups = normalized.reshape(tokens, hidden // 32, 32)
    amax_bits = groups.float().abs().amax(-1).bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    exponents = (((amax_bits + 0x1F) >> 7).clamp_min(113) - 8)
    sf_inv = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), 127 - exponents)
    result["y_fp8"] = (groups.float() * sf_inv[:, :, None]).to(torch.float8_e4m3fn).reshape(tokens, hidden)
    words = (exponents.to(torch.int64).reshape(tokens, hidden // 128, 4)
             << (torch.arange(4, device="cuda", dtype=torch.int64) * 8)).sum(-1).to(torch.int32)
    result["y_gemm_sf" if layout == "col" else "y_routed_sf"] = words
    if layout == "extra":
        result["y_shared_sf"] = words
    return result


def logical(outputs):
    values = {key: value for key, value in outputs.items() if isinstance(value, torch.Tensor)
              and key not in ("y_shared_sf", "y_shared_sf_storage")}
    if "y_shared_sf_storage" in outputs:
        tokens = outputs["y_bf16"].shape[0]
        block = outputs["shared_sf_block_m"]
        rows = torch.arange(tokens, device="cuda")
        index = rows % block
        physical = rows // block * ((block + 127) // 128 * 128) + (index & ~127) + (index & 31) * 4 + ((index >> 5) & 3)
        values["y_shared_sf"] = outputs["y_shared_sf_storage"][physical]
    return values


def check(outputs, expected):
    for name, value in logical(outputs).items():
        tolerance = 0 if value.dtype == torch.int32 else (0.1 if name == "y_fp8" else
                    1e-2 if value.dtype == torch.bfloat16 else 1e-5)
        if value.dtype == torch.int32:
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        else:
            torch.testing.assert_close(value.float(), expected[name].float(), atol=tolerance, rtol=tolerance)


def poison(outputs):
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            value.fill_(-559038737 if value.dtype == torch.int32 else float("nan"))


@pytest.mark.parametrize("tokens", (1, 64, 65, 200, 1025, 4096))
@pytest.mark.parametrize("shifted", (False, True))
@pytest.mark.parametrize("layout", ("col", "extra"))
def test_fused_values_scales_and_changed_input_replay(tokens, shifted, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103a required")
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip("The exported schedule requires 152 SMs")
    values = inputs(tokens, shifted)
    plan = prepare_mega_mhc(**values, sf_layout=layout)
    expected = reference(values, layout)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    first = None
    for _ in range(3):
        with torch.cuda.stream(stream):
            poison(plan.outputs)
            plan.run()
        stream.synchronize()
        check(plan.outputs, expected)
        actual = logical(plan.outputs)
        if first is None:
            first = {name: value.clone() for name, value in actual.items()}
        else:
            for name, value in actual.items():
                if value.dtype == torch.int32:
                    torch.testing.assert_close(value, first[name], atol=0, rtol=0)
                else:
                    torch.testing.assert_close(value.float(), first[name].float(), atol=0, rtol=0)
        stream.wait_stream(torch.cuda.current_stream())
    if not (tokens in (65, 1025) or (shifted and tokens in (1, 64, 200))):
        return
    graphs = []
    for _ in range(2):
        with torch.cuda.stream(stream):
            plan.run()
            plan.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        graphs.append(graph)
    stream.synchronize()
    previous = plan.launch_epochs.cpu().tolist()
    active = plan.config["num_launch_sms"]
    for repeat in range(12):
        with torch.cuda.stream(stream):
            values["x"].add_(0.015625)
            expected = reference(values, layout)
            poison(plan.outputs)
            if repeat in (5, 10):
                plan.run()
            else:
                graphs[0 if repeat < 7 else 1].replay()
        stream.synchronize()
        check(plan.outputs, expected)
        current = plan.launch_epochs.cpu().tolist()
        assert len(current) == 152
        assert len(set(current[:active])) == 1
        assert all(after == before + 1 for before, after in zip(previous[:active], current[:active], strict=True))
        assert current[active:] == previous[active:]
        previous = current
        stream.wait_stream(torch.cuda.current_stream())
