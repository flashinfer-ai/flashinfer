"""Real TP2 PCIe collective benchmark requests with retained native owners."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import (
    CollectiveRequirement,
    PreparedCall,
    PreparationRequest,
)


@dataclass
class _Collective:
    runtime: object
    source: object
    base: object
    residual: object
    weight: object
    epsilon: float
    world_size: int
    fused: bool


class _Owner:
    def __init__(self, *, device, max_bytes, group):
        self.device = device
        self.max_bytes = max_bytes
        self.group = group
        self.runtime = None

    def get(self):
        if self.runtime is None:
            from b12x.comm import pcie

            self.runtime = pcie.OneshotAllReduce.from_exchange_group(
                exchange_group=self.group,
                device=self.device,
                eager_buffer_bytes=self.max_bytes,
                max_size=self.max_bytes,
            )
        return self.runtime

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None




def make_benchmark_requests(
    metadata, *, device, rows
) -> list[PreparationRequest]:
    import torch
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.comm.pcie._oneshot_preparation import query_from_metadata

    hidden = int(metadata["hidden_size"])
    rank = dist.get_rank()
    tp = dist.get_world_size()
    if tp != 2:
        raise ValueError(f"PCIe benchmark fixtures require actual TP2, got world size {tp}")
    maximum = max(512 << 10, max(rows) * hidden * torch.empty((), dtype=torch.bfloat16).element_size())
    owner = _Owner(device=device, max_bytes=maximum, group=dist.group.WORLD)
    runtime = owner.get()
    requests = []
    for tokens in rows:
        for fused in (False, True):
            if fused and tokens > 36:
                continue
            surface = (
                "OneshotAllReduce.all_reduce_fused_add_rms_norm"
                if fused else "OneshotAllReduce.all_reduce"
            )
            query = query_from_metadata(
                runtime, surface=surface, shape=(tokens, hidden), dtype=torch.bfloat16,
            )
            declaration = pcie.plan(query, runtime=runtime)

            def collective_call(state, *, tokens=tokens, fused=fused):
                generator = torch.Generator(device=device).manual_seed(239)
                base = (
                    torch.randn(
                        (tokens, hidden), device=device, dtype=torch.bfloat16,
                        generator=generator,
                    ) * 0.1
                )
                source = (base.float() + rank * 0.01).to(torch.bfloat16)
                residual = torch.ones_like(source) * 0.25
                weight = torch.ones(hidden, device=device, dtype=torch.bfloat16)
                out = torch.empty_like(source)
                residual_out = torch.empty_like(source)
                epsilon = float(metadata.get("rms_norm_eps", 1e-5))
                context = _Collective(
                    runtime, source, base, residual, weight, epsilon, tp, fused
                )
                source_initial = source.clone()
                residual_initial = residual.clone()
                out_initial = out.clone()
                residual_out_initial = residual_out.clone()
                if fused:
                    def run():
                        return state.run_fused(
                            source, residual, weight, out, residual_out, epsilon
                        )
                else:
                    def run():
                        return state.run_plain(source, out)

                def reset():
                    source.copy_(source_initial)
                    residual.copy_(residual_initial)
                    out.copy_(out_initial)
                    residual_out.copy_(residual_out_initial)

                return PreparedCall(
                    run=run,
                    output=(out, residual_out) if fused else out,
                    reset=reset,
                    restore=reset,
                    owners=(context, owner),
                )

            request_name = (
                f"comm.{'fused_rmsnorm' if fused else 'allreduce'}.m{tokens}"
            )
            requests.append(declaration.request(
                name=request_name,
                prepare_call=collective_call,
                benchmark_call=collective_call,
                collective=CollectiveRequirement(
                    key=f"benchmark.{request_name}", ranks=(0, 1)
                ),
                retain_benchmark_call=True,
            ))
    return requests


def test_expected(call):
    import torch

    context = call.owners[0]
    expected = sum(
        (context.base.float() + rank * 0.01).to(torch.bfloat16).float()
        for rank in range(context.world_size)
    ).to(torch.bfloat16)
    if not context.fused:
        return expected
    residual = (expected.float() + context.residual.float()).to(torch.bfloat16)
    normalized = residual.float() * torch.rsqrt(
        residual.float().square().mean(-1, keepdim=True) + context.epsilon
    )
    return (normalized * context.weight.float()).to(torch.bfloat16), residual


def close_calls(calls):
    owners = {
        id(owner): owner
        for call in calls
        for owner in call.owners
        if isinstance(owner, _Owner)
    }
    for owner in owners.values():
        owner.close()
