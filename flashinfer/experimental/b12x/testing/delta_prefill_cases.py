"""Reproducible production-plan cases for GDN and KDA prefill qualification."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate

import torch

from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
from b12x.sequence._shared.delta_prefill.preparation import invocation_from_tensors


@dataclass(frozen=True)
class PrefillCase:
    recipe: str
    key_heads: int
    value_heads: int
    lengths: tuple[int, ...]

    @property
    def name(self) -> str:
        lengths = "x".join(map(str, self.lengths))
        return f"{self.recipe}-qk{self.key_heads}-v{self.value_heads}-t{lengths}"

    @property
    def tokens(self) -> int:
        return sum(self.lengths)


PREFILL_LENGTHS = (
    (16,), (64,), (256,), (1024,), (4096,), (8192,), (32768,),
    (1024,) * 4, (512,) * 8, (4096, 1024, 127, 1),
)
GDN_PREFILL_CASES = tuple(
    PrefillCase("gdn", key, 3 * key, lengths)
    for key in (2, 4, 8, 16) for lengths in PREFILL_LENGTHS
)
KDA_PREFILL_CASES = tuple(
    PrefillCase("kda", heads, heads, lengths)
    for heads in (16, 32, 64) for lengths in PREFILL_LENGTHS
)


def prefill_op(recipe: str):
    import importlib
    if recipe not in ("gdn", "kda"):
        raise ValueError(f"unknown prefill recipe {recipe!r}")
    return importlib.import_module(f"b12x.sequence.{recipe}_prefill")


def make_inputs(case: PrefillCase, *, device, seed=20260827, max_tokens=None, max_seqs=None):
    """Create identical values on any device, including disjoint source/destination slots."""
    token_capacity = max(1, case.tokens) if max_tokens is None else int(max_tokens)
    seq_capacity = max(1, len(case.lengths)) if max_seqs is None else int(max_seqs)
    if token_capacity < case.tokens or seq_capacity < len(case.lengths):
        raise ValueError("live prefill shape exceeds capacity")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape, dtype=torch.bfloat16, scale=0.25):
        return (torch.randn(shape, generator=generator) * scale).to(dtype=dtype, device=device)

    kh, vh = case.key_heads, case.value_heads
    raw_shape = (token_capacity, vh) if case.recipe == "gdn" else (token_capacity, vh, 128)
    bias_shape = (vh,) if case.recipe == "gdn" else (vh, 128)
    cu = [0, *accumulate(case.lengths)] + [case.tokens] * (seq_capacity-len(case.lengths))
    return {
        "q": rand((token_capacity, kh, 128)), "k": rand((token_capacity, kh, 128)),
        "v": rand((token_capacity, vh, 128)), "raw_g": rand(raw_shape, scale=1),
        "raw_beta": rand((token_capacity, vh), scale=1),
        "A_log": rand((vh,), torch.float32, 0.1), "dt_bias": rand(bias_shape, torch.float32, 0.1),
        "recurrent_state": rand((3 * seq_capacity + 1, vh, 128, 128), torch.float32, 0.1),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "initial_state_indices": torch.arange(seq_capacity, dtype=torch.int32, device=device),
        "final_state_indices": torch.arange(seq_capacity, 2*seq_capacity, dtype=torch.int32, device=device),
        "checkpoint_state_indices": torch.arange(2*seq_capacity, 3*seq_capacity, dtype=torch.int32, device=device),
        "checkpoint_offsets": torch.zeros(seq_capacity, dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([case.tokens], dtype=torch.int32, device=device),
        "num_seqs": torch.tensor([len(case.lengths)], dtype=torch.int32, device=device),
        "output": torch.full((token_capacity, vh, 128), float("nan"), dtype=torch.bfloat16, device=device),
    }


def _binding_args(case: PrefillCase, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    args = dict(tensors)
    if case.recipe == "gdn":
        args["a"] = args.pop("raw_g")
        args["b"] = args.pop("raw_beta")
    return args


@contextmanager
def prepared_binding(
    case: PrefillCase,
    tensors: dict[str, torch.Tensor],
    *,
    max_tokens=None,
    max_seqs=None,
    checkpoint_export=False,
    null_state_index=None,
    qk_l2norm=True,
    config=None,
):
    """Prepare a real prefill plan and bind it only for this caller scope.

    The priming call restores the caller's pool and output before this yields;
    the binding uses separately allocated serving scratch and remains valid
    until the scope releases its result/session.
    """
    op = prefill_op(case.recipe)
    device = tensors["q"].device
    geometry = ({"key_heads": case.key_heads, "value_heads": case.value_heads}
                if case.recipe == "gdn" else {"heads": case.value_heads})
    caps = op.Caps(
        device=device, max_tokens=tensors["q"].shape[0] if max_tokens is None else max_tokens,
        max_seqs=tensors["cu_seqlens"].numel() - 1 if max_seqs is None else max_seqs,
        max_state_slots=tensors["recurrent_state"].shape[0], checkpoint_export=checkpoint_export,
        null_state_index=null_state_index, qk_l2norm=qk_l2norm,
        **geometry,
    )
    override = None
    if config is not None:
        config_type = op.GdnPrefillConfig if case.recipe == "gdn" else op.KdaPrefillConfig
        override = config_type.from_config(FrozenMapping(config))
    plan = op.plan(caps, invocation=FrozenMapping(invocation_from_tensors(**tensors)), override=override)
    args = _binding_args(case, tensors)

    def prepare_call(state):
        spec, = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(scratch=scratch, **args)
        original_state = binding.recurrent_state.clone()
        original_output = binding.output.clone()

        def reset():
            binding.recurrent_state.copy_(original_state)
            binding.output.copy_(original_output)

        def run():
            if case.recipe == "kda":
                return state.run(binding, lower_bound=-5.0)
            return state.run(binding)

        return PreparedCall(run=run, output=binding.output, reset=reset, restore=reset)

    request = plan.request(
        name=f"{case.name}-{id(tensors)}", prepare_call=prepare_call,
    )
    with PreparationSession(device=device, autotune=False) as session:
        session.prepare((request,))
        state = plan.prepared.state
        spec, = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = op.bind(plan, scratch=scratch, **args)
        yield binding




def run_binding(recipe, binding):
    op = prefill_op(recipe)
    if recipe == "kda":
        return op.run(binding, lower_bound=-5.0)
    return op.run(binding)


def oracle(case: PrefillCase, tensors, *, qk_l2norm=True, null_state_index=None):
    op = prefill_op(case.recipe)
    pool = tensors["recurrent_state"].clone()
    output = tensors["output"].clone()
    args = [tensors[name] for name in ("q", "k", "v", "raw_g", "raw_beta", "A_log", "dt_bias")]
    args.append(pool)
    args.extend(tensors[name] for name in (
        "cu_seqlens", "initial_state_indices", "final_state_indices", "checkpoint_state_indices",
        "checkpoint_offsets", "num_seqs", "num_tokens",
    ))
    fn = op.reference.prefill_gdn if case.recipe == "gdn" else op.reference.prefill_kda
    fn(*args, output=output, qk_l2norm=qk_l2norm, null_state_index=null_state_index)
    return output, pool


def assert_close(name, actual, expected, *, ratio):
    """Apply fixed RMS and peak-error gates, including finite/zero checks."""
    if actual.numel() == 0:
        return {"relative_rmse": 0.0, "max_abs": 0.0, "nonzero": 0}
    a, r = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all()):
        raise AssertionError(f"{name}: non-finite values")
    delta = (a-r).abs()
    rms = r.square().mean().sqrt().item()
    error = delta.square().mean().sqrt().item() / (rms+1e-8)
    peak = delta.max().item()
    if peak > 1e-6:
        if error >= ratio or peak > 4e-2*rms + 2**-6*r.abs().max().item():
            raise AssertionError(f"{name}: relative RMSE {error:.6g} (limit {ratio}), peak {peak:.6g}")
    nonzero = int(torch.count_nonzero(a))
    if nonzero == 0 and bool(torch.count_nonzero(r)):
        raise AssertionError(f"{name}: nonzero reference produced zero output")
    return {"relative_rmse": error, "max_abs": peak, "nonzero": nonzero}


def check_binding(case, binding, expected_output, expected_state, initial_state):
    count = int(binding.num_tokens.item())
    seqs = int(binding.num_seqs.item())
    metrics = {"output": assert_close("output", binding.output[:count], expected_output[:count], ratio=1e-2)}
    caps = binding._state.caps
    null = caps.null_state_index
    writes = {int(i) for i in binding.final_state_indices[:seqs].tolist() if int(i) != null}
    if caps.checkpoint_export:
        writes.update(
            int(index)
            for index, offset in zip(
                binding.checkpoint_state_indices[:seqs].tolist(),
                binding.checkpoint_offsets[:seqs].tolist(),
            )
            if offset > 0 and int(index) != null
        )
    metrics["states"] = {
        str(slot): assert_close(f"state[{slot}]", binding.recurrent_state[slot], expected_state[slot], ratio=5e-3)
        for slot in sorted(writes)
    }
    untouched = [s for s in range(binding.recurrent_state.shape[0]) if s not in writes]
    torch.testing.assert_close(binding.recurrent_state[untouched], initial_state[untouched], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(binding.output[count:], expected_output[count:], rtol=0, atol=0, equal_nan=True)
    return metrics
