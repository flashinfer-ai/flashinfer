"""Model-derived native delta-rule and MTP startup calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
import importlib

import torch

from b12x.preparation import PreparedCall, PreparationRequest
from .norm import _producer


@dataclass
class _Expected:
    recipe: str
    mode: str
    binding: object
    initial: torch.Tensor | None
    options: dict


def _lengths(m, max_seqs):
    count = min(m, max_seqs)
    quotient, remainder = divmod(m, count)
    return tuple(quotient + (index < remainder) for index in range(count))


def _decode_tensors(recipe, m, kh, vh, lengths, columns, device, *, initialize):
    n = len(lengths)

    def alloc(shape, dtype=torch.bfloat16, scale=0.1):
        value = torch.empty(shape, device=device, dtype=dtype)
        if initialize:
            value.normal_(0, scale)
        return value

    def integers(values, shape, dtype=torch.int32):
        if initialize:
            return torch.tensor(values, device=device, dtype=dtype).reshape(shape)
        return torch.empty(shape, device=device, dtype=dtype)

    args = dict(
        mixed_qkv=alloc((m, 2 * kh * 128 + vh * 128)),
        z=alloc((m, vh, 128)),
        A_log=alloc((vh,), torch.float32),
        dt_bias=alloc(
            (vh,) if recipe == "gdn" else (vh, 128),
            torch.bfloat16 if recipe == "gdn" else torch.float32,
        ),
        norm_weight=alloc((128,), torch.bfloat16),
        recurrent_state=alloc((n * columns + 1, vh, 128, 128), torch.float32),
        query_start_loc=integers([0, *accumulate(lengths)], (n + 1,)),
        num_accepted_tokens=integers([1] * n, (n,)),
        state_indices=integers(list(range(1, n * columns + 1)), (n, columns)),
        num_seqs=integers([n], (1,)),
        num_tokens=integers([m], (1,)),
        output=torch.empty((m, vh, 128), device=device, dtype=torch.bfloat16),
    )
    if initialize:
        args["norm_weight"].fill_(1)
    if recipe == "gdn":
        args.update(a=alloc((m, vh)), b=alloc((m, vh)))
    else:
        args.update(raw_g=alloc((m, vh, 128)), raw_beta=alloc((m, vh)))
    return args


def _scratch_for(state, device):
    values = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in state.layout.scratch_specs()
    )
    return values[0] if len(values) == 1 else values


def _decode_binding(state, recipe, args, device):
    bind = state.bind_kda if recipe == "kda" else state.bind
    return bind(scratch=_scratch_for(state, device), **args)


def _prefill_tensors(recipe, m, kh, vh, lengths, device, *, initialize):
    n = len(lengths)

    def alloc(shape, dtype=torch.bfloat16):
        value = torch.empty(shape, device=device, dtype=dtype)
        if initialize:
            value.normal_(0, 0.1)
        return value

    def integers(values, shape):
        if initialize:
            return torch.tensor(values, device=device, dtype=torch.int32).reshape(shape)
        return torch.empty(shape, device=device, dtype=torch.int32)

    return dict(
        q=alloc((m, kh, 128)),
        k=alloc((m, kh, 128)),
        v=alloc((m, vh, 128)),
        raw_g=alloc((m, vh) if recipe == "gdn" else (m, vh, 128)),
        raw_beta=alloc((m, vh)),
        A_log=alloc((vh,), torch.float32),
        dt_bias=alloc(
            (vh,) if recipe == "gdn" else (vh, 128),
            torch.bfloat16 if recipe == "gdn" else torch.float32,
        ),
        recurrent_state=alloc((3 * n + 1, vh, 128, 128), torch.float32),
        cu_seqlens=integers([0, *accumulate(lengths)], (n + 1,)),
        initial_state_indices=integers(list(range(1, n + 1)), (n,)),
        final_state_indices=integers(list(range(n + 1, 2 * n + 1)), (n,)),
        checkpoint_state_indices=integers(list(range(2 * n + 1, 3 * n + 1)), (n,)),
        checkpoint_offsets=integers([0] * n, (n,)),
        num_tokens=integers([m], (1,)),
        num_seqs=integers([n], (1,)),
        output=torch.empty((m, vh, 128), device=device, dtype=torch.bfloat16),
    )


def _prefill_binding(state, recipe, args, device):
    kwargs = dict(args)
    if recipe == "gdn":
        kwargs["a"] = kwargs.pop("raw_g")
        kwargs["b"] = kwargs.pop("raw_beta")
    return state.bind(scratch=_scratch_for(state, device), **kwargs)


def _fixture_call(state, *, recipe, mode, args, producer, producer_owners, initial_state, initial_output,
                  device, options):
    binding = (
        _decode_binding(state, recipe, args, device)
        if mode == "decode"
        else _prefill_binding(state, recipe, args, device)
    )

    def restore():
        binding.recurrent_state.copy_(initial_state)
        binding.output.copy_(initial_output)

    def produce():
        restore()
        producer()

    def run():
        if mode == "decode":
            return state.run(binding, **options)
        return state.run(binding, **options)

    return PreparedCall(
        run=run,
        output=binding.output,
        produce=produce,
        reset=restore,
        restore=restore,
        owners=(_Expected(recipe, mode, binding, initial_state, options), *producer_owners),
    )


def _delta_requests(metadata, device, rows):
    from b12x.sequence import gdn_decode as decode

    tp = int(metadata["_tp"])
    recipe = "gdn" if "linear_num_key_heads" in metadata else "kda"
    if recipe == "gdn":
        kh, vh = (
            int(metadata["linear_num_key_heads"]) // tp,
            int(metadata["linear_num_value_heads"]) // tp,
        )
        dims = (
            int(metadata["linear_key_head_dim"]),
            int(metadata["linear_value_head_dim"]),
        )
    else:
        kh = vh = int(metadata["linear_num_heads"]) // tp
        dims = (int(metadata["linear_head_dim"]),) * 2
    if dims != (128, 128):
        raise ValueError(
            "native delta-rule adapters require 128-dimensional model heads"
        )
    gate = (
        str(metadata.get("output_gate_type", "silu")) if recipe == "gdn" else "sigmoid"
    )
    max_seqs = int(metadata.get("_max_seqs", 1))
    spec = int(metadata.get("_spec_tokens", 3))
    eps = float(metadata.get("rms_norm_eps", 1e-6))
    lower_bound = float(metadata.get("linear_lower_bound", -5.0))
    prefill = importlib.import_module(f"b12x.sequence.{recipe}_prefill")
    requests = []
    for m in rows:
        lengths = _lengths(m, max_seqs)
        if max(lengths) <= spec + 1:
            columns = max(lengths)
            invocation_args = _decode_tensors(
                recipe,
                m,
                kh,
                vh,
                lengths,
                columns,
                torch.device("meta"),
                initialize=False,
            )
            caps = decode.Caps(
                device=device,
                max_tokens=m,
                max_seqs=len(lengths),
                max_state_slots=len(lengths) * columns + 1,
                key_heads=kh,
                value_heads=vh,
                state_index_columns=columns,
                gate_activation=gate,
                null_state_index=0 if recipe == "kda" else None,
            )
            plan = decode.plan(
                caps,
                invocation=decode.invocation_from_tensors(caps, **invocation_args),
            )
            options = {"eps": eps}
            if recipe == "kda":
                options["lower_bound"] = lower_bound
            session = {}

            def call(
                state,
                *,
                session=session,
                options=options,
                recipe=recipe,
                m=m,
                kh=kh,
                vh=vh,
                lengths=lengths,
                columns=columns,
                eps=eps,
            ):
                if not session:
                    args = _decode_tensors(
                        recipe, m, kh, vh, lengths, columns, device, initialize=True
                    )
                    producer, owners = _producer(args["mixed_qkv"], eps)
                    session.update(
                        args=args,
                        producer=producer,
                        producer_owners=owners,
                        initial_state=args["recurrent_state"].clone(),
                        initial_output=args["output"].clone(),
                    )
                return _fixture_call(
                    state,
                    recipe=recipe,
                    mode="decode",
                    args=session["args"],
                    producer=session["producer"],
                    producer_owners=session["producer_owners"],
                    initial_state=session["initial_state"],
                    initial_output=session["initial_output"],
                    device=device,
                    options=options,
                )

            name = f"sequence.{recipe}.decode.m{m}"
            requests.append(plan.request(
                name=name,
                prepare_call=call,
                benchmark_call=call,
                retain_benchmark_call=True,
            ))
        geometry = (
            {"key_heads": kh, "value_heads": vh} if recipe == "gdn" else {"heads": vh}
        )
        invocation_args = _prefill_tensors(
            recipe, m, kh, vh, lengths, torch.device("meta"), initialize=False
        )
        caps = prefill.Caps(
            device=device,
            max_tokens=m,
            max_seqs=len(lengths),
            max_state_slots=3 * len(lengths) + 1,
            null_state_index=0 if recipe == "kda" else None,
            checkpoint_export=False,
            qk_l2norm=True,
            **geometry,
        )
        plan = prefill.plan(
            caps,
            invocation=prefill.invocation_from_tensors(**invocation_args),
        )
        options = {"lower_bound": lower_bound} if recipe == "kda" else {}
        session = {}

        def call(
            state,
            *,
            session=session,
            options=options,
            recipe=recipe,
            m=m,
            kh=kh,
            vh=vh,
            lengths=lengths,
            eps=eps,
        ):
            if not session:
                args = _prefill_tensors(
                    recipe, m, kh, vh, lengths, device, initialize=True
                )
                producer, owners = _producer(args["q"], eps)
                session.update(
                    args=args,
                    producer=producer,
                    producer_owners=owners,
                    initial_state=args["recurrent_state"].clone(),
                    initial_output=args["output"].clone(),
                )
            return _fixture_call(
                state,
                recipe=recipe,
                mode="prefill",
                args=session["args"],
                producer=session["producer"],
                producer_owners=session["producer_owners"],
                initial_state=session["initial_state"],
                initial_output=session["initial_output"],
                device=device,
                options=options,
            )

        name = f"sequence.{recipe}.prefill.m{m}"
        requests.append(plan.request(
            name=name,
            prepare_call=call,
            benchmark_call=call,
            retain_benchmark_call=True,
        ))
    return requests


def _feedback_requests(metadata, device, rows):
    from b12x.sequence import mtp_feedback as op

    h, s = int(metadata["hidden_size"]), int(metadata["hc_count"])
    eps = float(metadata.get("rms_norm_eps", 1e-6))
    weights = dict(
        token_norm_weight=torch.ones(h, device=device, dtype=torch.bfloat16),
        state_norm_weight=torch.ones(s * h, device=device, dtype=torch.bfloat16),
        embedding_fc_weight=torch.randn(
            (h, h), device=device, dtype=torch.bfloat16
        ).div_(h**0.5),
        hidden_fc_weight=torch.randn(
            (h, h), device=device, dtype=torch.bfloat16
        ).div_(h**0.5),
    )
    requests = []
    for m in rows:
        invocation_inputs = dict(
            token_embedding=torch.empty((m, h), device="meta", dtype=torch.bfloat16),
            multi_state=torch.empty((m, s, h), device="meta", dtype=torch.bfloat16),
            output=torch.empty((m, s, h), device="meta", dtype=torch.bfloat16),
        )
        plan = op.plan(
            op.Caps(device=device, max_tokens=m, hidden_size=h, streams=s),
            invocation=op.invocation_from_tensors(**invocation_inputs, **weights),
        )
        session = {}

        def call(state, *, session=session, m=m, h=h, s=s, eps=eps):
            if not session:
                inputs = dict(
                    token_embedding=torch.empty(
                        (m, h), device=device, dtype=torch.bfloat16
                    ),
                    multi_state=torch.randn(
                        (m, s, h), device=device, dtype=torch.bfloat16
                    ),
                    output=torch.empty((m, s, h), device=device, dtype=torch.bfloat16),
                )
                producer, owners = _producer(inputs["token_embedding"], eps)
                session.update(
                    inputs=inputs,
                    producer=producer,
                    producer_owners=owners,
                    initial_output=inputs["output"].clone(),
                )
            binding = state.bind(
                scratch=_scratch_for(state, device),
                **weights,
                **session["inputs"],
            )

            def restore():
                binding.output.copy_(session["initial_output"])

            def produce():
                restore()
                session["producer"]()

            return PreparedCall(
                run=lambda: state.run(binding, eps=eps),
                output=binding.output,
                produce=produce,
                reset=restore,
                restore=restore,
                owners=(
                    _Expected("mtp", "feedback", binding, None, {"eps": eps}),
                    *session["producer_owners"],
                ),
            )
        name = f"sequence.mtp.feedback.m{m}"
        requests.append(plan.request(
            name=name,
            prepare_call=call,
            benchmark_call=call,
            retain_benchmark_call=True,
        ))
    return requests


def make_benchmark_requests(
    metadata: Mapping, *, device: torch.device, rows: tuple[int, ...]
) -> list[PreparationRequest]:
    requests = _delta_requests(metadata, device, rows)
    if "hc_count" in metadata and metadata.get("mtp_num_hidden_layers", 0):
        requests.extend(_feedback_requests(metadata, device, rows))
    return requests


def test_expected(call: PreparedCall):
    context = next(owner for owner in call.owners if isinstance(owner, _Expected))
    b = context.binding
    if context.recipe == "mtp":
        from b12x.sequence.mtp_feedback.reference import feedback

        return feedback(
            b.token_embedding,
            b.multi_state,
            b.token_norm_weight,
            b.state_norm_weight,
            b.embedding_fc_weight,
            b.hidden_fc_weight,
            **context.options,
        )
    if context.mode == "decode":
        from b12x.sequence.gdn_decode import reference

        names = ("a", "b") if context.recipe == "gdn" else ("raw_g", "raw_beta")
        args = [
            b.mixed_qkv,
            *[getattr(b, k) for k in names],
            b.z,
            b.A_log,
            b.dt_bias,
            b.norm_weight,
            context.initial.clone(),
            b.query_start_loc,
            b.num_accepted_tokens,
            b.state_indices,
            b.num_seqs,
            b.num_tokens,
        ]
        options = dict(
            context.options,
            qk_l2norm=True,
            null_state_index=b.plan.caps.null_state_index,
        )
        if context.recipe == "gdn":
            return reference.decode(
                *args,
                key_heads=b.plan.caps.key_heads,
                value_heads=b.plan.caps.value_heads,
                gate_activation=b.plan.caps.gate_activation,
                **options,
            )
        return reference.decode_kda(*args, heads=b.plan.caps.value_heads, **options)
    op = importlib.import_module(f"b12x.sequence.{context.recipe}_prefill")
    fn = (
        op.reference.prefill_gdn
        if context.recipe == "gdn"
        else op.reference.prefill_kda
    )
    out = torch.empty_like(b.output)
    fn(
        b.q,
        b.k,
        b.v,
        b.raw_g,
        b.raw_beta,
        b.A_log,
        b.dt_bias,
        context.initial.clone(),
        b.cu_seqlens,
        b.initial_state_indices,
        b.final_state_indices,
        b.checkpoint_state_indices,
        b.checkpoint_offsets,
        b.num_seqs,
        b.num_tokens,
        output=out,
        qk_l2norm=True,
        null_state_index=b.plan.caps.null_state_index,
        **context.options,
    )
    return out
