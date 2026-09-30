"""PLE startup calls using full checkpoint table geometry and bounded disk I/O."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
import json
from pathlib import Path
import struct

import torch

from b12x.preparation import FrozenMapping, PreparedCall, PreparationRequest
from .norm import _producer, _scratch
from .sequence import _lengths


class _Checkpoint:
    """Read safetensors headers and small geometry tensors, never whole shards."""

    def __init__(self, directory):
        self.directory = Path(directory)
        index = self.directory / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(f"PLE requires reachable checkpoint index: {index}")
        self.index = json.loads(index.read_text())["weight_map"]
        self.headers = {}

    def location(self, key):
        path = self.directory / self.index[key]
        if path not in self.headers:
            with path.open("rb") as file:
                (length,) = struct.unpack("<Q", file.read(8))
                self.headers[path] = (8 + length, json.loads(file.read(length)))
        offset, header = self.headers[path]
        item = header[key]
        return path, offset + item["data_offsets"][0], item

    def small(self, key):
        path, offset, item = self.location(key)
        start, end = item["data_offsets"]
        if end - start > 65536:
            raise ValueError(f"not a small PLE geometry tensor: {key}")
        dtype = {"I64": torch.int64, "F32": torch.float32}[item["dtype"]]
        with path.open("rb") as file:
            file.seek(offset)
            payload = bytearray(file.read(end - start))
        return torch.frombuffer(payload, dtype=dtype).reshape(item["shape"]).clone()


@dataclass
class _Expected:
    role: str
    binding: object
    checkpoint: _Checkpoint | None = None
    prefix: str = ""
    initial: torch.Tensor | None = None
    eps: float = 1e-6


def _close_disk(disk):
    # This adapter created the mapped staging allocations; supplied TableStorage
    # remains caller-owned. Explicit release precedes Python module teardown.
    disk._weight_allocation.close()
    if disk._scale_allocation is not None:
        disk._scale_allocation.close()
    disk._reader = None


def _hash_inputs(m, lengths, order, vocab, eos, device):
    return dict(
        token_ids=torch.arange(1, m + 1, device=device, dtype=torch.int64).remainder_(
            vocab
        ),
        query_start_loc=torch.tensor(
            [0, *accumulate(lengths)], device=device, dtype=torch.int32
        ),
        committed_history=torch.full(
            (len(lengths), order - 1), eos, device=device, dtype=torch.int64
        ),
        num_seqs=torch.tensor([len(lengths)], device=device, dtype=torch.int32),
        num_tokens=torch.tensor([m], device=device, dtype=torch.int32),
    )


def _geometry(checkpoint, prefix):
    return dict(
        multipliers=checkpoint.small(prefix + "layer_multipliers"),
        prime_sizes=checkpoint.small(prefix + "ngram_heads_vocab_sizes"),
        table_offsets=checkpoint.small(prefix + "ngram_heads_offsets"),
    )



def _embedding_requests(metadata, device, rows):
    from b12x.sequence import ple_embedding, ple_hash

    memory = str(metadata.get("_ple_table_memory", "io_uring"))
    quant = {
        "nvfp4": "nvfp4_group16",
        "nvfp4_group16": "nvfp4_group16",
        "float4_e2m1fn_x2": "nvfp4_group16",
        "bfloat16": "bf16",
        "float8_e4m3fn": "fp8_e4m3_per_tensor",
    }[metadata.get("ple_embedding_dtype", "bfloat16")]
    model_path = metadata.get(
        "_checkpoint_path", metadata.get("_model_path", metadata["_model_id"])
    )
    checkpoint = _Checkpoint(model_path)
    tp, rank = int(metadata["_tp"]), int(metadata.get("_tp_rank", 0))
    vocab, order, heads, base, dim = (
        int(metadata[key])
        for key in (
            "vocab_size", "ngram_size", "heads_per_ngram",
            "ngram_vocab_size_base", "ple_embed_dim",
        )
    )
    eos = int(metadata["eos_token_id"])
    align = int(metadata.get("make_ngram_vocab_size_divisible_by", 128))
    requests = []
    for ordinal, layer in enumerate(sorted(set(metadata["ple_layer_ids"]))):
        suffix = f"layers.{int(layer) - 1}.ple.ple_embedding.layer_multipliers"
        key = next((key for key in checkpoint.index if key.endswith(suffix)), None)
        if key is None:
            raise KeyError(f"PLE checkpoint is missing layer geometry: {suffix}")
        prefix = key.removesuffix("layer_multipliers")
        geometry_tensors = _geometry(checkpoint, prefix)
        for m in rows:
            lengths = _lengths(m, int(metadata.get("_max_seqs", 1)))
            common = dict(
                device=device, max_tokens=m, max_seqs=len(lengths),
                vocab_size=vocab, eos_token_id=eos, max_order=order,
                heads_per_order=heads, dense_layer_ordinal=ordinal,
                base_table_size=base, table_alignment=align,
            )
            hash_caps = ple_hash.Caps(**common)
            geometry = ple_hash.compute_geometry(hash_caps, **geometry_tensors)
            hash_plan = ple_hash.plan(
                hash_caps, geometry=geometry, **geometry_tensors, invocation=FrozenMapping(),
            )
            embedding_caps = ple_embedding.Caps(
                **common, embedding_dim=dim, tp_size=tp, tp_rank=rank,
                quant_mode=quant, table_memory=memory,
            )
            embedding_plan = ple_embedding.plan(
                embedding_caps, geometry=geometry, **geometry_tensors, invocation=FrozenMapping(),
            )

            def hash_call(state, *, m=m, lengths=lengths):
                args = _hash_inputs(m, lengths, order, vocab, eos, device)
                binding = state.bind(
                    scratch=_scratch(state.layout, device), **args,
                    out=torch.empty(
                        (m, (order - 1) * heads), device=device, dtype=torch.int64
                    ),
                )
                return PreparedCall(
                    run=lambda: state.run(binding),
                    output=binding.out,
                    owners=(_Expected("hash", binding),),
                )

            requests.append(hash_plan.request(
                name=f"ple.hash.layer{layer}.m{m}",
                prepare_call=hash_call,
                benchmark_call=hash_call,
                retain_benchmark_call=True,
            ))

            def embedding_call(
                state, *, m=m, lengths=lengths, prefix=prefix, ordinal=ordinal,
            ):
                args = _hash_inputs(m, lengths, order, vocab, eos, device)
                if memory == "io_uring":
                    _, _, first = checkpoint.location(
                        prefix + "ngram_embedding.shard_0.weight"
                    )
                    shard_rows = int(first["shape"][0])
                    disk = ple_embedding.DiskTable(state.layout, shard_rows)
                    first_shard = state.layout.shard_start // shard_rows
                    last_shard = (state.layout.shard_end + shard_rows - 1) // shard_rows
                    for shard in range(first_shard, last_shard):
                        for scale in ((False, True) if quant == "nvfp4_group16" else (False,)):
                            name = f"{prefix}ngram_embedding.shard_{shard}.weight" + (
                                "_scale" if scale else ""
                            )
                            path, offset, item = checkpoint.location(name)
                            expected_cols = (
                                state.layout.head_dim // 16
                                if scale else state.layout.weight_shape[1]
                            )
                            if item["shape"][1] != expected_cols:
                                raise ValueError(
                                    "PLE checkpoint row width differs from production "
                                    f"plan: {name}"
                                )
                            disk.add_shard(shard, str(path), offset, scale=scale)
                    scale2 = (
                        checkpoint.small(prefix + "ngram_embedding.weight_scale_2").to(device)
                        if quant == "nvfp4_group16" else None
                    )
                    if quant == "fp8_e4m3_per_tensor":
                        raise ValueError(
                            "FP8 disk PLE requires checkpoint BF16 per-table "
                            "weight_scale loading"
                        )
                    storage, owners = (
                        dict(weight=None, weight_scale_2=scale2, disk_table=disk),
                        (disk,),
                    )
                else:
                    supplied = metadata.get("_ple_storage", {})
                    if ordinal not in supplied:
                        raise ValueError(
                            f"PLE {memory} requires caller-loaded full TableStorage "
                            f"for ordinal {ordinal}; supply metadata[_ple_storage][ordinal] "
                            "or declare _ple_table_memory=io_uring"
                        )
                    table = supplied[ordinal]
                    storage, owners = (
                        dict(
                            weight=table.weight, weight_scale=table.weight_scale,
                            weight_scale_2=table.weight_scale_2,
                        ),
                        (table,),
                    )
                binding = state.bind(
                    scratch=_scratch(state.layout, device), **args, **storage,
                    out=torch.empty(
                        state.layout.output_shape, device=device,
                        dtype=state.layout.output_dtype,
                    ),
                )
                return PreparedCall(
                    run=lambda: state.run(binding, token_count=m),
                    output=binding.out,
                    owners=(_Expected("embedding", binding, checkpoint, prefix), *owners),
                    close=(lambda: _close_disk(disk)) if memory == "io_uring" else None,
                    capture_safe=memory != "io_uring",
                )

            requests.append(embedding_plan.request(
                name=f"ple.embedding.layer{layer}.m{m}",
                prepare_call=embedding_call,
                benchmark_call=embedding_call,
                retain_benchmark_call=True,
            ))
    return requests


def _convolution_requests(metadata, device, rows):
    from b12x.sequence import ple as op

    h, s, k, dilation = (
        int(metadata[key])
        for key in ("hidden_size", "hc_count", "ple_conv_kernel_size", "ngram_size")
    )
    eps = float(metadata.get("rms_norm_eps", 1e-6))
    spec = int(metadata.get("_spec_tokens", 3))
    requests = []
    for m in rows:
        lengths = _lengths(m, int(metadata.get("_max_seqs", 1)))
        n = len(lengths)
        declaration = op.plan(op.Caps(
            device=device, mode="mixed", max_tokens=m, max_seqs=n,
            max_state_slots=n, max_speculative_tokens=spec, streams=s,
            hidden_size=h, kernel_size=k, dilation=dilation,
        ))

        def convolution_call(state, *, m=m, n=n, lengths=lengths):
            def rand(shape):
                return torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1

            args = dict(
                residual=rand((m, s, h)), key=rand((m, s, h)), value=rand((m, h)),
                k_norm_weight=rand((s * h,)), q_norm_weight=rand((s * h,)),
                u_norm_weight=rand((s * h,)), conv_weight=rand((s * h, k)),
                query_start_loc=torch.tensor(
                    [0, *accumulate(lengths)], device=device, dtype=torch.int32
                ),
                state_slot_ids=torch.arange(n, device=device, dtype=torch.int64),
                state_is_fresh=torch.zeros(n, device=device, dtype=torch.bool),
                num_accepted_tokens=torch.ones(n, device=device, dtype=torch.int32),
                num_seqs=torch.tensor([n], device=device, dtype=torch.int32),
                num_tokens=torch.tensor([m], device=device, dtype=torch.int32),
                conv_state=rand((n, s * h, state.state_capacity)),
                request_is_prefill=torch.tensor(
                    [length > spec + 1 for length in lengths],
                    device=device, dtype=torch.bool,
                ),
                out=torch.empty((m, s, h), device=device, dtype=torch.bfloat16),
            )
            initial = args["conv_state"].clone()
            producer, owners = _producer(args["residual"], eps)
            binding = state.bind(scratch=_scratch(state.layout, device), **args)

            def reset():
                binding.conv_state.copy_(initial)

            def produce():
                reset()
                producer()

            return PreparedCall(
                run=lambda: state.run(binding, eps=eps, token_count=m),
                output=binding.out, produce=produce, reset=reset, restore=reset,
                owners=(_Expected("convolution", binding, initial=initial, eps=eps), owners),
            )

        requests.append(declaration.request(
            name=f"ple.convolution.mixed.m{m}",
            prepare_call=convolution_call,
            benchmark_call=convolution_call,
            retain_benchmark_call=True,
        ))
    return requests


def make_benchmark_requests(
    metadata: Mapping, *, device: torch.device, rows: tuple[int, ...]
) -> list[PreparationRequest]:
    if not metadata.get("ple_layer_ids"):
        return []
    return _embedding_requests(metadata, device, rows) + _convolution_requests(
        metadata, device, rows
    )


def _expected_ids(binding):
    from b12x.sequence.ple_hash.reference import ple_hash_packed_reference

    hash_binding = getattr(binding, "_hash_binding", binding)
    layout = hash_binding._state
    geometry = hash_binding.geometry
    return ple_hash_packed_reference(
        hash_binding.token_ids,
        hash_binding.query_start_loc,
        hash_binding.committed_history,
        eos_token_id=layout.caps.eos_token_id,
        multipliers=geometry.multipliers,
        prime_sizes=geometry.prime_sizes,
        table_offsets=geometry.table_offsets,
        heads_per_order=layout.caps.heads_per_order,
    )


def test_expected(call: PreparedCall):
    """Independent selected-row oracle; no whole-table materialization."""
    context = next(owner for owner in call.owners if isinstance(owner, _Expected))
    b = context.binding
    if context.role == "hash":
        return _expected_ids(b)
    if context.role == "convolution":
        from b12x.sequence.ple.reference import ple_projected_packed_reference

        return ple_projected_packed_reference(
            b.residual,
            b.key,
            b.value,
            b.query_start_loc,
            **{
                key: getattr(b, key)
                for key in (
                    "k_norm_weight", "q_norm_weight", "u_norm_weight", "conv_weight",
                )
            },
            eps=context.eps,
            dilation=b._state.caps.dilation,
            prior_states=context.initial[:, :, : b._state.caps.state_length].contiguous(),
        )[0]
    ids = _expected_ids(b).cpu()
    layout = b._state
    if b.disk_table is None:
        from b12x.sequence.ple_embedding.reference import lookup

        return lookup(
            b.weight,
            b.weight_scale,
            ids.to(b.out.device),
            weight_scale_2=b.weight_scale_2,
            quant_mode=layout.caps.quant_mode,
            shard_start=layout.shard_start,
            embedding_dim=layout.caps.embedding_dim,
            output_dtype=b.out.dtype,
        )
    if layout.caps.quant_mode != "nvfp4_group16":
        raise ValueError(
            "selected-row startup oracle currently requires the target NVFP4 checkpoint"
        )
    from b12x.sequence.ple_embedding.reference import _dequantize_selected_nvfp4

    disk, checkpoint = b.disk_table, context.checkpoint
    packed = torch.zeros((*ids.shape, layout.head_dim // 2), dtype=torch.uint8)
    scales = torch.zeros(
        (*ids.shape, layout.head_dim // 16), dtype=torch.float8_e4m3fn
    )
    for row in range(ids.shape[0]):
        for head in range(ids.shape[1]):
            index = int(ids[row, head])
            if not layout.shard_start <= index < layout.shard_end:
                continue
            shard, local = divmod(index, disk.shard_rows)
            for is_scale, target in ((False, packed), (True, scales)):
                key = f"{context.prefix}ngram_embedding.shard_{shard}.weight" + (
                    "_scale" if is_scale else ""
                )
                path, offset, item = checkpoint.location(key)
                width = item["shape"][1]
                with path.open("rb") as file:
                    file.seek(offset + local * width)
                    value = torch.frombuffer(
                        bytearray(file.read(width)), dtype=target.dtype
                    )
                target[row, head].copy_(value)
    decoded = _dequantize_selected_nvfp4(
        packed, scales, b.weight_scale_2.cpu(), head_dim=layout.head_dim
    )
    return decoded.flatten(1).to(device=b.out.device, dtype=b.out.dtype)
