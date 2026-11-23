# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Preparation and capture-safe execution of the two collective operator families."""

from __future__ import annotations

import torch
from dataclasses import fields, is_dataclass

from . import _descriptors as descriptors
from .geometry import (
    ATTENTION_ARENA_BYTES,
    ATTENTION_Q1_COUNTER_OFFSET,
    ATTENTION_Q4_HIDDEN_RX_BYTES,
    ATTENTION_Q4_HIDDEN_RX_OFFSET,
    CACHE_PAD,
    HIDDEN_SIZE,
    MOE_DOWN_SCALE_SHAPE,
    MOE_DOWN_SHAPE,
    MOE_Q1_COUNTER_OFFSET,
    MOE_Q4_COUNTER_OFFSET,
    MOE_GATE_UP_SHAPE,
    Q1_RECORD_BYTES,
    Q4_RECORD_BYTES,
    SELECTED_TOKENS,
    WORLD_SIZE,
    check_q_len,
)
from .weights import MoEWeights
from .workspace import DecodeWorkspace

# Every launch is cooperative: the persistent peer-waiting grid must be fully
# resident, so the driver rejects a launch that cannot be instead of starting a
# partial grid that would wait forever.
_COOPERATIVE_LAUNCH = True
# Tag of the selected-ID records that rank zero publishes; it must equal the
# ll_tag that the attention binding gives the attention ranks.
_ID_RECORD_TAG = 1


def _tensor(name, value, dtype, device, shape=None, *, elements=None, alignment=16):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must have dtype {dtype} on {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}"
        )
    if elements is not None and value.numel() != elements:
        raise ValueError(f"{name} must contain {elements} elements")
    if value.numel() and value.data_ptr() % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")
    if value.requires_grad:
        raise ValueError(f"{name}: these inference operators do not support autograd")


def _disjoint(output, inputs):
    begin = output.data_ptr()
    end = begin + output.numel() * output.element_size()
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor) or not value.numel():
            continue
        other = value.data_ptr()
        if begin < other + value.numel() * value.element_size() and other < end:
            raise ValueError(f"out must not overlap {name}")


def _workspace(value, family):
    if not isinstance(value, DecodeWorkspace) or value._destroyed:
        raise ValueError("workspace must be a live GLM-5 TP8 workspace")
    if family not in value.operators:
        raise ValueError(f"workspace was not created for {family}")
    with torch.cuda.device(value.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare must run outside CUDA graph capture; capture the prepared plan"
            )


def _agree(values):
    if any(value != values[0] for value in values):
        raise ValueError(
            "all ranks must prepare the same operator mode and query length"
        )


class _Plan:
    def __init__(
        self,
        workspace,
        out,
        module,
        tensors,
        owned,
        owners,
        diagnostics,
        *,
        family,
        q_len,
        indexer="full",
        enable_pdl=True,
    ):
        self.workspace = workspace
        self.out = out
        self._module = module
        self._tensors = tuple(tensors)
        self._owned_tensors = list(owned)
        self._owners = owners

        # Descriptors embed pointers that are absent from the FFI tensor array.
        # Track every caller-owned allocation until its launch stream is done,
        # even when a temporary plan is dropped immediately after an eager call.
        def tensors_in(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from tensors_in(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    yield from tensors_in(child)
            elif is_dataclass(value):
                for field in fields(value):
                    yield from tensors_in(getattr(value, field.name))

        self._stream_tensors = tuple(
            {id(t): t for t in tensors_in((owners, owned))}.values()
        )
        self._diagnostics = diagnostics
        self._family = family
        self._q_len = q_len
        self._indexer = indexer
        self._enable_pdl = enable_pdl
        self._valid = True
        workspace._plans.add(self)

    def _invalidate(self):
        self._valid = False
        self._tensors = ()
        self._owned_tensors.clear()
        self._owners = None
        self._stream_tensors = ()
        self._diagnostics.clear()

    def __call__(self):
        ws = self.workspace
        if not self._valid or ws._destroyed:
            raise RuntimeError("plan's workspace has been destroyed")
        with torch.cuda.device(ws.device):
            ws.order_after_previous_stream()
            if self._family == "moe":
                self._module.run(
                    self._tensors,
                    ws.region(f"moe_q{self._q_len}"),
                    ws.rank,
                    self._enable_pdl,
                    _COOPERATIVE_LAUNCH,
                )
            else:
                self._module.run(
                    self._tensors,
                    ws.region("attention"),
                    ws.region("attention_q1"),
                    ws.rank,
                    self._indexer == "shared",
                    self._enable_pdl,
                    _COOPERATIVE_LAUNCH,
                )
            if not torch.cuda.is_current_stream_capturing():
                stream = torch.cuda.current_stream(ws.device)
                for tensor in self._stream_tensors:
                    tensor.record_stream(stream)
        return self.out

    run = __call__


def prepare_moe(x, weights, *, out=None, workspace, enable_pdl=True):
    _workspace(workspace, "moe")
    ws = workspace
    result = {}

    def validate():
        if not isinstance(weights, MoEWeights):
            raise TypeError("weights must be returned by pack_glm5_tp8_moe_weights")
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must have shape [q_len,6144]")
        q = check_q_len(x.shape[0])
        _tensor("x", x, torch.bfloat16, ws.device, (q, HIDDEN_SIZE))
        output = torch.empty_like(x) if out is None else out
        _tensor("out", output, torch.bfloat16, ws.device, x.shape)
        _tensor("gamma", weights.gamma, torch.float32, ws.device, (6144,))
        _tensor("router", weights.router, torch.bfloat16, ws.device, (256, 6144))
        _tensor("bias", weights.bias, torch.float32, ws.device, (256,))
        _tensor("gate_up", weights.gate_up, torch.uint8, ws.device, MOE_GATE_UP_SHAPE)
        _tensor("down", weights.down, torch.uint8, ws.device, MOE_DOWN_SHAPE)
        _tensor(
            "down_scale",
            weights.down_scale,
            torch.float32,
            ws.device,
            MOE_DOWN_SCALE_SHAPE,
        )
        _disjoint(output, {"x": x, **vars(weights), **ws._moe})
        result.update(q=q, out=output)
        return q, bool(enable_pdl)

    _agree(ws.collective_check(validate))
    q, output = result["q"], result["out"]
    s = ws._moe
    owned = []

    def prepare():
        with torch.cuda.device(ws.device):
            if q == 1:
                context = descriptors.prepare_q1_moe(
                    ws.device,
                    x=x,
                    gamma=weights.gamma,
                    router_w=weights.router,
                    gate_up_w=weights.gate_up,
                    bias=weights.bias,
                    down_w=weights.down,
                    down_scale=weights.down_scale,
                    scores=s["scores"],
                    hidden_mid=s["mid"],
                    x_norm=s["norm"],
                    hidden_out=output,
                    sel_probs=s["probs"],
                    sel_indices=s["ids"],
                )
            else:
                context = torch.empty(0, dtype=torch.int64, device=ws.device)
            owned.append(context)
            torch.cuda.synchronize(ws.device)

    ws.collective_check(prepare)
    state = ws._moe_states[q]
    offset = MOE_Q1_COUNTER_OFFSET if q == 1 else MOE_Q4_COUNTER_OFFSET
    counter = state[offset : offset + 4].view(torch.int32)
    diagnostics = dict(s)
    diagnostics["records"] = ws.record_view(
        f"moe_q{q}", Q1_RECORD_BYTES if q == 1 else Q4_RECORD_BYTES
    )
    diagnostics["generation"] = (
        lambda c=counter, delta=(q == 4): (int(c.item()) - int(delta)) & 0xFFFFFFFF
    )
    tensors = [
        x,
        weights.gamma,
        weights.router,
        weights.bias,
        weights.gate_up,
        weights.down,
        weights.down_scale,
        s["scores"],
        s["mid"],
        s["norm"],
        output,
        s["probs"],
        s["ids"],
        state,
        s["shared_down"],
        owned[0],
    ]
    return _Plan(
        ws,
        output,
        ws._modules["moe"],
        tensors,
        owned,
        (x, weights, output),
        diagnostics,
        family="moe",
        q_len=q,
        enable_pdl=enable_pdl,
    )


def _shared_context(device, q, x, position, selected, peer_table, own_slice, out):
    operands = dict(
        x=x,
        cur_pos=position,
        selected_ids=selected,
        id_peer_table=peer_table,
        own_slice=own_slice,
        hidden_out=out,
    )
    # Rank zero's descriptor table without indexer weights. Rows follow the
    # indexer table (CtxTable in csrc/attention/indexer_q4_types.cuh): one
    # 16-word row per stage, where row 6 publishes the selected IDs and row 7
    # adds the attention ranks' partial outputs to the residual. The weight and
    # indexer-scratch words stay zero because shared mode skips those stages.
    rows = [
        {0: "x"},
        {},
        {1: "cur_pos"},
        {},
        {4: "cur_pos", 6: "selected_ids"},
        {1: "selected_ids", 2: "cur_pos"},
        {0: "selected_ids", 1: "id_peer_table"},
        {0: "own_slice", 1: "x", 2: "hidden_out"},
    ]
    # Row 6, word 2: number of published IDs (high half) and record tag (low
    # half); word 3: number of attention peers.
    publication = [
        (6, 2, (q * SELECTED_TOKENS << 32) | _ID_RECORD_TAG),
        (6, 3, WORLD_SIZE - 1),
    ]
    return descriptors._upload(device, rows, operands, publication)


def prepare_attention(
    x,
    weights,
    *,
    caches,
    rope,
    cur_pos,
    out=None,
    workspace,
    indexer=None,
    selection=None,
    enable_pdl=True,
):
    from .attention_weights import validate_packed

    _workspace(workspace, "attention")
    ws = workspace
    result = {}
    caches = {} if caches is None else caches

    def validate():
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must have shape [q_len,6144]")
        q = check_q_len(x.shape[0])
        mode = ("shared" if q == 1 else "full") if indexer is None else indexer
        if mode not in ("full", "shared") or (q == 1 and mode != "shared"):
            raise ValueError(
                "indexer must be full/shared; q1 supports selection reuse only"
            )
        _tensor("x", x, torch.bfloat16, ws.device, (q, HIDDEN_SIZE))
        _tensor("cur_pos", cur_pos, torch.int32, ws.device, elements=1, alignment=4)
        position = int(cur_pos.item())
        if position < 0 or position + q > ws.max_seq_len:
            raise ValueError("cur_pos and query rows must fit workspace.max_seq_len")
        output = torch.empty_like(x) if out is None else out
        _tensor("out", output, torch.bfloat16, ws.device, x.shape)
        compute = ws.rank != 0 or mode == "full"
        w = {}
        if compute:
            validate_packed(weights, ws.rank, ws.device)
            w = weights.tensors
            if not isinstance(caches, dict):
                raise TypeError(
                    "caches must be a dictionary of contiguous BF16 tensors"
                )
            names = (
                {"indexer_k_cache": 128}
                if ws.rank == 0
                else {"kv_cache": 512, "pe_cache": 64}
            )
            sizes = []
            for name, width in names.items():
                if name not in caches:
                    raise ValueError(f"missing cache: {name}")
                value = caches[name]
                _tensor(name, value, torch.bfloat16, ws.device)
                if (
                    value.ndim != 2
                    or value.shape[1] != width
                    or value.shape[0] < ws.max_seq_len + CACHE_PAD
                ):
                    raise ValueError(
                        f"{name} must have at least "
                        f"[{ws.max_seq_len + CACHE_PAD},{width}] entries"
                    )
                sizes.append(value.shape[0])
            if len(set(sizes)) != 1:
                raise ValueError(
                    "latent and positional cache row capacities must match"
                )
            _tensor("rope", rope, torch.float32, ws.device)
            if rope.ndim != 2 or rope.shape[1] != 64 or rope.shape[0] < sizes[0]:
                raise ValueError("rope must be a full [cache_rows,64] table")
        if ws.rank == 0 and mode == "shared":
            _tensor("selection", selection, torch.int32, ws.device, (q, 2048))
        elif mode == "full" and selection is not None:
            raise ValueError("full indexing produces its own selection")
        _disjoint(
            output,
            {
                "x": x,
                "rope": rope,
                "cur_pos": cur_pos,
                "selection": selection,
                **w,
                **caches,
                **ws._attention,
            },
        )
        result.update(q=q, mode=mode, out=output, weights=w)
        return q, mode, position, bool(enable_pdl)

    _agree(ws.collective_check(validate))
    q, mode, output, w = result["q"], result["mode"], result["out"], result["weights"]
    a = ws._attention
    module = ws._modules["attention"]
    owned = []
    selected = selection if ws.rank == 0 and mode == "shared" else a["ids"]

    def tensor_map(base, box_rows, box_cols):
        host = torch.empty(128, dtype=torch.uint8)
        module.encode_tmap(base, box_rows, box_cols, host)
        value = host.to(ws.device)
        owned.append(value)
        return value

    def prepare():
        with torch.cuda.device(ws.device):
            if ws.rank == 0 and mode == "shared":
                context = _shared_context(
                    ws.device,
                    q,
                    x,
                    cur_pos,
                    selected,
                    a["id_peers"],
                    a["own_slice"],
                    output,
                )
            elif ws.rank == 0:
                kt = tensor_map(caches["indexer_k_cache"], 128, 64)
                qt = tensor_map(a["iqrt"].view(128, 128), 32, 64)
                context = descriptors.prepare_indexer(
                    ws.device,
                    x=x,
                    x_gamma=w["x_gamma"],
                    q_gamma=w["q_gamma"],
                    q_a_indexer_wk=w["q_a_indexer_wk"],
                    indexer_weights_proj=w["indexer_weights_proj"],
                    indexer_wq_b=w["indexer_wq_b"],
                    cur_pos=cur_pos,
                    rope=rope,
                    indexer_k_cache=caches["indexer_k_cache"],
                    ln_gamma=w["ln_gamma"],
                    ln_beta=w["ln_beta"],
                    q_a=a["qa"],
                    ki=a["ki"],
                    head_weights=a["headw"],
                    iq=a["iq"],
                    iq_rope=a["iqrt"],
                    logits=a["logits"],
                    histogram=a["hist"],
                    candidates=a["ws"],
                    selected_ids=selected,
                    id_peer_table=a["id_peers"],
                    k_tmap=kt,
                    q_tmap=qt,
                    hidden_out=output,
                )
            elif q == 1:
                context = descriptors.prepare_q1_attention(
                    ws.device,
                    x=x,
                    x_gamma=w["x_gamma"],
                    q_gamma=w["q_gamma"],
                    kv_gamma=w["kv_gamma"],
                    qkv_a_proj=w["qkv_a_proj"],
                    q_b_proj=w["q_b_proj"],
                    w_uk=w["w_uk"],
                    w_uv=w["w_uv"],
                    o_proj=w["o_proj"],
                    cur_pos=cur_pos,
                    rope=rope,
                    kv_cache=caches["kv_cache"],
                    pe_cache=caches["pe_cache"],
                    selected_ids=a["ids"],
                    ll_recv=a["ll"],
                    hidden_out=output,
                    **{
                        key: a[key]
                        for key in (
                            "post_q",
                            "post_kv",
                            "post_qnd",
                            "post_qpe",
                            "post_qnope",
                            "post_o",
                            "post_uv",
                            "o_acc",
                            "lse_acc",
                        )
                    },
                )
            else:
                context = torch.empty(0, dtype=torch.int64, device=ws.device)
            owned.append(context)
            torch.cuda.synchronize(ws.device)

    ws.collective_check(prepare)
    context = owned[-1]
    tensors = [x, output, context, ws._attention_epoch, ws._attention_q1_state]
    if ws.rank != 0:
        tensors += [w[key] for key in ("x_gamma", "q_gamma", "kv_gamma")]
        tensors += [rope] + [
            w[key] for key in ("qkv_a_proj", "q_b_proj", "w_uk", "w_uv", "o_proj")
        ]
        tensors += [
            caches["kv_cache"],
            caches["pe_cache"],
            a["ids"],
            a["lengths"],
            cur_pos,
        ]
        tensors += [
            a[key]
            for key in (
                "post_q",
                "post_kv",
                "post_qnd",
                "post_qpe",
                "post_qnope",
                "post_o",
                "post_uv",
                "ll",
            )
        ]
    diagnostics = dict(a)
    diagnostics["ids"] = selected
    arena = ws.record_view("attention", ATTENTION_ARENA_BYTES)
    diagnostics["arena"] = arena
    if q == 1:
        diagnostics["records"] = ws.record_view("attention_q1", Q1_RECORD_BYTES)
        offset = ATTENTION_Q1_COUNTER_OFFSET
        counter = ws._attention_q1_state[offset : offset + 4].view(torch.int32)
    else:
        offset = ATTENTION_Q4_HIDDEN_RX_OFFSET
        diagnostics["records"] = arena[offset : offset + ATTENTION_Q4_HIDDEN_RX_BYTES]
        counter = ws._attention_epoch[:1]
    diagnostics["generation"] = lambda c=counter: int(c.item()) & 0xFFFFFFFF
    owners = (
        x,
        output,
        cur_pos,
        selected,
        weights if w else None,
        caches if w else None,
        rope if w else None,
    )
    return _Plan(
        ws,
        output,
        module,
        tensors,
        owned,
        owners,
        diagnostics,
        family="attention",
        q_len=q,
        indexer=mode,
        enable_pdl=enable_pdl,
    )
