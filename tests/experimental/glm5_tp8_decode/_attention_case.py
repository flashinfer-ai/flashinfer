# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
import torch

from flashinfer.experimental.glm5_tp8_decode.attention_weights import (
    shard_attention_weights,
)

from . import _api, data
from .moe_reference import bitwise
from ._attention_checks import check_attention, check_indexer


class AttentionCase:
    def __init__(self, control, capacity=4104, seed=1, *, layer=None, packed=None):
        self.control, self.capacity = control, capacity
        self.ws = None
        self.stream = torch.cuda.Stream(device=control.device)
        self.plans, self.graphs = {}, {}
        self.counts = {1: 0, 4: 0}
        # Every rank builds the same CPU checkpoint layer; packing copies only
        # the rank's slices to its GPU.
        self.layer = data.make_attention_layer(seed) if layer is None else layer
        # Rank-local checkpoint tensors on the GPU, read by the references.
        self.weights = control.checked(
            lambda: data.to_device(
                shard_attention_weights(self.layer, rank=control.rank),
                control.device,
            )
        )
        self.packed = (
            control.checked(
                lambda: _api.pack_attention(self.layer, control.rank, control.device)
            )
            if packed is None
            else packed
        )
        # Packing must not write its inputs: snapshot the checkpoint layer it reads.
        self.layer_snapshots = {
            key: tensor.clone() for key, tensor in self.layer.items()
        }
        self.packed_snapshots = {
            key: tensor.clone() for key, tensor in self.packed.tensors.items()
        }
        self.caches = data.make_state(control.rank, capacity + 8, seed, control.device)
        self.rope = data.make_rope_table(capacity, control.device)
        self.rope_snapshot = self.rope.clone()
        self.position = torch.zeros(1, dtype=torch.int32, device=control.device)
        self.x, self.out, self.out_storage, self.selection = {}, {}, {}, {}
        try:
            self.ws = _api.workspace(
                control, capacity=capacity, operators=("attention",)
            )
            for q_len, mode in ((4, "full"), (4, "shared"), (1, "shared")):
                key = (q_len, mode)
                self.x[key] = torch.empty(
                    q_len, 6144, dtype=torch.bfloat16, device=control.device
                )
                self.out_storage[key] = torch.full(
                    (q_len + 1, 6144), 17, dtype=torch.bfloat16, device=control.device
                )
                self.out[key] = self.out_storage[key][:q_len]
                self.selection[key] = (
                    data.make_selection(q_len, 0, seed).to(control.device)
                    if mode == "shared" and control.rank == 0
                    else None
                )
                self.plans[key] = _api.prepare_attention(
                    self.x[key],
                    self.packed,
                    self.caches,
                    self.rope,
                    self.position,
                    self.out[key],
                    self.ws,
                    indexer=mode,
                    selection=self.selection[key],
                )
            control.sync()
        except Exception:
            if self.ws is not None:
                self.ws.destroy()
            raise

    def capture(self):
        self.control.sync()
        before = {
            key: int(plan._diagnostics["generation"]())
            for key, plan in self.plans.items()
        }
        for key, plan in self.plans.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                plan()
            self.graphs[key] = graph
        self.control.sync()
        assert {
            key: int(plan._diagnostics["generation"]())
            for key, plan in self.plans.items()
        } == before

    def run(
        self,
        q_len,
        mode,
        past,
        seed,
        *,
        graph=False,
        validate=True,
        selection_seed=None,
        explicit_selection=None,
    ):
        assert past >= 0 and past + q_len <= self.capacity
        key = (q_len, mode)
        expected_input = data.make_attention_input(q_len, seed, self.control.device)
        if explicit_selection is None:
            selected = data.make_selection(
                q_len, past, seed if selection_seed is None else selection_seed
            )
        else:
            assert mode == "shared" and selection_seed is None
            assert explicit_selection.shape == (q_len, 2048)
            assert explicit_selection.dtype == torch.int32
            selected = explicit_selection.detach().cpu().clone()
        initial = {name: tensor.clone() for name, tensor in self.caches.items()}
        self.stream.wait_stream(torch.cuda.current_stream(self.control.device))
        with torch.cuda.stream(self.stream):
            self.x[key].copy_(expected_input)
            self.position.fill_(past)
            self.out_storage[key].fill_(17)
            if self.selection[key] is not None:
                self.selection[key].copy_(selected)
            if graph:
                self.graphs[key].replay()
            else:
                result = self.plans[key]()
                assert result.data_ptr() == self.out[key].data_ptr()
        self.counts[q_len] += 1
        self.control.sync()

        def guards():
            assert bitwise(self.x[key], expected_input), "attention modified input"
            assert bool((self.out_storage[key][q_len:] == 17).all()), (
                "attention output guard modified"
            )
            assert self.position.item() == past, "attention modified cur_pos"
            assert bitwise(self.rope, self.rope_snapshot), (
                "attention modified RoPE table"
            )
            if self.selection[key] is not None:
                assert bitwise(self.selection[key].cpu(), selected), (
                    "shared indexer modified selection"
                )
            generation = int(self.plans[key]._diagnostics["generation"]())
            assert generation == self.counts[q_len], (
                key,
                generation,
                self.counts[q_len],
            )
            return generation

        generation = self.control.checked(guards)
        assert self.control.gather(generation) == [generation] * 8
        if validate:
            return check_attention(
                self.control,
                self.plans[key],
                self.weights,
                self.caches,
                initial,
                self.x[key],
                self.out[key],
                self.rope,
                past,
                indexer=mode,
                selection=self.selection[key],
            )
        return self.out[key].cpu().clone()

    def full_selection(self, past):
        """Validate the current full indexer and copy its actual publication order.

        Dense rows are canonicalized to the causal prefix because they have no
        LL publication. Sparse rows keep their published order, so a shared-mode
        replay can reproduce the call.
        """
        key = (4, "full")
        return check_indexer(
            self.control,
            self.plans[key],
            self.weights,
            self.caches,
            self.x[key],
            self.rope,
            past,
            indexer="full",
            selection=None,
        )

    def close(self):
        if self.ws is None:
            return
        self.control.sync()

        def immutable():
            for key, tensor in self.layer.items():
                assert bitwise(tensor, self.layer_snapshots[key]), (
                    f"checkpoint tensor {key} was modified"
                )
            for key, tensor in self.packed.tensors.items():
                assert bitwise(tensor, self.packed_snapshots[key]), (
                    f"packed weight {key} was modified"
                )

        self.control.checked(immutable)
        self.graphs.clear()
        self.ws.destroy()
        assert self.ws.memory_usage() == {"symmetric_bytes": 0, "local_tensor_bytes": 0}
        self.ws.destroy()
