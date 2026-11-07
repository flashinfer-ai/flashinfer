# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone host contracts; these tests neither load nor validate GPU code.

Can run without the GPU package via unittest discovery. Loading the two leaf
modules is deliberate: it tests their no-Torch/no-NCCL import contract without
changing normal pytest collection. Resource failure tests use host doubles for
CUDA/NCCL; they do not validate device execution. The normal repository
conftest still requires a full FlashInfer environment.
"""

import importlib.util
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _load_leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PACKAGE = (
    Path(__file__).resolve().parents[3]
    / "flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe"
)
gin = _load_leaf("_fi_gin_host_contract", _PACKAGE / "shim/gin.py")
wire = _load_leaf("_fi_gin_wire_layout", _PACKAGE / "src/src/nccl_gin_layout.py")


class ArenaTests(unittest.TestCase):
    def layout(self, **changes):
        kwargs = dict(
            transport_bytes=257,
            num_tokens=3,
            hidden=80,
            num_topk=4,
            shared_workspace_bytes=129,
        )
        kwargs.update(changes)
        return gin.build_gin_arena_layout(**kwargs)

    def test_arena_partition_preserves_root_origin_and_padding(self):
        layout = self.layout()
        self.assertEqual(
            [(r.name, r.offset, r.nbytes) for r in layout.regions],
            [
                ("gin_transport", 0, 257),
                ("activation", 384, 120),
                ("activation_sf", 512, 24),
                ("topk_weights", 640, 48),
                ("shared_workspace", 768, 129),
            ],
        )
        self.assertEqual(layout.total_bytes, 1024)
        # 80/16=5 SF bytes per row padded to 8, independently of token count.
        self.assertEqual(layout.region("activation_sf").nbytes, 24)

    def test_invalid_capacity_and_nvfp4_block(self):
        for changes in (
            {"hidden": 17},
            {"num_tokens": 0},
            {"num_topk": True},
            {"transport_bytes": -1},
            {"shared_workspace_bytes": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.layout(**changes)

    def test_invalid_manual_partitions(self):
        region = gin.GinArenaRegion
        for regions, total in (
            ((region("gin_transport", 128, 128),), 256),
            ((region("gin_transport", 0, 256), region("activation", 128, 128)), 384),
            ((region("gin_transport", 0, 128), region("gin_transport", 128, 128)), 256),
            ((region("gin_transport", 0, 257),), 256),
        ):
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                gin.GinArenaLayout(regions, total).validate()

    def test_wire_message_and_bf16_double_buffer(self):
        layout = wire.build_megamoe_gin_window_layout(
            world_size=4,
            ranks_per_node=2,
            num_experts_per_rank=12,
            max_relay_tokens_per_node=32,
            hidden_bytes=3584,
            sf_uint32_per_token=112,
            num_topk=4,
            combine_hidden=7168,
        )
        message = layout.relay_message
        self.assertEqual(
            (
                message.src_metadata_offset,
                message.route_words_offset,
                message.topk_weights_offset,
                message.token_data_offset,
                message.sf_data_offset,
                message.message_bytes,
            ),
            (0, 16, 32, 48, 3632, 4096),
        )
        self.assertEqual(layout.internode_send_count_shape, (1, 25))
        self.assertEqual(layout.internode_recv_splits_offset, 128)
        self.assertEqual(layout.relay_send_buffer_offset, 256)
        self.assertEqual(layout.relay_recv_buffer_offset, 131328)
        self.assertEqual(layout.intra_node_reduce_buf_offset, 262400)
        self.assertEqual(layout.internode_combine_buffer_offset, 721152)
        self.assertEqual(layout.internode_combine_buffer_shape, (2, 1, 32, 7168))
        self.assertEqual(layout.total_bytes, 1638656)
        self.assertEqual(layout.total_signal_count, 5)
        self.assertEqual(
            self.layout(transport_bytes=layout.total_bytes).region("activation").offset,
            layout.total_bytes,
        )


class LsaTests(unittest.TestCase):
    def offsets(self, **changes):
        kwargs = dict(
            rank=2,
            world_size=4,
            ranks_per_node=2,
            base_address=1024,
            peer_addresses=((2, 1024), (3, 768)),
        )
        kwargs.update(changes)
        return gin.lsa_peer_offsets(**kwargs)

    def test_same_domain_global_ranks_and_signed_deltas(self):
        self.assertEqual(self.offsets(), (0, -256))
        self.assertEqual(
            self.offsets(peer_addresses=((2, 2048), (3, 768))), (1024, -256)
        )

    def test_unreachable_cross_domain_misordered_and_invalid_self(self):
        for changes in (
            {"peer_addresses": ((0, 1024), (1, 2048))},
            {"peer_addresses": ((2, 1024), (3, 0))},
            {"peer_addresses": ((3, 768), (2, 1024))},
            {"peer_addresses": ((2, 2049), (3, 768))},
            {"world_size": 64, "ranks_per_node": 32},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.offsets(**changes)


class _ControlWorld:
    """In-memory all-gather transport for concurrent host contract testing."""

    def __init__(self, size):
        self.reports = [None] * size
        self.barrier = threading.Barrier(size, timeout=5)
        self.world = object()

    def rank(self, rank):
        def gather(output, report, *, group):
            assert group is self.world
            self.reports[rank] = report
            self.barrier.wait()
            output[:] = self.reports
            self.barrier.wait()

        return SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_rank=lambda *, group: rank,
            get_world_size=lambda *, group: len(self.reports),
            all_gather_object=gather,
        )


class PreflightTests(unittest.TestCase):
    def test_cpu_control_reuses_world(self):
        world = object()
        dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=world),
            get_backend_config=lambda group: "cpu:gloo,cuda:nccl",
        )
        self.assertEqual(gin._gin_control_group(dist), (world, True))

    def test_nccl_only_control_is_reused_and_scoped_to_world(self):
        class World:
            pass

        created = []

        def new_group(*, backend):
            self.assertEqual(backend, "gloo")
            group = object()
            created.append(group)
            return group

        dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=World()),
            get_backend_config=lambda group: "cuda:nccl",
            new_group=new_group,
        )
        first, cpu = gin._gin_control_group(dist)
        self.assertTrue(cpu)
        self.assertIs(gin._gin_control_group(dist)[0], first)
        self.assertEqual(len(created), 1)
        dist.group.WORLD = World()
        self.assertIsNot(gin._gin_control_group(dist)[0], first)
        self.assertEqual(len(created), 2)

    def test_rail_capability_requires_device_rdma_and_distinct_domains(self):
        def support(**changes):
            fields = dict(device_api_support=True, railed_gin_type="GPI", n_lsa_teams=2)
            fields.update(changes)
            return SimpleNamespace(**fields)

        gin.validate_gin_rail_support(support(), world_size=4, ranks_per_node=2)
        for changes in (
            {"device_api_support": False},
            {"railed_gin_type": "NONE"},
            {"railed_gin_type": None},
            {"n_lsa_teams": 1},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises(gin.GinPreflightError),
            ):
                gin.validate_gin_rail_support(
                    support(**changes), world_size=4, ranks_per_node=2
                )

    def run_world(self, *, errors=None, subgroup_rank=None, differing_rank=None):
        world = _ControlWorld(4)

        def run(rank):
            try:
                gin.collective_gin_preflight(
                    world.rank(rank),
                    phase="before-window",
                    signature=("N2R2", 32 if rank != differing_rank else 64),
                    local_error=(errors or {}).get(rank),
                    requested_group=object() if rank == subgroup_rank else None,
                )
            except gin.GinPreflightError as exc:
                return str(exc)
            return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(run, range(4)))

    def test_agreement_allows_boundary(self):
        self.assertEqual(self.run_world(), [None] * 4)

    def test_one_rank_failure_reaches_every_rank(self):
        results = self.run_world(errors={2: "nccl4py is missing"})
        self.assertEqual(len(set(results)), 1)
        self.assertIn("rank 2: nccl4py is missing", results[0])

    def test_subgroup_rejected_on_default_group(self):
        results = self.run_world(subgroup_rank=1)
        self.assertEqual(len(set(results)), 1)
        self.assertIn("rank 1: EP subgroups are unsupported", results[0])

    def test_configuration_mismatch_reaches_every_rank(self):
        results = self.run_world(differing_rank=3)
        self.assertEqual(len(set(results)), 1)
        self.assertIn("rank 3: configuration differs", results[0])

    def test_uninitialized_control_plane_fails_before_collective(self):
        dist = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
        with self.assertRaisesRegex(gin.GinPreflightError, "initialized default"):
            gin.collective_gin_preflight(
                dist, phase="init", signature=(), local_error=None
            )


class ConstructionTests(unittest.TestCase):
    def run_world(self, *, fail_build=False, fail_cleanup=False, capture_init=False):
        world = _ControlWorld(4)
        local = threading.local()
        dist = ModuleType("torch.distributed")
        dist.group = SimpleNamespace(WORLD=world.world)
        dist.is_available = lambda: True
        dist.is_initialized = lambda: True
        dist.get_rank = lambda group=None: local.rank
        dist.get_world_size = lambda group=None: 4
        dist.all_gather_object = lambda *args, **kwargs: world.rank(
            local.rank
        ).all_gather_object(*args, **kwargs)
        torch = ModuleType("torch")
        torch.distributed = dist
        torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            is_current_stream_capturing=lambda: local.rank == 1,
            current_device=lambda: local.rank,
        )

        def run(rank):
            local.rank = rank
            calls = []
            session = gin.GinSession.__new__(gin.GinSession)
            gin.GinResourceOwner.__init__(
                session, synchronize=lambda: calls.append("drain")
            )
            session.identity = "shared-session"
            session.views = {"view": object()}
            session.root = object()

            def close_comm():
                calls.append("comm")
                if fail_cleanup and rank == 2:
                    raise RuntimeError("injected close failure")

            session.comm = SimpleNamespace(close=close_comm)
            try:
                if capture_init:
                    gin.create_gin_session(
                        rank=rank,
                        world_size=4,
                        ranks_per_node=2,
                        device=rank,
                        layout=gin.build_gin_arena_layout(
                            transport_bytes=128,
                            num_tokens=1,
                            hidden=64,
                            num_topk=4,
                            shared_workspace_bytes=128,
                        ),
                        signal_count=1,
                        signature=("init",),
                    )
                else:
                    with session.workspace_construction():
                        if fail_build and rank == 1:
                            raise MemoryError("injected allocation failure")
                        calls.append("allocated")
            except Exception as exc:
                return type(exc), str(exc), calls, session
            return None, None, calls, session

        with (
            patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}),
            patch.object(gin, "require_gin_apis", return_value=None),
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            return list(executor.map(run, range(4)))

    def test_publish_only_after_all_ranks_allocate(self):
        results = self.run_world()
        self.assertTrue(all(error is None for error, _, _, _ in results))
        self.assertTrue(all(not owner._closed for _, _, _, owner in results))

    def test_one_allocation_failure_cleans_every_session(self):
        results = self.run_world(fail_build=True)
        self.assertEqual({error for error, _, _, _ in results}, {gin.GinPreflightError})
        self.assertEqual(len({message for _, message, _, _ in results}), 1)
        self.assertIn("rank 1: MemoryError", results[0][1])
        for _, _, calls, owner in results:
            self.assertEqual(calls[-2:], ["drain", "comm"])
            self.assertTrue(owner._closed)
            self.assertIsNone(owner.root)
            self.assertEqual(owner.views, {})

    def test_cleanup_failure_is_fatal_on_every_rank(self):
        count = len(gin._failed_sessions)
        try:
            results = self.run_world(fail_build=True, fail_cleanup=True)
            self.assertEqual({error for error, _, _, _ in results}, {gin.GinFatalError})
            self.assertIsNotNone(results[2][3].root)
            self.assertIn(results[2][3], gin._failed_sessions)
        finally:
            del gin._failed_sessions[count:]

    def test_rank_local_capture_init_is_collectively_rejected(self):
        results = self.run_world(capture_init=True)
        self.assertEqual({error for error, _, _, _ in results}, {gin.GinPreflightError})
        self.assertEqual(len({message for _, message, _, _ in results}), 1)
        self.assertIn("rank 1: ValueError", results[0][1])
        self.assertIn("capture", results[0][1])


class SessionInitializationTests(unittest.TestCase):
    def run_world(self, *, fail_cleanup=False, fail_control=False):
        """Exercise the real factory with host doubles for CUDA/NCCL resources."""
        world = _ControlWorld(4)
        local = threading.local()
        calls = [[] for _ in range(4)]
        sessions = [None] * 4
        dist = ModuleType("torch.distributed")
        dist.group = SimpleNamespace(WORLD=world.world)
        dist.is_available = dist.is_initialized = lambda: True
        dist.get_rank = lambda group=None: local.rank
        dist.get_world_size = lambda group=None: 4

        def gather(output, report, *, group):
            calls[local.rank].append(report[1])
            if fail_control and report[1] == "lsa-peer-mapping-cleanup":
                raise OSError("injected cleanup agreement failure")
            world.rank(local.rank).all_gather_object(output, report, group=group)

        def broadcast(payload, **kwargs):
            reports = [None] * 4
            world.rank(local.rank).all_gather_object(
                reports, payload[0], group=world.world
            )
            payload[0] = reports[0]

        dist.all_gather_object = gather
        dist.broadcast_object_list = broadcast
        torch = ModuleType("torch")
        torch.distributed = dist
        torch.uint8 = object()
        torch.device = lambda *args: args
        torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            is_current_stream_capturing=lambda: False,
            current_device=lambda: local.rank,
            synchronize=lambda device: calls[local.rank].append("drain"),
        )

        def make_comm(*args):
            rank = local.rank

            def close():
                calls[rank].append("comm")
                if fail_cleanup and rank == 2:
                    raise RuntimeError("injected close failure")

            def peer_pointer(peer):
                if rank == 1:
                    raise RuntimeError("injected LSA mapping failure")
                return 1024 + peer * 128

            return SimpleNamespace(
                close=close,
                device_api_support=True,
                railed_gin_type="GPI",
                n_lsa_teams=2,
                create_dev_comm=lambda requirements: SimpleNamespace(
                    ptr=1, destroy=lambda: calls[rank].append("device")
                ),
                register_window=lambda *args: SimpleNamespace(
                    handle=1,
                    get_peer_device_pointer=peer_pointer,
                    close=lambda: calls[rank].append("window"),
                ),
            )

        core = ModuleType("nccl.core")
        core.Communicator = SimpleNamespace(init=make_comm)
        core.NCCLDevCommRequirements = lambda **kwargs: kwargs
        core.NcclGinConnectionType = SimpleNamespace(RAIL=1)
        core.UniqueId = SimpleNamespace(from_bytes=lambda value: value)
        core.WindowFlag = SimpleNamespace(COLL_SYMMETRIC=1)
        core.get_unique_id = lambda: b"shared-uid"
        interop = ModuleType("nccl.core.interop.torch")
        interop.empty = lambda *args, **kwargs: SimpleNamespace(
            zero_=lambda: None, data_ptr=lambda: 1024
        )
        session_type = gin.GinSession

        def make_session(**kwargs):
            session = sessions[local.rank] = session_type(**kwargs)
            return session

        def run(rank):
            local.rank = rank
            try:
                gin.create_gin_session(
                    rank=rank,
                    world_size=4,
                    ranks_per_node=2,
                    device=rank,
                    layout=gin.build_gin_arena_layout(
                        transport_bytes=128,
                        num_tokens=1,
                        hidden=64,
                        num_topk=4,
                        shared_workspace_bytes=128,
                    ),
                    signal_count=1,
                    signature=("init",),
                )
            except Exception as exc:
                return exc
            self.fail("rank-local LSA failure must reject initialization")

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "torch.distributed": dist,
                    "nccl.core": core,
                    "nccl.core.interop.torch": interop,
                },
            ),
            patch.object(gin, "require_gin_apis", return_value=None),
            patch.object(gin, "GinSession", side_effect=make_session),
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            errors = list(executor.map(run, range(4)))
        return errors, calls, sessions

    def test_init_rejection_cleans_all_ranks_and_preserves_recoverable_error(self):
        errors, calls, sessions = self.run_world()
        self.assertEqual({type(error) for error in errors}, {gin.GinPreflightError})
        self.assertEqual(len({str(error) for error in errors}), 1)
        self.assertIn(
            "rank 1: RuntimeError: injected LSA mapping failure", str(errors[0])
        )
        for trace, session in zip(calls, sessions, strict=True):
            self.assertEqual(
                trace[-5:],
                ["drain", "window", "device", "comm", "lsa-peer-mapping-cleanup"],
            )
            self.assertTrue(session._closed)
            self.assertIsNone(session.root)

    def test_init_cleanup_failure_is_fatal_on_every_rank(self):
        count = len(gin._failed_sessions)
        try:
            errors, calls, sessions = self.run_world(fail_cleanup=True)
            self.assertEqual({type(error) for error in errors}, {gin.GinFatalError})
            for error, trace in zip(errors, calls, strict=True):
                self.assertIn(
                    "rank 2: RuntimeError: injected close failure", str(error.__cause__)
                )
                self.assertEqual(trace[-1], "lsa-peer-mapping-cleanup")
            self.assertIsNotNone(sessions[2].root)
            self.assertIn(sessions[2], gin._failed_sessions)
            self.assertTrue(
                all(session._closed for i, session in enumerate(sessions) if i != 2)
            )
        finally:
            del gin._failed_sessions[count:]

    def test_init_cleanup_control_failure_is_fatal_on_every_rank(self):
        errors, _, sessions = self.run_world(fail_control=True)
        self.assertEqual({type(error) for error in errors}, {gin.GinFatalError})
        self.assertTrue(all(session._closed for session in sessions))


class LifecycleTests(unittest.TestCase):
    def owner(self, *, fail_drain=False, fail_window=False):
        calls = []

        def drain():
            calls.append("drain")
            if fail_drain:
                raise RuntimeError("in flight")

        def window_close():
            calls.append("window")
            if fail_window:
                raise RuntimeError("window failure")

        owner = gin.GinResourceOwner(synchronize=drain)
        owner.root = object()
        owner.window = SimpleNamespace(close=window_close)
        owner.dev_comm = SimpleNamespace(destroy=lambda: calls.append("device"))
        owner.comm = SimpleNamespace(close=lambda: calls.append("comm"))
        return owner, calls

    def test_close_order_and_idempotence(self):
        owner, calls = self.owner()
        owner.close()
        owner.close()
        self.assertEqual(calls, ["drain", "window", "device", "comm"])
        self.assertIsNone(owner.root)

    def test_failed_drain_keeps_registered_allocation_alive(self):
        owner, calls = self.owner(fail_drain=True)
        with self.assertRaisesRegex(RuntimeError, "in flight"):
            owner.close()
        self.assertEqual(calls, ["drain"])
        self.assertIsNotNone(owner.root)
        self.assertIsNotNone(owner.window)

    def test_failed_window_retains_dependencies_for_retry(self):
        owner, calls = self.owner(fail_window=True)
        with self.assertRaisesRegex(RuntimeError, "window failure"):
            owner.close()
        self.assertEqual(calls, ["drain", "window"])
        self.assertIsNotNone(owner.dev_comm)
        self.assertIsNotNone(owner.comm)
        self.assertIsNotNone(owner.root)
        owner.window.close = lambda: calls.append("retry-window")
        owner.close()
        self.assertEqual(calls[-4:], ["drain", "retry-window", "device", "comm"])

    def test_partial_initialization_releases_only_owned_resources(self):
        calls = []
        owner = gin.GinResourceOwner(synchronize=lambda: calls.append("drain"))
        owner.comm = SimpleNamespace(destroy=lambda: calls.append("comm"))
        owner.close()
        self.assertEqual(calls, ["drain", "comm"])


if __name__ == "__main__":
    unittest.main()
