"""CPU-only regression tests for interrupted/corrupt CuTe cache entries.

Load the stdlib helper directly and execute the real compiler entry points with
mock CUDA boundaries. This exercises preparation and recompile decisions without
initializing B12X, CUTLASS, Torch, or a GPU.
"""

from __future__ import annotations


import ast
from collections import namedtuple
from contextlib import contextmanager, nullcontext, suppress
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import RLock
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


LIB = Path(__file__).resolve().parents[4] / "flashinfer/experimental/b12x/_lib"
spec = importlib.util.spec_from_file_location(
    "cache_integrity", LIB / "cache_integrity.py"
)
integrity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integrity)


def load_functions(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )


class CompileCacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        payload = ("test-compile-payload",)
        self.key = hashlib.sha256(repr(payload).encode()).hexdigest()
        self.obj = self.root / self.key[:2] / (self.key + ".o")
        self.manifest = self.obj.with_suffix(".json")
        self.obj.parent.mkdir()
        self.data = b"compiled object fixture"
        self.compiled = SimpleNamespace(dump_to_object=lambda _: self.data)
        self.compile_kernel = Mock(return_value=self.compiled)
        self.loader = Mock(
            return_value=SimpleNamespace(
                **{
                    "b12x_cute_" + self.key: self.compiled,
                }
            )
        )
        self.modules = {
            name: ModuleType(name)
            for name in (
                "b12x",
                "b12x._lib",
                "b12x._lib.compiler",
                "b12x._lib.compile_plan",
                "b12x._lib.runtime_control",
                "cutlass",
                "cutlass.cute",
                "cutlass.base_dsl",
                "cutlass.base_dsl.export",
                "cutlass.base_dsl.export.external_binary_module",
            )
        }
        self.modules["cutlass"].cute = self.modules["cutlass.cute"]
        self.modules["cutlass.cute"].compile = Mock()
        self.modules[
            "cutlass.base_dsl.export.external_binary_module"
        ].ExternalBinaryModule = self.loader
        self.modules[
            "b12x._lib.runtime_control"
        ].raise_if_kernel_resolution_frozen = Mock()
        self.program_type = namedtuple("ProgramKey", "dialect key name")
        plan = self.modules["b12x._lib.compile_plan"]
        plan.ProgramKey = self.program_type
        plan.DeferredCuTeKernel = object
        plan.planning = lambda: False
        plan.record_program = Mock()
        plan.tag_compiled = lambda compiled, program: compiled
        self.module = self.modules["b12x._lib.compiler"]
        self.namespace = self.module.__dict__
        self.memory = {}
        self.namespace.update(
            __package__="b12x._lib",
            Path=Path,
            os=os,
            json=json,
            hashlib=hashlib,
            tempfile=tempfile,
            shutil=shutil,
            contextmanager=contextmanager,
            nullcontext=nullcontext,
            suppress=suppress,
            atomic_write_bytes=integrity.atomic_write_bytes,
            _mkdir_durable=integrity._mkdir_durable,
            valid_object=integrity.valid_object,
            _cute_compile_cache_dir=lambda: self.root,
            _build_compile_manifest=lambda key, payload, func, data, **kw: {
                "cache_key": key,
                "object_bytes": len(data),
                "object_sha256": hashlib.sha256(data).hexdigest(),
            },
            _compile_memory_cache_key=lambda *args: "memory-key",
            _compile_disk_cache_payload=lambda *args: payload,
            _cute_compile_disk_cache_enabled_for_payload=lambda *args: True,
            _memory_cache_get=self.memory.get,
            _memory_cache_put=self.memory.__setitem__,
            _cute_compile_post_engine_start_log_enabled=lambda: False,
            _cute_compile_log_enabled=lambda: False,
            _compile_target_name=lambda func: "test.kernel",
            _call_cute_compile=self.compile_kernel,
            _MEMORY_CACHE_LOCK=RLock(),
            _DISK_CACHE_HITS=0,
            _COMPILE_MISSES=0,
            _OFFLINE_CUTE_NO_JIT=False,
        )
        load_functions(
            LIB / "compiler.py",
            (
                "_cache_prefix",
                "_cache_object_path",
                "_cache_manifest_path",
                "_cache_lock_path",
                "_write_compile_manifest",
                "_valid_cute_compile_cache",
                "_ensure_cute_compile_manifest",
                "_disk_cache_key_lock",
                "_load_cute_compile_from_disk",
                "_store_cute_compile_to_disk",
                "compile",
            ),
            self.namespace,
        )
        plan.__dict__.update(
            __package__="b12x._lib", _RESIDENT_PROGRAMS=set(), Path=Path
        )
        load_functions(
            LIB / "compile_plan.py", ("compiled_program_available",), plan.__dict__
        )
        self.available = plan.compiled_program_available
        self.program = self.program_type("cute", self.key, "test.kernel")
        modules_patch = patch.dict(sys.modules, self.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def write_pair(self):
        self.obj.write_bytes(self.data)
        self.manifest.write_text(
            json.dumps(
                {
                    "cache_key": self.key,
                    "object_bytes": len(self.data),
                    "object_sha256": hashlib.sha256(self.data).hexdigest(),
                }
            )
        )

    def test_valid_cache_reused_without_compilation(self):
        self.write_pair()
        self.assertTrue(self.available(self.program))
        self.assertIs(self.module.compile(object()), self.compiled)
        self.compile_kernel.assert_not_called()

    def test_corruption_is_unavailable_and_recompiled(self):
        corruptions = {
            "empty_manifest": lambda: self.manifest.write_bytes(b""),
            "nul_manifest": lambda: self.manifest.write_bytes(b"\0" * 64),
            "malformed_manifest": lambda: self.manifest.write_text("{"),
            "missing_manifest": self.manifest.unlink,
            "missing_object": self.obj.unlink,
            "empty_object": lambda: self.obj.write_bytes(b""),
            "truncated_object": lambda: self.obj.write_bytes(self.data[:3]),
            "same_size_corruption": lambda: self.obj.write_bytes(b"x" * len(self.data)),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name):
                self.write_pair()
                self.assertTrue(self.available(self.program))
                corrupt()
                self.assertFalse(self.available(self.program))
                self.memory.clear()
                self.compile_kernel.reset_mock()
                self.assertIs(self.module.compile(object()), self.compiled)
                self.compile_kernel.assert_called_once()
                self.assertTrue(self.available(self.program))

    def test_invalid_manifest_fields_are_misses(self):
        for field, value in (
            ("cache_key", "wrong"),
            ("object_bytes", True),
            ("object_bytes", -1),
            ("object_sha256", None),
            ("object_sha256", "x" * 64),
        ):
            with self.subTest(field=field, value=value):
                self.write_pair()
                data = json.loads(self.manifest.read_text())
                data[field] = value
                self.manifest.write_text(json.dumps(data))
                self.assertFalse(self.available(self.program))
        self.manifest.write_text("[]")
        self.assertFalse(self.available(self.program))

    def test_checksums_rechecked_after_same_size_rewrite(self):
        self.write_pair()
        with patch.object(integrity.hashlib, "sha256", wraps=hashlib.sha256) as sha:
            self.assertTrue(self.available(self.program))
            self.assertTrue(self.available(self.program))
            self.assertEqual(sha.call_count, 2)
            info = self.obj.stat()
            self.obj.write_bytes(b"x" * len(self.data))
            os.utime(self.obj, ns=(info.st_atime_ns, info.st_mtime_ns))
            self.assertFalse(self.available(self.program))
            self.assertEqual(sha.call_count, 3)

    def test_rewrites_detected_when_stat_identity_is_unchanged(self):
        for target in (self.obj, self.manifest):
            with self.subTest(target=target.suffix):
                self.write_pair()
                identities = {
                    path: integrity._file_identity(path)
                    for path in (self.obj, self.manifest)
                }
                with patch.object(
                    integrity, "_file_identity", side_effect=identities.__getitem__
                ):
                    self.assertTrue(self.available(self.program))
                    if target == self.obj:
                        target.write_bytes(b"x" * len(self.data))
                    else:
                        target.write_text(
                            target.read_text().replace(self.key, "0" * len(self.key))
                        )
                    self.assertEqual(target.stat().st_size, identities[target][2])
                    self.assertFalse(self.available(self.program))

    def test_staged_copy_is_validated_and_canonical_object_is_not_modified(self):
        self.write_pair()

        def mutate_copy(path):
            Path(path).write_bytes(b"loader changed this private copy")
            return SimpleNamespace(**{"b12x_cute_" + self.key: self.compiled})

        self.loader.side_effect = mutate_copy
        self.assertIs(self.module._load_cute_compile_from_disk(self.key), self.compiled)
        self.assertEqual(self.obj.read_bytes(), self.data)
        self.assertTrue(self.available(self.program))

        def corrupt_copy(source, target):
            Path(target).write_bytes(b"bad staged object")

        self.loader.reset_mock()
        with patch.object(shutil, "copy2", side_effect=corrupt_copy):
            self.assertIsNone(self.module._load_cute_compile_from_disk(self.key))
        self.loader.assert_not_called()

    def test_interrupted_publication_is_rebuilt_and_healthy_neighbor_is_untouched(self):
        self.write_pair()
        neighbor = self.obj.parent / "other.o"
        neighbor.write_bytes(b"keep this")
        self.data = b"new object with a different checksum"
        with (
            patch.object(
                self.module, "_write_compile_manifest", side_effect=OSError("crash")
            ),
            self.assertRaises(OSError),
        ):
            self.module._store_cute_compile_to_disk(
                self.key,
                self.compiled,
                cache_payload=(1,),
                func=object(),
            )
        self.assertFalse(self.available(self.program))
        self.module.compile(object())
        self.assertTrue(self.available(self.program))
        self.assertEqual(neighbor.read_bytes(), b"keep this")

    def test_compiler_rechecks_after_another_writer_repairs_the_entry(self):
        self.obj.write_bytes(b"corrupt")

        @contextmanager
        def repaired_by_other_writer(key):
            self.write_pair()
            yield

        with patch.object(
            self.module, "_disk_cache_key_lock", repaired_by_other_writer
        ):
            self.module.compile(object())
        self.compile_kernel.assert_not_called()
        self.assertTrue(self.available(self.program))

    def test_publication_fsyncs_file_before_rename_then_directory(self):
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            events.append("fsync")
            real_fsync(fd)

        def replace(source, target):
            events.append("replace")
            real_replace(source, target)

        with patch.object(os, "fsync", fsync), patch.object(os, "replace", replace):
            self.module._store_cute_compile_to_disk(
                self.key,
                self.compiled,
                cache_payload=(1,),
                func=object(),
            )
        self.assertEqual(events, ["fsync", "replace", "fsync"] * 2)
        self.assertTrue(self.available(self.program))

    def test_failure_before_rename_preserves_previous_file_and_cleans_temporary(self):
        self.write_pair()
        with (
            patch.object(os, "fsync", side_effect=OSError("disk failure")),
            self.assertRaises(OSError),
        ):
            integrity.atomic_write_bytes(self.obj, b"replacement")
        self.assertEqual(self.obj.read_bytes(), self.data)
        self.assertEqual(list(self.obj.parent.glob("*.tmp")), [])

    def test_new_cache_directory_entries_are_synced(self):
        target = self.root / "new" / "shard" / "object.o"
        with patch.object(
            integrity, "_fsync_directory", wraps=integrity._fsync_directory
        ) as sync:
            integrity.atomic_write_bytes(target, self.data)
        self.assertEqual(
            [call.args[0] for call in sync.call_args_list],
            [self.root, self.root / "new", self.root / "new/shard"],
        )

    def test_preparation_metadata_is_not_scanned(self):
        selection = self.root / "preparation" / "selection.json"
        selection.parent.mkdir()
        selection.write_text('{"identity":{},"records":{}}')
        self.write_pair()
        self.assertTrue(self.available(self.program))
        self.module.compile(object())
        self.assertEqual(selection.read_text(), '{"identity":{},"records":{}}')


if __name__ == "__main__":
    unittest.main()
