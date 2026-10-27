"""CPU-only Docker lifecycle tests, kept outside GPU test discovery."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "ci/bash.sh"
IMAGE_ID = "sha256:" + "a" * 64


class CompileCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "docker.jsonl"
        self.cache = self.root / "cache"
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("CI_COMPILE_", "MOCK_"))
            and key not in {"RUNNER_NAME", "DOCKER_IMAGE_ID"}
        }
        self.env.update(
            PATH=f"{self.bin}:/usr/bin:/bin",
            CI_PIP_CACHE_DIR="",
            CI_COMPILE_CACHE_DIR=str(self.cache),
            CI_COMPILE_CACHE_MODE="read-write",
            CI_COMPILE_CACHE_MIN_FREE_MB="0",
            CI_COMPILE_CACHE_RUNNER="runner-a",
            MOCK_DOCKER_LOG=str(self.log),
            MOCK_IMAGE_ID=IMAGE_ID,
        )
        self.tool(
            "docker",
            """#!/usr/bin/python3
import json, os, shutil, sys, time
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['MOCK_DOCKER_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] == 'pull':
    sys.exit(int(os.environ.get('MOCK_PULL_STATUS', '0')))
if args[:2] == ['image', 'inspect']:
    print(os.environ['MOCK_IMAGE_ID'])
if args[0] == 'run':
    mounts = [args[i+1] for i, arg in enumerate(args) if arg == '-v']
    private = next((m[:-len(':/compile-cache')] for m in mounts if m.endswith(':/compile-cache')), None)
    seed = next((m[:-len(':/compile-cache-seed:ro')] for m in mounts if m.endswith(':/compile-cache-seed:ro')), None)
    if seed:
        shutil.copytree(seed, private, dirs_exist_ok=True)
    if private:
        target = Path(private) / 'flashinfer/.cache/flashinfer/v1/sm100/cached_ops/from-run/build.ninja'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        (Path(private) / 'new-artifact').touch()
    if os.environ.get('MOCK_RUNNING_MARKER'):
        Path(os.environ['MOCK_RUNNING_MARKER']).touch()
    if os.environ.get('MOCK_RELEASE_MARKER'):
        deadline = time.monotonic() + 10
        while not Path(os.environ['MOCK_RELEASE_MARKER']).exists():
            if time.monotonic() >= deadline:
                sys.exit(99)
            time.sleep(0.01)
    sys.exit(int(os.environ.get('MOCK_DOCKER_STATUS', '0')))
""",
        )

    def tool(self, name, contents):
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def image_dir(self, runner="runner-a", image_id=IMAGE_ID):
        key = hashlib.sha256(runner.encode()).hexdigest()
        return self.cache / f"runner-{key}/images" / image_id.removeprefix("sha256:")

    def put(self, relative, **kwargs):
        path = self.image_dir(**kwargs) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("warm artifact")
        return path

    def run_script(self, *args):
        result = subprocess.run(
            ["bash", str(SCRIPT), "org/image:tag", *args, "true"],
            env=self.env,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=15,
        )
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, calls

    def test_opt_out_and_cpu_mode(self):
        for args in (("--no-gpu",), ()):
            if not args:
                self.env.pop("CI_COMPILE_CACHE_DIR")
            result, calls = self.run_script(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("TRITON_CACHE_DIR=/compile-cache/triton", calls[-1])
        self.assertFalse(self.cache.exists())

    def test_pruning_retains_content_keyed_objects(self):
        base = "flashinfer/.cache/flashinfer/v1/sm100/"
        removed = [
            self.put(base + name)
            for name in (
                "cached_ops/nvcc/build.ninja",
                "cached_ops/nvcc/kernel.so",
                "generated/source.cu",
                "autotune/results.json",
                "flashinfer_jit.log",
            )
        ]
        retained = [
            self.put(base + "cached_ops/example_cute_dsl/kernel.o"),
            self.put("triton/kernel.cubin"),
        ]
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(not path.exists() for path in removed))
        self.assertTrue(all(path.exists() for path in retained))
        self.assertFalse(any(self.cache.rglob("build.ninja")))

    def test_pulled_image_identity_isolates_rebuilt_tags(self):
        old = self.put("triton/old")
        self.env["MOCK_IMAGE_ID"] = "sha256:" + "b" * 64
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0], ["pull", "org/image:tag"])
        self.assertTrue(old.exists())
        self.assertIn(self.env["MOCK_IMAGE_ID"], calls[-1])
        image_dir = self.image_dir(image_id=self.env["MOCK_IMAGE_ID"])
        self.assertTrue(image_dir.exists())
        self.assertIn(f"{image_dir}:/compile-cache", calls[-1])
        self.assertIn("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1", calls[-1])

    def test_runner_eviction_cannot_remove_another_runners_cache(self):
        own = self.put("triton/old")
        other = self.put("triton/other", runner="runner-b")
        self.env["CI_COMPILE_CACHE_MAX_MB"] = "0"
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(own.exists())
        self.assertTrue(other.exists())

    def test_disk_pressure_and_stale_image_eviction(self):
        old = self.put("triton/old", image_id="sha256:" + "b" * 64)
        os.utime(old.parents[1], (0, 0))
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(old.exists())
        seed = self.put("triton/warm")
        self.tool(
            "df",
            '#!/bin/bash\nprintf "Filesystem MB Used Available Capacity Mount\\nmock 100 99 1 99%% /tmp\\n"\n',
        )
        self.env.update(
            CI_COMPILE_CACHE_MIN_FREE_MB="2", CI_COMPILE_CACHE_MODE="read-only"
        )
        result, calls = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(seed.exists())
        self.assertNotIn(f"{self.image_dir()}:/compile-cache-seed:ro", calls[-1])
        self.env["CI_COMPILE_CACHE_MODE"] = "read-write"
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(seed.exists())
        self.assertIn("Clearing compile cache", result.stdout)

    def test_read_only_jobs_preserve_seed_and_cleanup_on_success_or_failure(self):
        seed = self.put("triton/warm")
        self.env.pop("CI_COMPILE_CACHE_MODE")  # read-only is the default
        before = seed.stat().st_mtime_ns
        for status in (0, 42):
            with self.subTest(status=status):
                self.env["MOCK_DOCKER_STATUS"] = str(status)
                result, calls = self.run_script()
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertIn(f"{self.image_dir()}:/compile-cache-seed:ro", calls[-1])
                self.assertIn("/docker/compile_cache_entrypoint.sh", calls[-1])
                self.assertEqual(seed.read_text(), "warm artifact")
                self.assertEqual(seed.stat().st_mtime_ns, before)
                self.assertFalse((self.image_dir() / "new-artifact").exists())
                self.assertFalse(list(self.cache.rglob(".job.*")))

    def test_docker_failure_preserves_exit_status_and_prunes(self):
        self.env["MOCK_DOCKER_STATUS"] = "42"
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertFalse(any(self.cache.rglob("build.ninja")))

    def test_failed_pull_does_not_run_or_mutate_cache(self):
        seed = self.put("triton/warm")
        self.env["MOCK_PULL_STATUS"] = "1"
        result, calls = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(seed.exists())

    def test_invalid_configuration_disables_caching(self):
        for key in (
            "CI_COMPILE_CACHE_MAX_MB",
            "CI_COMPILE_CACHE_MIN_FREE_MB",
            "CI_COMPILE_CACHE_MODE",
            "MOCK_IMAGE_ID",
        ):
            with self.subTest(key=key):
                previous = self.env.get(key)
                self.env[key] = "invalid"
                result, calls = self.run_script()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("TRITON_CACHE_DIR=/compile-cache/triton", calls[-1])
                if previous is None:
                    del self.env[key]
                else:
                    self.env[key] = previous

    def test_same_runner_is_locked_across_eviction_and_execution(self):
        marker = self.root / "running"
        release = self.root / "release"
        self.env.update(
            MOCK_RUNNING_MARKER=str(marker), MOCK_RELEASE_MARKER=str(release)
        )
        command = ["bash", str(SCRIPT), "org/image:tag", "true"]
        first = subprocess.Popen(
            command,
            env=self.env,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(first.communicate, timeout=10)
        self.addCleanup(release.touch)
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(marker.exists())
        lock = next(self.cache.glob(".runner-*.lock"))
        inode = lock.stat().st_ino
        # Probe the real lock while Docker waits for an explicit release.
        # A fixed sleep could end before the contender reaches flock on busy CI.
        contender = subprocess.run(["flock", "-n", str(lock), "true"], timeout=5)
        self.assertEqual(contender.returncode, 1)
        release.touch()
        stdout, stderr = first.communicate(timeout=10)
        self.assertEqual(first.returncode, 0, stdout + stderr)
        self.env["CI_COMPILE_CACHE_MAX_MB"] = "0"
        result, _ = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        # The stable lock file survives eviction, preventing inode replacement.
        self.assertEqual(lock.stat().st_ino, inode)

    def test_private_entrypoint_preserves_timestamps_and_command_status(self):
        seed = self.root / "seed"
        private = self.root / "private"
        seed.mkdir()
        private.mkdir()
        (seed / "artifact").write_text("warm")
        os.utime(seed / "artifact", ns=(123456789000, 123456789000))
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "ci/compile_cache_entrypoint.sh"),
                str(seed),
                str(private),
                "bash",
                "-c",
                'touch "$1/command-ran"; exit 42',
                "test-command",
                str(private),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertTrue((private / "command-ran").exists())
        self.assertEqual(
            (private / "artifact").stat().st_mtime_ns,
            (seed / "artifact").stat().st_mtime_ns,
        )


if __name__ == "__main__":
    unittest.main()
