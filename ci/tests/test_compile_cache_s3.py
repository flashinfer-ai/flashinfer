"""CPU-only tests of the compile cache's shared S3 layer (ci/compile_cache_s3.sh).

A local HTTP server with ETag support stands in for the bucket's anonymous
reads, and a fake ``aws`` on PATH stores uploads in the server's directory.
"""

import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import threading
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
S3_SCRIPT = REPO_ROOT / "ci/compile_cache_s3.sh"
IMAGE = "a" * 64
LANE = "H100/all"
JIT = "flashinfer/.cache/flashinfer/0.6.0/90a/cached_ops"
META = {"arch": "sm_90a", "cute_dsl_version": "4.7.0", "source_sha256": "a" * 64}


class _Bucket(http.server.SimpleHTTPRequestHandler):
    not_modified = 0
    denied = False

    def do_GET(self):
        if type(self).denied:
            self.send_error(403)
            return
        path = Path(self.directory) / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        etag = '"%s"' % hashlib.md5(data).hexdigest()
        if self.headers.get("If-None-Match") == etag:
            type(self).not_modified += 1
            self.send_response(304)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def write(path, data="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def cute_module(root, name, kernels, meta=META):
    module = root / JIT / f"{name}_sm90a_cute_dsl"
    write(module / "meta.json", json.dumps(meta))
    for kernel in kernels:
        write(module / f"{kernel}.o", f"{name}/{kernel}")
    return module


def members(path):
    with tarfile.open(path) as tar:
        return {m.name for m in tar.getmembers() if m.isfile()}


class S3LayerTestBase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bucket = self.root / "bucket"
        self.bucket.mkdir()
        _Bucket.not_modified = 0
        _Bucket.denied = False
        bucket = str(self.bucket)
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *a, **kw: _Bucket(*a, directory=bucket, **kw),
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_address[1]}/compile-cache/main"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.aws_log = self.root / "aws.log"
        self.tool(
            "aws",
            "#!/usr/bin/env bash\n"
            f'echo "$*" >> "{self.aws_log}"\n'
            'src="${@: -2:1}"; uri="${@: -1}"; key="${uri#s3://*/}"\n'
            f'mkdir -p "$(dirname "{bucket}/$key")" && cp "$src" "{bucket}/$key"\n',
        )

    def tool(self, name, contents):
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)

    def s3(self, *args, **env):
        result = subprocess.run(
            ["bash", str(S3_SCRIPT), *map(str, args)],
            env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", **env},
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def upload(self, src, **env):
        env.setdefault("AWS_ACCESS_KEY_ID", "test")
        return self.s3(
            "upload", "s3://bucket/compile-cache/main", IMAGE, LANE, src, **env
        )

    def tarball(self):
        return self.bucket / "compile-cache/main" / IMAGE / f"{LANE}.tar.gz"

    def publish(self, populate):
        """Upload a writer directory that ``populate`` fills."""
        writer = Path(tempfile.mkdtemp(dir=self.root))
        os.rmdir(writer)
        self.s3("restore", self.url, IMAGE, LANE, writer)
        populate(writer)
        self.assertIn("Uploaded", self.upload(writer))
        return writer


class CompileCacheS3ScriptTest(S3LayerTestBase):
    def test_writer_uploads_only_content_keyed_artifacts(self):
        def populate(root):
            cute_module(root, "gemm", ["k1", "k2"])
            write(root / JIT / "kda_tirx_abc/kernel.so")
            write(root / JIT / "kda_ptx/k.cubin")
            write(root / "flashinfer/.cache/flashinfer/cubins/abc/x.cubin")
            write(root / "triton/HASH/kernel.cubin")
            # Never shared: nvcc modules, tuning, generated sources, temporaries.
            write(root / JIT / "norm/build.ninja")
            write(root / JIT / "norm/norm.so")
            write(root / "flashinfer/.cache/flashinfer/0.6.0/90a/generated/x.cu")
            write(root / "flashinfer/.cache/flashinfer/autotune/x.json")
            write(root / JIT / "gemm_sm90a_cute_dsl/k3.o.tmp.123")
            write(root / JIT / "gemm_sm90a_cute_dsl/k3.lock")

        self.publish(populate)
        self.assertEqual(
            members(self.tarball()),
            {
                f"{JIT}/gemm_sm90a_cute_dsl/meta.json",
                f"{JIT}/gemm_sm90a_cute_dsl/k1.o",
                f"{JIT}/gemm_sm90a_cute_dsl/k2.o",
                f"{JIT}/kda_tirx_abc/kernel.so",
                f"{JIT}/kda_ptx/k.cubin",
                "flashinfer/.cache/flashinfer/cubins/abc/x.cubin",
                "triton/HASH/kernel.cubin",
            },
        )

    def test_writer_skips_upload_without_new_builds(self):
        self.publish(lambda root: cute_module(root, "gemm", ["k1"]))
        uploads = self.aws_log.read_text().count("s3 cp")
        second = self.root / "second"
        self.assertIn("Restored", self.s3("restore", self.url, IMAGE, LANE, second))
        self.assertEqual(
            (second / JIT / "gemm_sm90a_cute_dsl/k1.o").read_text(), "gemm/k1"
        )
        self.assertIn("compiled nothing new", self.upload(second))
        self.assertEqual(self.aws_log.read_text().count("s3 cp"), uploads)
        write(second / JIT / "gemm_sm90a_cute_dsl/k9.o")
        self.assertIn("Uploaded", self.upload(second))
        self.assertIn(f"{JIT}/gemm_sm90a_cute_dsl/k9.o", members(self.tarball()))

    def test_writer_refuses_a_directory_with_local_files(self):
        writer = self.root / "writer"
        write(writer / JIT / "planted_sm90a_cute_dsl/meta.json")
        out = self.s3("restore", self.url, IMAGE, LANE, writer)
        self.assertIn("is not empty", out)
        self.assertFalse((writer / ".s3-manifest").exists())
        self.assertIn("did not complete", self.upload(writer))
        self.assertFalse(self.aws_log.exists())

    def test_failed_restore_blocks_upload(self):
        # A writer that starts cold because the download failed must not
        # replace the lane's tarball with only its own builds.
        self.publish(lambda root: cute_module(root, "gemm", ["k1", "k2"]))
        published = self.tarball().read_bytes()
        writer = self.root / "writer"
        out = self.s3("restore", "http://127.0.0.1:9", IMAGE, LANE, writer)
        self.assertIn("will not upload", out)
        cute_module(writer, "norm", ["k1"])
        self.assertIn("did not complete", self.upload(writer))
        self.assertEqual(self.tarball().read_bytes(), published)

    def test_denied_restore_blocks_upload(self):
        # S3 answers 403 for a denied read of an existing tarball, so only a
        # 404 lets a cold writer publish the lane's first tarball.
        self.publish(lambda root: cute_module(root, "gemm", ["k1", "k2"]))
        published = self.tarball().read_bytes()
        writer = self.root / "writer"
        _Bucket.denied = True
        out = self.s3("restore", self.url, IMAGE, LANE, writer)
        _Bucket.denied = False
        self.assertIn("HTTP 403", out)
        self.assertIn("will not upload", out)
        cute_module(writer, "norm", ["k1"])
        self.assertIn("did not complete", self.upload(writer))
        self.assertEqual(self.tarball().read_bytes(), published)

    def test_same_size_rebuild_is_uploaded(self):
        self.publish(lambda root: cute_module(root, "gemm", ["k1"]))
        writer = self.root / "writer"
        self.s3("restore", self.url, IMAGE, LANE, writer)
        # A module rebuilt for a new source hash: same sizes, new contents.
        module = writer / JIT / "gemm_sm90a_cute_dsl"
        write(module / "meta.json", json.dumps({**META, "source_sha256": "d" * 64}))
        write(module / "k1.o", "gemm/k9")
        self.assertIn("Uploaded", self.upload(writer))

    def test_oversized_upload_keeps_only_this_jobs_builds(self):
        def populate(root):
            write(root / "triton/OLD/kernel.cubin", os.urandom(3 << 20))
            cute_module(root, "gemm", ["k1"])

        self.publish(populate)
        second = self.root / "second"
        self.s3("restore", self.url, IMAGE, LANE, second)
        write(second / JIT / "gemm_sm90a_cute_dsl/k2.o")
        out = self.upload(second, CI_COMPILE_CACHE_S3_MAX_MB="1")
        self.assertIn("uploading only this job's builds", out)
        self.assertEqual(
            members(self.tarball()),
            {f"{JIT}/gemm_sm90a_cute_dsl/meta.json", f"{JIT}/gemm_sm90a_cute_dsl/k2.o"},
        )

    def test_fetch_keeps_a_mirror_current(self):
        mirror = self.root / "mirror"
        self.assertIn("No shared", self.s3("fetch", self.url, IMAGE, LANE, mirror))
        self.assertFalse((mirror / "tree").exists())

        writer = self.publish(lambda root: cute_module(root, "gemm", ["k1"]))
        self.s3("fetch", self.url, IMAGE, LANE, mirror)
        self.assertTrue((mirror / "tree" / JIT / "gemm_sm90a_cute_dsl/k1.o").exists())
        self.assertIn("unchanged", self.s3("fetch", self.url, IMAGE, LANE, mirror))
        self.assertEqual(_Bucket.not_modified, 1)

        write(writer / JIT / "gemm_sm90a_cute_dsl/k2.o")
        self.upload(writer)
        self.s3("fetch", self.url, IMAGE, LANE, mirror)
        self.assertTrue((mirror / "tree" / JIT / "gemm_sm90a_cute_dsl/k2.o").exists())

        # An expired object takes the mirror with it.
        self.tarball().unlink()
        self.s3("fetch", self.url, IMAGE, LANE, mirror)
        self.assertFalse((mirror / "tree").exists())
        self.assertFalse(list(mirror.glob(".fetch.*")))

    def test_merge_fills_gaps_without_mixing_modules(self):
        src, dest = self.root / "src", self.root / "dest"
        cute_module(src, "same", ["k1", "k2"])
        cute_module(
            src, "stale", ["k1", "k2"], meta={**META, "source_sha256": "b" * 64}
        )
        cute_module(src, "missing", ["k1"])
        write(src / "triton/A/kernel.cubin", "src")
        write(src / "triton/B/kernel.cubin", "src")
        write(cute_module(dest, "same", ["k1"]) / "k1.o", "dest")
        cute_module(dest, "stale", ["k1"])
        write(dest / "triton/A/kernel.cubin", "dest")
        self.s3("merge", src, dest)

        same = dest / JIT / "same_sm90a_cute_dsl"
        self.assertEqual((same / "k1.o").read_text(), "dest")
        self.assertEqual((same / "k2.o").read_text(), "same/k2")
        # A module whose meta.json differs gets nothing: its objects would be
        # loaded under the other meta.json.
        stale = dest / JIT / "stale_sm90a_cute_dsl"
        self.assertFalse((stale / "k2.o").exists())
        self.assertEqual(json.loads((stale / "meta.json").read_text()), META)
        self.assertTrue((dest / JIT / "missing_sm90a_cute_dsl/k1.o").exists())
        self.assertEqual((dest / "triton/A/kernel.cubin").read_text(), "dest")
        self.assertEqual((dest / "triton/B/kernel.cubin").read_text(), "src")
        self.assertTrue((src / "triton/B/kernel.cubin").exists())

    def test_invalid_image_or_lane_is_skipped(self):
        for image, lane in (
            (IMAGE, "../escape"),
            (IMAGE, "H100/.."),
            (IMAGE, "a b"),
            (IMAGE, "/abs"),
            ("..", LANE),
            ("a/b", LANE),
        ):
            with self.subTest(image=image, lane=lane):
                out = self.s3("restore", self.url, image, lane, self.root / "dir")
                self.assertIn("invalid image", out)
                self.assertFalse((self.root / "dir").exists())

    def test_failures_only_warn(self):
        # Nothing listens on port 9; the job continues cold.
        out = self.s3("restore", "http://127.0.0.1:9", IMAGE, LANE, self.root / "dir")
        self.assertIn("::warning", out)
        write(self.root / "writer" / "triton/HASH/kernel.cubin")
        out = self.upload(self.root / "writer", AWS_ACCESS_KEY_ID="")
        self.assertIn("no AWS credentials", out)
        self.assertFalse(self.aws_log.exists())


class CompileCacheS3IntegrationTest(S3LayerTestBase):
    """ci/bash.sh with a mocked Docker, as in test_compile_cache.py."""

    def setUp(self):
        super().setUp()
        self.cache = self.root / "cache"
        self.docker_log = self.root / "docker.jsonl"
        self.tool(
            "docker",
            "#!/usr/bin/python3\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['MOCK_DOCKER_LOG'], 'a') as log:\n"
            "    log.write(json.dumps(args) + '\\n')\n"
            "if args[:2] == ['image', 'inspect']:\n"
            f"    print('sha256:{IMAGE}')\n",
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("CI_COMPILE_", "MOCK_"))
        }
        self.env.update(
            PATH=f"{self.bin}:/usr/bin:/bin",
            CI_PIP_CACHE_DIR="",
            CI_COMPILE_CACHE_DIR=str(self.cache),
            CI_COMPILE_CACHE_MIN_FREE_MB="0",
            CI_COMPILE_CACHE_RUNNER="runner-a",
            CI_COMPILE_CACHE_S3_URL=self.url,
            CI_COMPILE_CACHE_S3_LANE=LANE,
            MOCK_DOCKER_LOG=str(self.docker_log),
        )

    def run_job(self):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "ci/bash.sh"), "org/image:tag", "true"],
            env=self.env,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run = [json.loads(line) for line in self.docker_log.read_text().splitlines()]
        return result.stdout, run[-1]

    def runner_dir(self):
        key = hashlib.sha256(b"runner-a").hexdigest()
        return self.cache / f"runner-{key}"

    def test_writer_bypasses_and_retires_the_seeds(self):
        self.publish(lambda root: cute_module(root, "gemm", ["k1"]))
        seed = write(self.runner_dir() / "images" / IMAGE / "triton/planted")
        writer = self.root / "writer"
        write(writer / "leftover")
        self.env.update(
            CI_COMPILE_CACHE_MODE="read-write", CI_COMPILE_CACHE_WRITER_DIR=str(writer)
        )
        out, run = self.run_job()
        self.assertIn("mode=s3-writer", out)
        self.assertIn(f"{writer}:/compile-cache", run)
        self.assertFalse(any("/compile-cache-seed" in arg for arg in run))
        self.assertFalse((writer / "leftover").exists())
        self.assertTrue((writer / JIT / "gemm_sm90a_cute_dsl/k1.o").exists())
        self.assertFalse((writer / "triton/planted").exists())
        # Seeds stop growing once writers exist; the S3 mirror replaces them.
        self.assertFalse(seed.exists())

    def test_pull_request_jobs_mount_the_mirror(self):
        out, run = self.run_job()
        self.assertIn("No shared compile cache", out)
        self.assertNotIn("COMPILE_CACHE_S3_TREE=/compile-cache-s3", run)

        self.publish(lambda root: cute_module(root, "gemm", ["k1"]))
        out, run = self.run_job()
        tree = self.runner_dir() / "s3" / IMAGE / "H100_all" / "tree"
        self.assertIn(f"{tree}:/compile-cache-s3:ro", run)
        self.assertIn("COMPILE_CACHE_S3_TREE=/compile-cache-s3", run)
        self.assertTrue((tree / JIT / "gemm_sm90a_cute_dsl/k1.o").exists())
        out, _ = self.run_job()
        self.assertIn("unchanged", out)

    def test_entrypoint_prefers_s3_and_fills_gaps_from_the_seed(self):
        s3_tree, seed, private = (self.root / n for n in ("s3", "seed", "private"))
        cute_module(s3_tree, "gemm", ["k1"])
        cute_module(
            seed, "gemm", ["k1", "k2"], meta={**META, "source_sha256": "c" * 64}
        )
        cute_module(seed, "norm", ["k1"])
        private.mkdir()
        # The real entrypoint finds compile_cache_s3.sh beside itself.
        docker_dir = self.root / "docker"
        docker_dir.mkdir()
        for name in ("compile_cache_entrypoint.sh", "compile_cache_s3.sh"):
            shutil.copy(REPO_ROOT / "ci" / name, docker_dir / name)
        result = subprocess.run(
            [
                "bash",
                str(docker_dir / "compile_cache_entrypoint.sh"),
                seed,
                private,
                "true",
            ],
            env={**os.environ, "COMPILE_CACHE_S3_TREE": str(s3_tree)},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        gemm = private / JIT / "gemm_sm90a_cute_dsl"
        self.assertEqual(json.loads((gemm / "meta.json").read_text()), META)
        self.assertFalse((gemm / "k2.o").exists())
        self.assertTrue((private / JIT / "norm_sm90a_cute_dsl/k1.o").exists())


if __name__ == "__main__":
    unittest.main()
