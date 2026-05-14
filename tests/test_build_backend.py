import csv
import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import build_backend


class BuildBackendTest(unittest.TestCase):
    def test_selected_cutlass_dsl_requirement_uses_cuda_13_override(self):
        with mock.patch.dict(
            os.environ,
            {"FLASHINFER_CUDA_VERSION": "13.0"},
            clear=False,
        ):
            self.assertEqual(
                build_backend._selected_cutlass_dsl_requirement(),
                "nvidia-cutlass-dsl[cu13]>=4.5.0",
            )

    def test_selected_cutlass_dsl_requirement_uses_cuda_version_env(self):
        with mock.patch.dict(
            os.environ,
            {"CUDA_VERSION": "cu130"},
            clear=True,
        ):
            self.assertEqual(
                build_backend._selected_cutlass_dsl_requirement(),
                "nvidia-cutlass-dsl[cu13]>=4.5.0",
            )

    def test_selected_cutlass_dsl_requirement_defaults_to_base_when_cuda_unknown(
        self,
    ):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                build_backend, "_detect_cuda_version_from_nvcc", return_value=None
            ),
            mock.patch.object(
                build_backend, "_detect_cuda_version_from_torch", return_value=None
            ),
        ):
            self.assertEqual(
                build_backend._selected_cutlass_dsl_requirement(),
                "nvidia-cutlass-dsl>=4.5.0",
            )

    def test_invalid_flashinfer_cuda_version_fails_clearly(self):
        with (
            mock.patch.dict(
                os.environ,
                {"FLASHINFER_CUDA_VERSION": "not-cuda"},
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "FLASHINFER_CUDA_VERSION"),
        ):
            build_backend._selected_cutlass_dsl_requirement()

    def test_patch_metadata_content_replaces_only_unmarked_cutlass_dependency(self):
        with mock.patch.dict(
            os.environ,
            {"FLASHINFER_CUDA_VERSION": "cu130"},
            clear=False,
        ):
            metadata = "\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: flashinfer-python",
                    "Requires-Dist: numpy",
                    "Requires-Dist: nvidia-cutlass-dsl>=4.5.0",
                    "Requires-Dist: nvidia-cutlass-dsl[dev]>=4.5.0",
                    'Requires-Dist: nvidia-cutlass-dsl>=4.5.0; extra == "cu12"',
                    'Requires-Dist: nvidia-cutlass-dsl[cu13]>=4.5.0; extra == "cu13"',
                    "",
                ]
            )

            patched = build_backend._patch_metadata_content(metadata)

        self.assertIn("Requires-Dist: nvidia-cutlass-dsl[cu13]>=4.5.0\n", patched)
        self.assertIn("Requires-Dist: nvidia-cutlass-dsl[dev]>=4.5.0", patched)
        self.assertIn(
            'Requires-Dist: nvidia-cutlass-dsl>=4.5.0; extra == "cu12"',
            patched,
        )
        self.assertIn(
            'Requires-Dist: nvidia-cutlass-dsl[cu13]>=4.5.0; extra == "cu13"',
            patched,
        )

    def test_patch_metadata_content_preserves_crlf_and_invalid_requirements(self):
        with mock.patch.dict(
            os.environ,
            {"FLASHINFER_CUDA_VERSION": "cu130"},
            clear=False,
        ):
            metadata = "\r\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: flashinfer-python",
                    "Requires-Dist: ???",
                    "Requires-Dist: nvidia-cutlass-dsl>=4.5.0",
                    "",
                ]
            )

            patched = build_backend._patch_metadata_content(metadata)

        self.assertIn("Requires-Dist: ???\r\n", patched)
        self.assertIn("Requires-Dist: nvidia-cutlass-dsl[cu13]>=4.5.0\r\n", patched)

    def test_patch_wheel_metadata_updates_record(self):
        with (
            mock.patch.dict(
                os.environ,
                {"FLASHINFER_CUDA_VERSION": "13.0"},
                clear=False,
            ),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            wheel_path = Path(tmp_dir) / "flashinfer_python-0.0.0-py3-none-any.whl"
            dist_info = "flashinfer_python-0.0.0.dist-info"
            metadata_name = f"{dist_info}/METADATA"
            record_name = f"{dist_info}/RECORD"
            large_name = "flashinfer/data/large.bin"
            large_content = b"x" * 1024
            metadata = "\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: flashinfer-python",
                    "Requires-Dist: nvidia-cutlass-dsl>=4.5.0",
                    "",
                ]
            ).encode()
            record = "\n".join(
                [
                    "flashinfer/__init__.py,,",
                    f"{large_name},,",
                    f"{metadata_name},sha256=old,1",
                    f"{record_name},,",
                    "",
                ]
            ).encode()

            with zipfile.ZipFile(wheel_path, "w") as wheel:
                wheel.writestr("flashinfer/__init__.py", b"")
                wheel.writestr(large_name, large_content)
                wheel.writestr(metadata_name, metadata)
                wheel.writestr(record_name, record)

            original_read = zipfile.ZipFile.read

            def guarded_read(self, name, pwd=None):
                filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
                if filename == large_name:
                    raise AssertionError("large wheel entries should be streamed")
                return original_read(self, name, pwd)

            with mock.patch.object(zipfile.ZipFile, "read", guarded_read):
                build_backend._patch_wheel_metadata(wheel_path)

            with zipfile.ZipFile(wheel_path, "r") as wheel:
                patched_metadata = wheel.read(metadata_name)
                patched_record = wheel.read(record_name)
                patched_large_content = wheel.read(large_name)

        self.assertIn(
            b"Requires-Dist: nvidia-cutlass-dsl[cu13]>=4.5.0",
            patched_metadata,
        )
        self.assertEqual(patched_large_content, large_content)

        rows = list(csv.reader(io.StringIO(patched_record.decode())))
        metadata_rows = [row for row in rows if row and row[0] == metadata_name]
        self.assertEqual(len(metadata_rows), 1)
        self.assertNotEqual(metadata_rows[0][1], "sha256=old")
        self.assertEqual(metadata_rows[0][2], str(len(patched_metadata)))


if __name__ == "__main__":
    unittest.main()
