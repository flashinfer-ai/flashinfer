"""
Copyright (c) 2026 by the PatchShift Conv3d contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "include/flashinfer/conv3d/patchshift"
HOST_ROOT = REPO_ROOT / "csrc/patchshift_conv3d"


KERNEL_INCLUDE_ORDER = [
    "mainloop.cuh",
    "cluster_b_c32.cuh",
    "c64_and_hybrid.cuh",
    "cluster_a_hybrid_c96.cuh",
    "cluster_b_c64.cuh",
    "cluster_a.cuh",
    "output_tail.cuh",
    "small_grid.cuh",
    "m32_c64_small_grid.cuh",
    "m32_d1_shallow_c64.cuh",
    "m64_c64_small_grid.cuh",
    "m64_cluster_b.cuh",
    "m64n128_micro_d1.cuh",
]


def test_kernel_umbrella_has_stable_include_order():
    umbrella = (CORE_ROOT / "kernels.cuh").read_text()
    positions = [
        umbrella.index(f"detail/kernels/{name}") for name in KERNEL_INCLUDE_ORDER
    ]
    assert positions == sorted(positions)


def test_compute_core_has_no_framework_or_model_dependencies():
    forbidden = (
        "torch/extension.h",
        "ATen/",
        "pybind11",
        "Wan",
        "VAE",
        "patchshift_3d_causal",
        "parse_options",
        "int main(",
    )
    for source in CORE_ROOT.rglob("*.cuh"):
        text = source.read_text()
        for token in forbidden:
            assert token not in text, (
                f"{source.relative_to(REPO_ROOT)} contains {token!r}"
            )


def test_process_terminating_cuda_helpers_are_not_imported():
    all_source = "\n".join(
        path.read_text()
        for root in (CORE_ROOT, HOST_ROOT)
        for path in root.rglob("*")
        if path.suffix in {".cuh", ".cu", ".h", ".inl"}
    )
    assert "std::exit" not in all_source
    assert "CUDA_DRIVER_CHECK" not in all_source
    assert "CUDA_CHECK" not in all_source


def test_host_only_files_stay_outside_public_include_tree():
    assert (HOST_ROOT / "tensor_maps.cuh").is_file()
    assert (HOST_ROOT / "select_policy.inl").is_file()
    assert not list(CORE_ROOT.rglob("*tensor_map*"))
    assert not list(CORE_ROOT.rglob("*select_policy*"))


def test_runtime_tensormap_updates_have_release_acquire_pair():
    launcher = (HOST_ROOT / "launcher.cu").read_text()
    common = (CORE_ROOT / "common.cuh").read_text()
    kernels = "\n".join(
        source.read_text() for source in (CORE_ROOT / "detail/kernels").glob("*.cuh")
    )

    assert "fence.proxy.tensormap::generic.release.gpu" in launcher
    assert "fence.proxy.tensormap::generic.acquire.gpu" in common
    assert "tma_descriptor_fence_acquire" in kernels
