"""
Copyright (c) 2023 by FlashInfer team.

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
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple, Union, Optional
import torch.distributed as dist

import torch
from einops import rearrange, reduce, repeat

from flashinfer.utils import round_up


def _ceil_to_ue8m0(x: torch.Tensor):
    """imported from DeepGEMM"""
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """imported from DeepGEMM"""
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """imported from DeepGEMM"""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (round_up(m, 128), round_up(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def quantize_fp8(x, scale_shape, tile_shape, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(x_scale, "s0 s1 -> (s0 t0) (s1 t1)", t0=t0, t1=t1)
        else:
            # Handle column-major tiling
            x_tiled = rearrange(x, "(s1 t0) (s0 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 -> s1 s0")
            scales_repeated = repeat(
                scales_permuted, "s1 s0 -> (s1 t0) (s0 t1)", t0=t0, t1=t1
            )

    elif ndim == 3:
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(
                x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(
                x_scale, "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)", t0=t0, t1=t1, t2=t2
            )
        else:
            # Handle layout where the last two axes are swapped
            x_tiled = rearrange(
                x, "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1")
            scales_repeated = repeat(
                scales_permuted,
                "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)",
                t0=t0,
                t1=t1,
                t2=t2,
            )

    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_fp32 = x / (scales_repeated + 1e-8)

    # Convert the result to the target FP8 format
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, x_scale


def dequantize_fp8(x, x_scale, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(x_scale.shape)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        if scale_major_mode == "K":
            s0, s1 = x_scale.shape
        else:
            s1, s0 = x_scale.shape
        x = rearrange(
            x.to(torch.float32), "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 -> s0 s1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 -> s1 s0 1 1")
        out = rearrange(x * x_scale, "s0 s1 t0 t1 -> (s0 t0) (s1 t1)")

    elif ndim == 3:
        if scale_major_mode == "K":
            s0, s1, s2 = x_scale.shape
        else:
            s0, s2, s1 = x_scale.shape
        x = rearrange(
            x.to(torch.float32),
            "(s0 t0) (s1 t1) (s2 t2)-> s0 s1 s2 t0 t1 t2",
            s0=s0,
            s1=s1,
            s2=s2,
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s1 s2 1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1 1 1 1")
        out = rearrange(x * x_scale, "s0 s1 s2 t0 t1 t2 -> (s0 t0) (s1 t1) (s2 t2)")
    return out


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# copy and modified from DeepGEMM and DeepEP
def bench_kineto(fn, kernel_names: Union[str, tuple], num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = False, flush_l2: bool = True,
                 num_kernels_per_period: int = 1):
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get('DG_NSYS_PROFILING', 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output and not using_nsys else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) if not using_nsys else None
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) if not using_nsys else empty_suppress()
        with profiler:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    # print(f"prof_lines=\n" + "\n".join(prof_lines))

    # Save chrome traces
    if trace_path is not None:
        print(f"export_chrome_trace to {trace_path=}")
        profiler.export_chrome_trace(trace_path)

    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    kernel_durations = [None] * len(kernel_names)

    # Expand the kernels by periods
    with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
        profiler.export_chrome_trace(tmp.name)
        profile_data = json.loads(Path(tmp.name).read_text())

    for i, kernel_name in enumerate(kernel_names):
        events = [event for event in profile_data['traceEvents'] if kernel_name in event['name']]
        events = sorted(events, key=lambda event: event['ts'])
        durations = [event['dur'] / 1e6 for event in events]
        assert len(durations) % num_kernels_per_period == 0
        num_kernel_patterns = len(durations) // num_kernels_per_period
        kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                               for j in range(num_kernels_per_period)]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]
