"""Source-built Cake selective-state-update programs for Blackwell."""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from filelock import FileLock
from tvm_ffi import cpp

from .. import env as jit_env


class _Program(NamedTuple):
    module_ident: str
    flags: tuple[str, ...]
    device_sha256: str
    host_sha256: str


_PROGRAMS = {
    "stp_bf16_direct": _Program(
        "cake_selective_state_update_stp_bf16_direct_7f536ad937",
        ("--fmad=false",),
        "e87516043c3a8880a3a9139be932ace585fde8d8fca465b39dbe2f6f1dcc5200",
        "87df4de19f02f9c70e271acddf45af1ffe47c0e9bf4b3230a2e50ccee93b5f2a",
    ),
    "stp_bf16_ratio8": _Program(
        "cake_selective_state_update_stp_bf16_ratio8_2b01869233",
        ("--fmad=false",),
        "9f538e45b1c1be86e31ad9bd6a4d4c321d73f5f8dfa7a576b1b7f0f936144c13",
        "ec5215a667fb665721b25c1270e390be701d2bbc3c027af8acf36bf2d46df57c",
    ),
    "stp_bf16_ratio8_saturated": _Program(
        "cake_selective_state_update_stp_bf16_ratio8_saturated_6caa473c10",
        ("--fmad=false",),
        "58a7c2aae2cc33c72d19fc0be2772298ae8148e7da64609678aab0cb119429ae",
        "cf78dd2810ccc922dcd7eae9c4f0e084521224eb5d9dc8da569158cb820daa27",
    ),
    "stp_bf16_ratio16": _Program(
        "cake_selective_state_update_stp_bf16_ratio16_f06a928344",
        ("--fmad=false",),
        "aaf674e38a63a774f87b6119beebfa0eafb59e30f46259e0fd5de0b0e26b6929",
        "e352602e0685e8449e392569c8ffd499c101b0c9741a45c9090809529a4220cb",
    ),
    "stp_bf16_persistent": _Program(
        "cake_selective_state_update_stp_bf16_persistent_49ebcdd724",
        ("--fmad=false",),
        "c6faebe59e862775a69866a7917a769d1fa9f848319597414f87ecfef09e34b8",
        "7a42d56af9cc621ac5a08985407f402ded8c38e1477bdd64fc92ed826ce106af",
    ),
    "stp_fp32_identity": _Program(
        "cake_selective_state_update_stp_fp32_identity_0aa3025026",
        (),
        "1406f399d602b93a5989d81ad8f13cc9999629521eed7a4cf9112ad3266a05a1",
        "7e87cd746ee04f6fb991d2abea1e5404facc90b78a1656bbf9830e3531d1722d",
    ),
    "mtp_short": _Program(
        "cake_selective_state_update_mtp_short_4262bee913",
        ("--use_fast_math",),
        "b1f7283802a93ec67ff56e2b4c35ebf751ff4352e1ac34c7c4c1f1db2325a5a6",
        "516c82fd9b47fb23e5d89267629a2be2a34d26682fdcb82db3c0fa215dbd7f59",
    ),
    "mtp_cache_bf16_c4_t6": _Program(
        "cake_selective_state_update_mtp_cache_bf16_c4_t6_829517a093",
        (),
        "babe752c7baf690ec9eee6d799e5a4d7e7ae80fb5626246b55c6d637fc19c885",
        "573f0a9c88675540b344726fe42f71d1cce226b4df49662fbc9066c17b7c51f8",
    ),
    "mtp_horizontal": _Program(
        "cake_selective_state_update_mtp_horizontal_65ba035815",
        ("--fmad=false",),
        "006c640d61cf051858aca291ec42584868e68239f393334c301c6a701cc66739",
        "ba30249d5f5b7ec0436817d39d835a2a82f3b266289805af742b144b57614e43",
    ),
    "dynamic_checkpoint0": _Program(
        "cake_selective_state_update_dynamic_checkpoint0_3d22b73c40",
        ("--fmad=false",),
        "3ddf244fee8b8dc337daed2b1dc6d6149d723e74e997e273e1b0df51bb04d074",
        "685fe8d716410fda420e34a8fe7c8c7622e0ad2d791e2298891b9b83afd0976b",
    ),
    "dynamic_checkpoint1": _Program(
        "cake_selective_state_update_dynamic_checkpoint1_36de5e4ac0",
        ("--fmad=false",),
        "8e5855b3e13de163cf31c8631b7287991a5336a67a969e3057f81ac9187d9a8c",
        "feae411915defaad27320b17b592c9cd9171ed53b617ed1e16f789ea49f40cf3",
    ),
    "dynamic_checkpoint3": _Program(
        "cake_selective_state_update_dynamic_checkpoint3_a1482464e7",
        ("--fmad=false",),
        "9abb9ec7df41e496372a72fdc6287f8cb48277eccb1c374c9c71dc7ce05b68e3",
        "1892ca1e2203b0319ae60d144cccb58612586c0e7cd961783510e7a263137115",
    ),
    "dynamic_checkpoint7": _Program(
        "cake_selective_state_update_dynamic_checkpoint7_b2c8658514",
        ("--fmad=false",),
        "7923af7ed1fc25d56b4d8e11bde4ed7348c15ac4090e6307790a3dfcbed9e479",
        "63f5ce57a7ff512d3f4cde15e25aafb22558b258995ce2b880ad5d14ac34d747",
    ),
}


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_selective_state_update"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[3] / "csrc" / "cake_selective_state_update"


def _target_arch(device: Optional[torch.device] = None) -> str:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 0):
        return "sm_100a"
    if capability == (10, 3):
        return "sm_103a"
    raise ValueError(
        "Cake selective_state_update requires SM100 or SM103, got "
        f"SM{capability[0]}{capability[1]}"
    )


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build the Cake backend")
    return Path(candidate).resolve()


@functools.cache
def _load_program(name: str, arch: str, device_index: int):
    program = _PROGRAMS[name]
    source_dir = _source_dir()
    device_source = source_dir / "cuda" / f"cake_selective_state_update_{name}.cu"
    host_source = source_dir / "host" / f"cake_selective_state_update_{name}.cc"
    for path, expected in (
        (device_source, program.device_sha256),
        (host_source, program.host_sha256),
    ):
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"Cake source checksum mismatch: {path}")

    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(device_source.read_bytes())
    digest.update(host_source.read_bytes())
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    module_name = f"cake_selective_state_update_{name}_{arch}_{digest.hexdigest()[:16]}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / f"{program.module_ident}.cubin"
    with FileLock(build_dir / f"{program.module_ident}.lock", thread_local=False):
        if not cubin_path.is_file():
            temporary = build_dir / f"{program.module_ident}.{os.getpid()}.tmp.cubin"
            command = [
                str(nvcc),
                "-cubin",
                f"-arch={arch}",
                "--std=c++17",
                "-O3",
                "-I",
                str(nvcc.parent.parent / "include"),
                *program.flags,
                str(device_source),
                "-o",
                str(temporary),
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            if process.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Cake nvcc failed for {name} ({arch}):\n{process.stderr}"
                )
            os.replace(temporary, cubin_path)

        return cpp.load_inline(
            module_name,
            cpp_sources=host_source.read_text(encoding="utf-8"),
            embed_cubin={program.module_ident: cubin_path.read_bytes()},
            extra_include_paths=[str(nvcc.parent.parent / "include")],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )


def _compact_broadcasts(
    dt: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dt.stride(-1) != 0 or A.stride(-1) != 0 or A.stride(-2) != 0:
        raise ValueError("Cake requires broadcast dt and A coefficient axes")
    if D.stride(-1) != 0 or dt_bias.stride(-1) != 0:
        raise ValueError("Cake requires broadcast D and dt_bias dimensions")
    dt_shape = dt.shape[:-1]
    return (
        dt.as_strided(dt_shape, dt.stride()[:-1]),
        A.as_strided((A.shape[0],), (A.stride(0),)),
        D.as_strided((D.shape[0],), (D.stride(0),)),
        dt_bias.as_strided((dt_bias.shape[0],), (dt_bias.stride(0),)),
    )


def _balanced_worker_count(work: int, cap: int) -> int:
    if work <= cap:
        return work
    trips = (work + cap - 1) // cap
    return (work + trips - 1) // trips


def _is_sglang_raw_mtp_cache_layout(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    output: torch.Tensor,
    state_batch_indices: torch.Tensor,
    dst_state_batch_indices: Optional[torch.Tensor],
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
) -> bool:
    """Recognize the unmaterialized Granite target-verify TensorViews."""

    if x.ndim != 4 or not 1 <= x.shape[0] <= 4:
        return False
    batch_size = x.shape[0]
    if (
        state.dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or dt.dtype != torch.bfloat16
        or A.dtype != torch.float32
        or B.dtype != torch.bfloat16
        or C.dtype != torch.bfloat16
        or D.dtype != torch.bfloat16
        or dt_bias.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or state_batch_indices.dtype != torch.int32
        or intermediate_states_buffer is None
        or intermediate_states_buffer.dtype != torch.bfloat16
        or intermediate_state_indices is None
        or intermediate_state_indices.dtype != torch.int32
        or dst_state_batch_indices is not None
    ):
        return False
    if (
        tuple(state.shape[1:]) != (64, 64, 128)
        or tuple(x.shape) != (batch_size, 6, 64, 64)
        or tuple(dt.shape) != (batch_size, 6, 64, 64)
        or tuple(A.shape) != (64, 64, 128)
        or tuple(B.shape) != (batch_size, 6, 1, 128)
        or tuple(C.shape) != tuple(B.shape)
        or tuple(D.shape) != (64, 64)
        or tuple(dt_bias.shape) != (64, 64)
        or tuple(output.shape) != tuple(x.shape)
        or tuple(intermediate_states_buffer.shape[1:]) != (6, 64, 64, 128)
        or intermediate_states_buffer.shape[0] < batch_size
        or tuple(state_batch_indices.shape) != (batch_size,)
        or tuple(intermediate_state_indices.shape) != (batch_size,)
    ):
        return False
    if (
        tuple(x.stride()) != (26112, 4352, 64, 1)
        or tuple(dt.stride()) != (51072, 8512, 1, 0)
        or tuple(A.stride()) != (1, 0, 0)
        or tuple(B.stride()) != (26112, 4352, 128, 1)
        or tuple(C.stride()) != (26112, 4352, 128, 1)
        or tuple(D.stride()) != (1, 0)
        or tuple(dt_bias.stride()) != (1, 0)
    ):
        return False
    if not (
        state.is_contiguous()
        and output.is_contiguous()
        and state_batch_indices.is_contiguous()
        and intermediate_states_buffer.is_contiguous()
        and intermediate_state_indices.is_contiguous()
    ):
        return False
    return all(tensor.data_ptr() % 16 == 0 for tensor in (x, B, C))


def _launch_stp_bf16(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: torch.Tensor,
    output: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    ngroups: int,
    dt_softplus: bool,
    disable_state_update: bool,
    arch: str,
    device_index: int,
) -> None:
    batch_size, nheads, _ = x.shape
    heads_per_group = nheads // ngroups
    head_tiles = (heads_per_group + 3) // 4
    total_head_tiles = batch_size * ngroups * head_tiles
    num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    direct_cap = 9 * 3 * num_sms
    dt_compact, A_compact, D_compact, bias_compact = _compact_broadcasts(
        dt, A, D, dt_bias
    )
    z_arg = x if z is None else z
    common = (
        state,
        x,
        dt_compact,
        A_compact,
        B,
        C,
        D_compact,
        z_arg,
        bias_compact,
        output,
        source,
        destination,
        nheads,
        ngroups,
        head_tiles,
    )
    if total_head_tiles > direct_cap:
        worker_count = _balanced_worker_count(total_head_tiles, 3 * num_sms)
        _load_program("stp_bf16_persistent", arch, device_index).run(
            *common,
            total_head_tiles,
            int(dt_softplus),
            int(z is not None),
            int(disable_state_update),
            worker_count,
            1,
            1,
        )
        return

    if heads_per_group == 16:
        program = "stp_bf16_ratio16"
    elif heads_per_group == 8 and total_head_tiles >= 2048:
        program = "stp_bf16_ratio8_saturated"
    elif heads_per_group == 8:
        program = "stp_bf16_ratio8"
    else:
        program = "stp_bf16_direct"
    _load_program(program, arch, device_index).run(
        *common,
        int(dt_softplus),
        int(z is not None),
        int(disable_state_update),
        total_head_tiles,
        1,
        1,
    )


def try_cake_selective_state_update(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    output: torch.Tensor,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    state_scale: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: str,
    dt_softplus: bool,
) -> bool:
    raw_sglang_layout = (
        dt_bias is not None
        and state_batch_indices is not None
        and _is_sglang_raw_mtp_cache_layout(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            output=output,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
        )
    )
    legacy_types = (
        state_batch_indices is not None
        and state_batch_indices.dtype == torch.int64
        and dt.dtype == torch.float32
        and A.dtype == torch.float32
        and D.dtype == torch.float32
        and dt_bias is not None
        and dt_bias.dtype == torch.float32
        and (
            dst_state_batch_indices is None
            or dst_state_batch_indices.dtype == torch.int64
        )
        and (
            intermediate_state_indices is None
            or intermediate_state_indices.dtype == torch.int64
        )
    )
    if (
        dt_bias is None
        or state_batch_indices is None
        or state_scale is not None
        or intermediate_state_scales is not None
        or rand_seed is not None
        or cu_seqlens is not None
        or num_accepted_tokens is not None
        or state.ndim != 4
        or state.dtype not in (torch.bfloat16, torch.float32)
        or x.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or C.dtype != torch.bfloat16
        or not (legacy_types or raw_sglang_layout)
    ):
        return False
    device_index = state.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    try:
        arch = _target_arch(state.device)
    except ValueError:
        return False

    batch_size = x.shape[0]
    nheads = state.shape[1]
    dim = state.shape[2]
    dstate = state.shape[3]
    if B.shape[-2] <= 0 or nheads % B.shape[-2]:
        return False
    ngroups = B.shape[-2]

    with torch.cuda.device(device_index):
        if (
            x.ndim == 3
            and cache_steps == 0
            and (dim, dstate) == (128, 128)
            and state_batch_indices.ndim == 1
        ):
            destination = (
                state_batch_indices
                if dst_state_batch_indices is None
                else dst_state_batch_indices
            )
            if destination.ndim != 1:
                return False
            if state.dtype == torch.bfloat16:
                _launch_stp_bf16(
                    state=state,
                    x=x,
                    dt=dt,
                    A=A,
                    B=B,
                    C=C,
                    D=D,
                    z=z,
                    dt_bias=dt_bias,
                    output=output,
                    source=state_batch_indices,
                    destination=destination,
                    ngroups=ngroups,
                    dt_softplus=dt_softplus,
                    disable_state_update=disable_state_update,
                    arch=arch,
                    device_index=device_index,
                )
                return True
            if (
                z is None
                and not dt_softplus
                and not disable_state_update
                and destination.data_ptr() == state_batch_indices.data_ptr()
                and batch_size * nheads
                >= 8
                * torch.cuda.get_device_properties(device_index).multi_processor_count
            ):
                _load_program("stp_fp32_identity", arch, device_index).run(
                    state,
                    x,
                    dt.data_ptr(),
                    A.data_ptr(),
                    B,
                    C,
                    D.data_ptr(),
                    x,
                    dt_bias.data_ptr(),
                    output,
                    state_batch_indices,
                    destination,
                    nheads,
                    ngroups,
                    1,
                    state.stride(0),
                    dt.stride(0),
                    dt.stride(1),
                    A.stride(0),
                    D.stride(0),
                    dt_bias.stride(0),
                    0,
                    0,
                    0,
                    pad_slot_id,
                    batch_size * nheads,
                    1,
                    1,
                )
                return True

        if x.ndim != 4:
            return False
        token_steps = x.shape[1]
        if (
            state.dtype == torch.bfloat16
            and (dim, dstate) == (128, 128)
            and token_steps in (1, 2)
            and z is None
            and dst_state_batch_indices is None
            and intermediate_states_buffer is None
            and not disable_state_update
            and state_batch_indices.ndim == 1
        ):
            try:
                dt_compact, A_compact, D_compact, bias_compact = (
                    _compact_broadcasts(dt, A, D, dt_bias)
                )
            except ValueError:
                return False
            _load_program("mtp_short", arch, device_index).run(
                state,
                x,
                dt_compact,
                A_compact,
                B,
                C,
                D_compact,
                bias_compact,
                output,
                state_batch_indices,
                batch_size,
                nheads,
                dim,
                dstate,
                ngroups,
                token_steps,
                state.stride(0),
                int(dt_softplus),
                pad_slot_id,
                batch_size * nheads,
                1,
                1,
            )
            return True

        if (
            state.dtype == torch.bfloat16
            and (dim, dstate, token_steps) == (64, 128, 6)
            and z is None
            and dst_state_batch_indices is None
            and disable_state_update
            and intermediate_states_buffer is not None
            and intermediate_state_indices is not None
            and state_batch_indices.ndim == 1
        ):
            total_tiles = batch_size * nheads
            if raw_sglang_layout:
                _load_program(
                    "mtp_cache_bf16_c4_t6_sglang_raw", arch, device_index
                ).run(
                    state,
                    x,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    dt_bias,
                    output,
                    state_batch_indices,
                    intermediate_states_buffer,
                    intermediate_state_indices,
                    nheads,
                    ngroups,
                    state.stride(0),
                    intermediate_states_buffer.stride(0),
                    pad_slot_id,
                    batch_size,
                    nheads,
                    4,
                )
                return True
            try:
                dt_compact, A_compact, D_compact, bias_compact = (
                    _compact_broadcasts(dt, A, D, dt_bias)
                )
            except ValueError:
                return False
            num_sms = torch.cuda.get_device_properties(
                device_index
            ).multi_processor_count
            if batch_size < 32 and (num_sms * 10) // total_tiles >= 4:
                _load_program("mtp_cache_bf16_c4_t6", arch, device_index).run(
                    state,
                    x,
                    dt_compact,
                    A_compact,
                    B,
                    C,
                    D_compact,
                    bias_compact,
                    output,
                    state_batch_indices,
                    intermediate_states_buffer,
                    intermediate_state_indices,
                    nheads,
                    ngroups,
                    state.stride(0),
                    intermediate_states_buffer.stride(0),
                    pad_slot_id,
                    batch_size,
                    nheads,
                    4,
                )
                return True
            if batch_size >= 32 and algorithm == "horizontal":
                work_groups = (total_tiles + 1) // 2
                workers = _balanced_worker_count(work_groups, 2 * num_sms)
                _load_program("mtp_horizontal", arch, device_index).run(
                    state,
                    x,
                    B,
                    C,
                    dt_compact,
                    A_compact,
                    D_compact,
                    bias_compact,
                    output,
                    state_batch_indices,
                    intermediate_states_buffer,
                    intermediate_state_indices,
                    nheads,
                    ngroups,
                    total_tiles,
                    intermediate_states_buffer.stride(0),
                    int(dt_softplus),
                    1,
                    workers,
                    1,
                    1,
                )
                return True

        if (
            state.dtype == torch.float32
            and (nheads, dim, dstate, ngroups) == (16, 64, 128, 1)
            and token_steps in (1, 2, 4, 8)
            and algorithm == "simple"
            and dt_softplus
            and z is None
            and dst_state_batch_indices is not None
            and dst_state_batch_indices.ndim == 2
            and intermediate_states_buffer is None
            and not disable_state_update
            and state_batch_indices.ndim == 1
        ):
            nonpad = torch.nonzero(
                dst_state_batch_indices[0] != pad_slot_id,
                as_tuple=False,
            )
            if nonpad.numel() != 1:
                return False
            checkpoint_step = int(nonpad[0, 0].item())
            if checkpoint_step not in (0, 1, 3, 7):
                return False
            try:
                dt_compact, A_compact, D_compact, bias_compact = (
                    _compact_broadcasts(dt, A, D, dt_bias)
                )
            except ValueError:
                return False
            _load_program(
                f"dynamic_checkpoint{checkpoint_step}", arch, device_index
            ).run(
                state,
                x,
                dt_compact,
                A_compact,
                B,
                C,
                D_compact,
                bias_compact,
                output,
                state_batch_indices,
                dst_state_batch_indices,
                batch_size,
                token_steps,
                checkpoint_step + 1,
                state.stride(0),
                pad_slot_id,
                batch_size,
                nheads,
                4,
            )
            return True
    return False


__all__ = ["try_cake_selective_state_update"]
