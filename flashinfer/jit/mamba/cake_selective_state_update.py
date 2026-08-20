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
        "cake_selective_state_update_stp_bf16_direct_c020352b2a",
        ("--fmad=false",),
        "5b1a9a39130cdc29752c842b101b601ef9ecc542c77e36d868bc90db813fadc6",
        "68246b34da8997234c302676a6fafd29b8c7487e2a46b31d600e643ffb97cd3b",
    ),
    "stp_bf16_ratio8": _Program(
        "cake_selective_state_update_stp_bf16_ratio8_5bf1b03b5e",
        ("--fmad=false",),
        "293afb7978b59c68fe8444351a886241df1b6d3f9b396e1e79111cb9035f37b3",
        "28a85fc84be3447eda19f842bc54d44996bcedf8138673bfe9c3d11958080d0d",
    ),
    "stp_bf16_ratio8_saturated": _Program(
        "cake_selective_state_update_stp_bf16_ratio8_saturated_d48b00dba2",
        ("--fmad=false",),
        "3a7ca339df8af5c05c53ff5e55080a2957be22078f8500f004d06a87f1c5632e",
        "35d89afa260f02632a2014eeb04065d2f712573e73a3ae8cfb41fcd0a2c90eef",
    ),
    "stp_bf16_ratio16": _Program(
        "cake_selective_state_update_stp_bf16_ratio16_c67b335fc4",
        ("--fmad=false",),
        "6567c49a5ebd8f8874f09e6b2c2509db929f53c43727e881ccd09694d9c0dd1a",
        "ac9970bf0d62fc292b291eac62b2344c6d329c14a2847fa75b32c4b62560ec80",
    ),
    "stp_bf16_persistent": _Program(
        "cake_selective_state_update_stp_bf16_persistent_a098dbb568",
        ("--fmad=false",),
        "432429604ec998aefb49ded39e2dff733d3b41ea8c0f4ab9eb9fa4e208d42530",
        "0af4d8b6a2c61afacb0bcc1ddfc4691144f7a234d1edde7dd0ae06183fa0e85e",
    ),
    "stp_fp32_identity": _Program(
        "cake_selective_state_update_stp_fp32_identity_0aa3025026",
        (),
        "205b6215df1496f4823e41d505f8b4632954ff623878ee8f8b0f65ee64f9827b",
        "0b7915a8b1a54639691941fb22c369c330a4f7577fdb18d8fdc524a8db0f89bb",
    ),
    "mtp_short": _Program(
        "cake_selective_state_update_mtp_short_4262bee913",
        ("--use_fast_math",),
        "563d9ffbedb38f7f9572d4caf0775bc4fb698b71aff69483439c113267fcee24",
        "87e828484ba53aa7e375bc1ffee2d8166fb5ad93a1f2029142216b5604a56aa8",
    ),
    "mtp_cache_bf16_c4_t6": _Program(
        "cake_selective_state_update_mtp_cache_bf16_c4_t6_44202decf6",
        (),
        "e44807a7566ae24cb783325ae7d784d9ff840b58090a62ed684c557cd8d7ef09",
        "f72922fb7abaab8ffd063820f8c18acf40d3a099a761113f0845903d45e0b1fb",
    ),
    "mtp_cache_bf16_c4_t6_sglang_raw": _Program(
        "cake_selective_state_update_mtp_cache_bf16_c4_t6_sglang_raw_bce0e723e8",
        (),
        "7089b28b57f2c611d8f487b1446712db04ada64b1187c34fa1679258cf6348f0",
        "5806d42e22d1795a622d0a9b885e6330582a315ee2aa2d91cad209b1d1cc5158",
    ),
    "mtp_horizontal": _Program(
        "cake_selective_state_update_mtp_horizontal_ed965a5db1",
        ("--fmad=false",),
        "86bfe41f7e97a4ec591d8c6f44eb139db6ee6d5e7d722ee3b3c8621d60ccc4ab",
        "bd8f6935a771aa4cbaf26272c4760dd0abd705e86721bf9e66ec34ce96504741",
    ),
    "dynamic_checkpoint0": _Program(
        "cake_selective_state_update_dynamic_checkpoint0_3865adb5d9",
        ("--fmad=false",),
        "9b9937420df8d94f98e3b8a14d32a3a2372b35f694d05b58fdcf5cf047fb8f83",
        "d2a929f67956faead8605d654ec9db96ca30571da018226be509e0856b3c7abd",
    ),
    "dynamic_checkpoint1": _Program(
        "cake_selective_state_update_dynamic_checkpoint1_d42cbc4ee1",
        ("--fmad=false",),
        "7d69a23761cb20376c6e24017344634a8d7d6822ccfbdf9fe2aff5b31f0c2c13",
        "55988fd91c176a0f92a3c08f5ba9c5c72383e23eec8d0556b27021f954ac06ca",
    ),
    "dynamic_checkpoint3": _Program(
        "cake_selective_state_update_dynamic_checkpoint3_48f49412a0",
        ("--fmad=false",),
        "5a7c21e62a2311409b394d516c36912d2bc91487dae5819827c32242fb7f2815",
        "6797f662bd58e5ac73bc6edf821075df2c152aea672988766f60e1d14646f95a",
    ),
    "dynamic_checkpoint7": _Program(
        "cake_selective_state_update_dynamic_checkpoint7_270291ffc9",
        ("--fmad=false",),
        "6c98faf0715bc3e9ee814a8364c03bec1497bab883fce41c9392528e4cec3b34",
        "cf2c6bfcbd0f9be267fb6b2624879725f2f3770a00b332d5f85d04cb0db49eb3",
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


def _uniform_checkpoint_step(
    dst_state_batch_indices: torch.Tensor, pad_slot_id: int
) -> Optional[int]:
    """Return the shared checkpoint column, or ``None`` for a non-uniform table."""
    if (
        dst_state_batch_indices.ndim != 2
        or dst_state_batch_indices.shape[0] == 0
        or dst_state_batch_indices.shape[1] == 0
    ):
        return None
    nonpad = dst_state_batch_indices != pad_slot_id
    checkpoint_steps = nonpad.to(torch.int64).argmax(dim=1)
    valid = (nonpad.sum(dim=1) == 1).all() & (
        checkpoint_steps == checkpoint_steps[0]
    ).all()
    selected = torch.where(
        valid,
        checkpoint_steps[0],
        checkpoint_steps.new_full((), -1),
    )
    checkpoint_step = int(selected.item())
    return checkpoint_step if checkpoint_step >= 0 else None


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
                dt_compact, A_compact, D_compact, bias_compact = _compact_broadcasts(
                    dt, A, D, dt_bias
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
            and dt_softplus
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
                dt_compact, A_compact, D_compact, bias_compact = _compact_broadcasts(
                    dt, A, D, dt_bias
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
            if torch.cuda.is_current_stream_capturing():
                return False
            checkpoint_step = _uniform_checkpoint_step(
                dst_state_batch_indices,
                pad_slot_id,
            )
            if checkpoint_step not in (0, 1, 3, 7):
                return False
            try:
                dt_compact, A_compact, D_compact, bias_compact = _compact_broadcasts(
                    dt, A, D, dt_bias
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
