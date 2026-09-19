"""
Copyright (c) 2026 by FlashInfer team.

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

import pytest
import torch

import flashinfer

ROWS = 16
WIDTH = 32
NUM_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16
PAGES = 64
NUM_SLOTS = PAGES * PAGE_SIZE


def _geometry(device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    indptr = torch.arange(
        0, (ROWS + 1) * WIDTH, WIDTH, dtype=torch.int32, device=device
    )
    indices = torch.randint(
        0,
        NUM_SLOTS,
        (ROWS * WIDTH,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    q = torch.randn(
        ROWS,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        PAGES,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn_like(k)
    return indptr, indices, q, k, v


def _plan(wrapper, indptr, indices):
    wrapper.plan(
        indptr,
        indices,
        ROWS,
        NUM_SLOTS,
        1,
        1,
        num_qo_heads=NUM_HEADS,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        kv_cache_page_size=PAGE_SIZE,
    )


def _size(wrapper, indptr):
    return wrapper.workspace_size(
        indptr,
        ROWS,
        1,  # R
        1,  # C
        NUM_HEADS,
        1,
        HEAD_DIM,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        kv_cache_page_size=PAGE_SIZE,
    )


FLOAT_BYTES = 16 * 1024 * 1024


def _float_workspace(device):
    """The float workspace a wrapper runs its plans out of.

    Sizing does not read it -- the planner reports how much the split-k
    algorithm wants rather than fitting itself to what is there -- but a
    wrapper cannot be built without one, and the plans it makes run out of it.
    """
    return torch.empty(FLOAT_BYTES, dtype=torch.uint8, device=device)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def test_asking_the_size_plans_nothing(device):
    """A size query has to leave the wrapper as it found it.

    It resolves the backend the same way ``plan`` does, so the answer is for the
    kernel that will run; writing that choice back would let a query decide what
    a later plan picks.
    """
    wrapper = flashinfer.BlockSparseAttentionWrapper(_float_workspace(device))
    indptr, _indices, _q, _k, _v = _geometry(device, 0)

    before = torch.cuda.memory_allocated()
    float_bytes, int_bytes = _size(wrapper, indptr)
    torch.cuda.synchronize()

    assert int_bytes > 0
    assert float_bytes >= 0
    assert wrapper._backend == "auto", "the query resolved the backend in place"
    assert not hasattr(wrapper, "_cached_module"), "the query cached a module"
    assert not hasattr(wrapper, "_plan_info"), "the query left a plan behind"
    assert torch.cuda.memory_allocated() == before, "the query allocated"


def test_two_live_plans_over_one_arena(device):
    """Non-overlapping slices, and each plan still runs its own schedule.

    ``plan`` writes the scheduler's metadata into the integer workspace and
    keeps byte offsets into it; ``run`` reads them back. So the order below --
    both plans made before either runs -- is the one that catches two wrappers
    sharing those bytes.
    """
    indptr, indices_a, q, k, v = _geometry(device, 1)
    _, indices_b, _, _, _ = _geometry(device, 2)

    floats = [_float_workspace(device) for _ in range(2)]
    _, int_bytes = flashinfer.BlockSparseAttentionWrapper(floats[0]).workspace_size(
        indptr,
        ROWS,
        1,
        1,
        NUM_HEADS,
        1,
        HEAD_DIM,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        kv_cache_page_size=PAGE_SIZE,
    )

    # Each slice starts on a 16-byte boundary: the planner carves aligned
    # allocations out of what it is given and would otherwise lose the bytes it
    # skips past.
    slice_bytes = (int_bytes + 15) // 16 * 16
    arena = torch.empty(2 * slice_bytes, dtype=torch.uint8, device=device)
    # One staging buffer for both: the planner consumes it before it returns,
    # and the plans below are made one after the other.
    staging = torch.empty(slice_bytes, dtype=torch.uint8, pin_memory=True)

    wrappers = [
        flashinfer.BlockSparseAttentionWrapper(
            floats[i],
            int_workspace_buffer=arena[i * slice_bytes : (i + 1) * slice_bytes],
            pin_memory_int_workspace_buffer=staging,
        )
        for i in range(2)
    ]
    assert (
        wrappers[0]._int_workspace_buffer.data_ptr()
        != wrappers[1]._int_workspace_buffer.data_ptr()
    )

    # Both planned before either runs: a shared region would have the second
    # plan overwrite what the first is about to read.
    _plan(wrappers[0], indptr, indices_a)
    _plan(wrappers[1], indptr, indices_b)
    got = [wrappers[0].run(q, k, v), wrappers[1].run(q, k, v)]

    # Against the same plans made on wrappers that own everything.
    expected = []
    for indices in (indices_a, indices_b):
        alone = flashinfer.BlockSparseAttentionWrapper(_float_workspace(device))
        _plan(alone, indptr, indices)
        expected.append(alone.run(q, k, v))

    for index, (a, b) in enumerate(zip(got, expected, strict=True)):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f"plan {index}")


def test_the_shared_arena_replays_under_a_graph(device):
    """Capture reaches the same buffers; replay has to keep giving the answer."""
    indptr, indices, q, k, v = _geometry(device, 3)
    floats = _float_workspace(device)
    _, int_bytes = _size(flashinfer.BlockSparseAttentionWrapper(floats), indptr)
    arena = torch.empty(int_bytes, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BlockSparseAttentionWrapper(floats, int_workspace_buffer=arena)
    _plan(wrapper, indptr, indices)
    out = torch.empty(ROWS, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    expected = wrapper.run(q, k, v, out=out).clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            wrapper.run(q, k, v, out=out)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_an_exact_size_is_enough_and_one_byte_less_is_refused(device):
    """The size query answers for the plan, and the planner says so.

    A buffer cut to what the query returned plans; a byte short fails while the
    plan is being made, not somewhere inside a later run.
    """
    indptr, indices, _q, _k, _v = _geometry(device, 4)
    floats = _float_workspace(device)
    _, int_bytes = _size(flashinfer.BlockSparseAttentionWrapper(floats), indptr)

    exact = flashinfer.BlockSparseAttentionWrapper(
        floats,
        int_workspace_buffer=torch.empty(int_bytes, dtype=torch.uint8, device=device),
    )
    _plan(exact, indptr, indices)

    short = flashinfer.BlockSparseAttentionWrapper(
        floats,
        int_workspace_buffer=torch.empty(
            int_bytes - 1, dtype=torch.uint8, device=device
        ),
    )
    with pytest.raises(RuntimeError, match="Buffer overflow"):
        _plan(short, indptr, indices)


def test_a_workspace_on_another_device_is_refused(device):
    """A buffer the kernels cannot reach is a mistake worth naming."""
    with pytest.raises(ValueError, match="device"):
        flashinfer.BlockSparseAttentionWrapper(
            torch.empty(1024, dtype=torch.uint8, device=device),
            int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="uint8"):
        flashinfer.BlockSparseAttentionWrapper(
            torch.empty(1024, dtype=torch.uint8, device=device),
            int_workspace_buffer=torch.empty(1024, dtype=torch.int32, device=device),
        )


def test_the_default_wrapper_still_owns_its_buffers(device):
    """Passing nothing has to behave as it did before the parameters existed."""
    wrapper = flashinfer.BlockSparseAttentionWrapper(_float_workspace(device))
    assert wrapper._int_workspace_buffer.numel() == 8 * 1024 * 1024
    assert wrapper._pin_memory_int_workspace_buffer.is_pinned()
    assert not hasattr(wrapper, "_kv_lens_buffer"), (
        "the device-side kv length buffer is written and never read"
    )


def test_a_misaligned_slice_is_refused(device):
    """A slice starting off a 16-byte boundary loses room it was told it had."""
    floats = _float_workspace(device)
    arena = torch.empty(4096, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="aligned"):
        flashinfer.BlockSparseAttentionWrapper(floats, int_workspace_buffer=arena[1:])


def test_a_caller_packed_mask_gives_what_the_wrapper_would_have_packed(device):
    """The two ways of giving a mask have to agree.

    ``plan`` counts the mask's rows in bits, which is what packing consumes,
    and the kernel indexes the packed mask by byte. Building the mask inside
    ``plan`` goes through ``segment_packbits``, which returns the byte pointer;
    a caller who packed it themselves has to be given the same thing, or every
    row after the first is read from the wrong offset. A width that is not a
    multiple of eight is what separates the two.
    """
    from flashinfer.quantization import packbits

    rows, width = 8, 35  # 35 bits is five bytes with five to spare
    generator = torch.Generator(device=device).manual_seed(41)
    indptr = torch.arange(
        0, (rows + 1) * width, width, dtype=torch.int32, device=device
    )
    indices = torch.randint(
        0,
        NUM_SLOTS,
        (rows * width,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    q = torch.randn(
        rows,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        PAGES,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn_like(k)
    mask = (
        torch.rand(rows * width, device=device, generator=generator) > 0.3
    ).contiguous()

    def plan_and_run(**mask_arguments):
        wrapper = flashinfer.BlockSparseAttentionWrapper(_float_workspace(device))
        wrapper.plan(
            indptr,
            indices,
            rows,
            NUM_SLOTS,
            1,
            1,
            num_qo_heads=NUM_HEADS,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            kv_cache_page_size=PAGE_SIZE,
            **mask_arguments,
        )
        return wrapper.run(q, k, v)

    built = plan_and_run(mask=mask.reshape(-1, 1, 1))
    # The same mask, packed the way the kernel reads it: little-endian, one
    # row of ceil(width / 8) bytes at a time.
    per_row = -(-width // 8)
    packed = torch.empty(rows * per_row, dtype=torch.uint8, device=device)
    for row in range(rows):
        packed[row * per_row : (row + 1) * per_row] = packbits(
            mask[row * width : (row + 1) * width], bitorder="little"
        )
    given = plan_and_run(packed_mask=packed)
    torch.testing.assert_close(given, built, rtol=0, atol=0)


def test_sizing_survives_a_default_device_context(device):
    """The context a model is built in, which is where this is called from.

    ``vllm`` builds a model under ``with target_device:`` and inside that a
    tensor factory with no ``device=`` makes a CUDA tensor. The planner walks
    ``qo_indptr`` on the host, so one built that way is dereferenced as a host
    pointer and the process dies -- no exception, no traceback that names this
    call. Every test here used to run outside such a context and none of them
    could see it.
    """
    indptr = torch.arange(17, dtype=torch.int32, device=device) * 2051

    with torch.device("cuda"):
        inside = flashinfer.BlockSparseAttentionWrapper.query_workspace_size(
            device,
            indptr,
            16,
            1,
            1,
            24,
            2,
            256,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            o_data_type=torch.bfloat16,
            use_custom_mask=True,
            kv_cache_page_size=16,
        )
    outside = flashinfer.BlockSparseAttentionWrapper.query_workspace_size(
        device,
        indptr,
        16,
        1,
        1,
        24,
        2,
        256,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
        use_custom_mask=True,
        kv_cache_page_size=16,
    )
    assert inside == outside


def test_a_device_qo_indptr_is_refused_rather_than_dereferenced(device, tmp_path):
    """The guard under the fix above, reached directly.

    Run in a subprocess: what is being claimed is that the binding raises, and
    a claim about not dying cannot be made from inside the process that would
    die. A wrong input is the caller's, so it is a ValueError.
    """
    import subprocess
    import sys

    program = tmp_path / "device_indptr.py"
    program.write_text(
        "\n".join(
            [
                "import torch, flashinfer",
                "from flashinfer.sparse import _resolve_prefill_module",
                "device = torch.device('cuda', 0)",
                "workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)",
                "# What the planner walks on the host, built on the device.",
                "qo = torch.arange(17, dtype=torch.int32, device=device)",
                "kv = torch.arange(17, dtype=torch.int32, device=device) * 64",
                "lens = (kv[1:] - kv[:-1]).contiguous()",
                "_name, module = _resolve_prefill_module(",
                "    device=device, requested_backend='auto', kv_cache_page_size=16,",
                "    pos_encoding_mode='NONE', use_fp16_qk_reduction=False,",
                "    use_custom_mask=True, q_data_type=torch.bfloat16,",
                "    kv_data_type=torch.bfloat16, index_dtype=torch.int32,",
                "    head_dim=128, logits_soft_cap=0.0, o_data_type=torch.bfloat16)",
                "try:",
                "    module.workspace_size(",
                "        workspace, qo, kv, lens, 16, 16, 4, 1, 16, False, 128, 128,",
                "        False, -1, -1, False, 0, 0)",
                "except Exception as error:",
                "    print('REFUSED', type(error).__name__, str(error).replace(chr(10), ' '))",
                "else:",
                "    print('ACCEPTED')",
            ]
        )
    )
    finished = subprocess.run(
        [sys.executable, str(program)], capture_output=True, text=True, timeout=1800
    )
    assert finished.returncode == 0, (
        "the binding took the process down instead of raising: "
        f"rc={finished.returncode} {finished.stderr[-400:]}"
    )
    assert "REFUSED" in finished.stdout, finished.stdout + finished.stderr
    assert "ValueError" in finished.stdout, finished.stdout
    assert "must be a host tensor" in finished.stdout, finished.stdout
