# moe_ep Runbook

Practical how-to for building, testing, and extending the Expert-Parallel MoE
stack. For the design/architecture, see
[`moe_ep_architecture.md`](./moe_ep_architecture.md).

---

## Build & test environment

### Create the container

Tested on 1×4 GB200 GPUs.

Build the flashinfer-EP PyTorch image (pins NCCL-EP runtime wheels + mega deps:
DeepGEMM, NVSHMEM, CUTLASS DSL) into a `.sqsh`, then launch it:

```shell
export RW=/path/to/flashinfer/repo

# 1. Build the container image
srun --jobid="$SLURM_JOB_ID" -N1 \
  --container-image=nvcr.io/nvidia/pytorch:26.05-py3 \
  --container-save=$RW/flashinfer-ep-pt2605-mega_moe_ep.sqsh \
  --container-mounts=$RW:/host \
  bash -lc 'bash /host/flashinfer/docker/install/build_flashinfer_ep_pytorch.sh'

# 2. Launch an interactive shell in the saved image
export IMG=$RW/flashinfer-ep-pt2605-mega_moe_ep.sqsh
export ROOT=/workspace

srun --jobid="$SLURM_JOB_ID" \
  --overlap \
  --container-image="$IMG" \
  --container-mounts="$ROOT:$ROOT" \
  --container-workdir="$ROOT/flashinfer" \
  --pty bash -l

# 3. (Re)build FlashInfer in editable mode (EP backends are on by default;
#    NCCL-EP needs no build step — nccl4py is a base dependency)
BUILD_NIXL_EP=0 \
    pip install --no-cache-dir --no-build-isolation -e .
```

To exercise the NIXL-EP tests, drop `BUILD_NIXL_EP=0` (best-effort build) or set
`BUILD_NIXL_EP=1` (strict — missing build deps abort the install). See
[Running the NIXL-EP tests](#running-the-nixl-ep-tests).

Build flags (tri-state; unset = on, best-effort): `BUILD_NIXL_EP=0` skips the
NIXL-EP meson build, `BUILD_NIXL_EP=1` makes its missing build deps a hard
error, `BUILD_NVEP=0` turns both backends off. Probe availability at runtime
with `have_nccl_ep()`, `have_nixl_ep()`, `available_backends()`.

### CUTLASS DSL version

After the editable install, bring the DSL up to a supported version:

```bash
pip install -U "nvidia-cutlass-dsl[cu13]"   # or pin, e.g. ==4.6.1
```

The pt2605 container ships nvidia-cutlass-dsl **4.5.0**; the cutedsl mega
kernels need ≥ 4.6.x. **4.6.1** is the perf-validated reference (the TUNING.md
and benchmark tables were measured on it); **4.7.0** is correctness-validated
(2026-08-10, jobs 2384640/2384641/2384650: drop harness, fused-quant unit
tests, and the full deep_gemm/nvfp4/mxfp8 mega multirank + oracle suites all
green) but its perf has not been measured — pin 4.6.1 when producing numbers
meant to compare against the reference tables.

History: this section used to be a hard `==4.6.1` pin because 4.7.0 crashed
every 4-rank `deep_gemm.fp8_fp4_mega_moe` launch with
`CUDA_ERROR_MISALIGNED_ADDRESS` (bisected 2026-08-05 on prenyx B200). The
root cause was not deep_gemm or the dsl's bundled CUDA libs but the fused
activation-quant staging (`DataPreprocess` in
`kernel_src/cutedsl_megamoe/src/src/inputs_process.py`), shared by every mega
staging path — fixed by the upstream `50117315d` sync recorded in that drop's
VENDOR.md, after which the pin was lifted. The vLLM e2e sections below keep
their own separate **4.5.2** pin (vLLM 0.25.1's requirement) — that pin is for
the vLLM engine env, not for running the moe_ep test suite.

Tolerance note (dsl-version-independent, resolved 2026-08-05): on B200 nodes
`test_moe_ep_mxfp8_cutedsl_mega_multirank_torch_oracle[False]` used to fail by
one bf16 cell (rank 3, |d|=16.0 vs atol=8.0, rel_l2≈0.0017) — the flat atol
was really "1 bf16 ULP at |term|≈2048" calibrated on GB200's rounding, and
where large per-topk terms nearly cancel the achievable agreement is bounded
by the bf16 round-off of the TERMS, not of the final value. The mxfp8 oracle
compares (multirank + single-GPU) now use a per-cell term-magnitude band
derived from the oracle's own pre-reduce terms
(`_assert_mega_oracle_term_band_close` in
`tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py`), which is
arch-independent. GB200 verified 2026-08-05; if a B200 run still trips the
band, that is a real signal, not marginality.

### Run tests

`tests/moe_ep/run_tests.sh <target>` — targets and requirements:

| Command | GPUs | Requires |
|---------|------|----------|
| `bash tests/moe_ep/run_tests.sh unit` | 1 (host-only) | none — mocks + single GPU, no multirank |
| `bash tests/moe_ep/run_tests.sh multirank` | 4 | NCCL-EP (NIXL-EP too if built) |
| `bash tests/moe_ep/run_tests.sh split_path_correctness_bf16` | 4 | Blackwell |
| `bash tests/moe_ep/run_tests.sh mega` | 4 | Blackwell sm_100+; DeepGEMM + BF16 + NVFP4 + MXFP8 |

- **unit** — host-only pytest (mocks + single-GPU). The full run accumulates
  native heap damage somewhere in the GPU/DSL/transport stack: with every
  test PASSING, the process aborts either (a) at the first heavy
  import/compile burst — historically
  `test_workspace_pool.py::test_two_nvfp4_layers_share_one_symm_buffer`
  (`Fatal Python error: Aborted` in the nvfp4 warmup's module imports) — or
  (b) in CPython teardown after the pytest summary
  (`malloc(): unaligned tcache chunk detected`, job 2388315). Not a kernel
  or test bug: everything passes standalone, per-file, and in every subset
  tried (observed since 2026-07-22; B200, dsl 4.6.1). `run_tests.sh unit`
  therefore (1) runs that test in its own pytest process and (2) exits both
  processes via `os._exit(pytest_rc)` to skip interpreter finalization. If
  the isolated invocation ever FAILS (not crashes), that is a real signal.
  Root cause still open — needs an ASAN/valgrind pass over the suite.
- **multirank** — 4-GPU split path over NCCL-EP (and NIXL-EP when built).
- **split_path_correctness_bf16** — 4-GPU bf16 split-path numerics vs a
  single-process `MoELayer` reference.
- **mega** — 4-GPU DeepGEMM + BF16 + NVFP4 + MXFP8 mega parity, plus single-rank
  MXFP8 preprocess-vs-reference check.

`all` and `smoke` targets also exist. Split-path numerics are **bf16-only** for
now.

### Running the NIXL-EP tests

NIXL-EP is the second split-path transport (`backend="nixl_ep"`), currently
**low-latency + `EXPERT_MAJOR` only**. Constraints enforced by
`validate_fleet_params`: `max_tokens_per_rank ≤ 1024`, `token_hidden_size ∈
{2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}`, torch built for CUDA ≥ 13,
sm_90+.

**1. Build with NIXL-EP enabled.** Build deps: `meson`, `ninja`, `pkg-config`,
`nvcc`, UCX (`pkg-config --exists ucx`), `libibverbs-dev`. The build hook
pre-installs the `nixl-cu13` wheel it links against (`_ensure_nixl_wheel`), so
no manual NIXL install is needed.

**UCX caveat:** the UCX found via pkg-config must ship the *device API*
(`ucp/api/device/ucp_device_impl.h`) — NGC images' HPC-X UCX and Ubuntu's apt
UCX both predate it, and the `flashinfer-ep-pt2605*` images skip this
provisioning. Follow `docker/Dockerfile.flashinfer-nvep`: install DOCA 3.2
host packages + GDRCopy, then build UCX `v1.21.x` from source with
`--enable-experimental-api --with-cuda --with-verbs --with-dm` and put its
`lib/pkgconfig` first on `PKG_CONFIG_PATH`. Without it, `BUILD_NIXL_EP=1`
fails compiling `nixl_device.cuh`:

```shell
BUILD_NIXL_EP=1 pip install --no-cache-dir --no-build-isolation -e .

# Verify the backend actually built (best-effort builds skip it silently):
python -c "from flashinfer.moe_ep import available_backends; print(available_backends())"
# expect: [..., 'nixl_ep']
```

**2. Smoke test** (4 GPUs, single node — fixed shape: 64 tokens, 8 experts,
hidden 4096, topk 4, bf16, LL):

```shell
torchrun --nproc_per_node=4 tests/moe_ep/smoke_nixl_ep.py
# each rank prints: SMOKE_RESULT: nixl_ep OK
```

**3. Multirank roundtrip + split kernels** (4 GPUs). The `multirank` target
runs NCCL-EP first, then repeats over NIXL-EP when `have_nixl_ep()` is true:

```shell
bash tests/moe_ep/run_tests.sh multirank
```

or NIXL-only, directly:

```shell
torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_layer_multirank.py -v \
    -m "nvep and gpu_4" --backend=nixl_ep
torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_split_kernels.py -v \
    -m "nvep and gpu_4" --backend=nixl_ep
```

**4. Host-only mock tests** (no GPU fabric / no NIXL build needed) are part of
the `unit` target — `tests/moe_ep/nixl_ep/test_fleet_mock.py` stubs the NIXL
`Buffer` and checks fleet sizing, `update_topology` rank diffs, and combine
knob validation.

Notes:

- **Rendezvous**: unlike NCCL-EP (which mirrors the torch process group), the
  NIXL `Buffer` rendezvous needs a `torch.distributed.TCPStore` passed via
  `BootstrapConfig(tcp_store=...)`. The tests open one themselves on
  `MASTER_PORT + 1` (sharing torch's port would clash) — see
  `tests/moe_ep/smoke_nixl_ep.py` for the pattern.
- `NPROC_SMOKE` / `NPROC_MULTIRANK` (default 4) override the rank count for
  the `run_tests.sh` targets.
- UCX/ibverbs are **build-time** deps; the tests set no `NIXL_*`/`UCX_*` env.
  When UCX lives in a non-default prefix (e.g. `/opt/ucx` from the source
  build above), put its `lib/` on `LD_LIBRARY_PATH` at **runtime** too —
  loading a different UCX than the one nixl was built against fails at fleet
  creation with `registerMem(...) != NIXL_SUCCESS`.
- **Pin the `nixl-cu13` wheel to the 3rdparty/nixl submodule tag** (currently
  `==1.3.1`; the build hook installs the pinned version via
  `_NIXL_WHEEL_VERSION` in `build_backend.py`). `nixl_ep_cpp.so` compiles the
  submodule's device kernels but loads the wheel's `libnixl.so` at runtime —
  a skewed pair (e.g. a 1.4.x wheel over the v1.3.1 kernels) builds and
  imports fine, then dies at the first dispatch with device asserts
  (`nixlPut(...) == NIXL_IN_PROG` → `cudaErrorIllegalAddress`).
- **Do not run concurrent `BUILD_NIXL_EP=1` installs from one checkout**
  (e.g. several SLURM jobs sharing a network-filesystem clone): the build
  patches `3rdparty/nixl` in place and shares `build_nvep/`, so parallel
  installs race and fail with `git apply` / meson `Unknown option` errors.
  Serialize the first build; later installs reuse the staged `_libs/` .so.
- The install and the launcher must resolve to the **same interpreter**: an
  editable install into a venv is invisible to a `torchrun` that resolves to
  the system python (`ModuleNotFoundError: flashinfer` in the spawned ranks
  while parent-shell imports work). When in doubt, launch with
  `python -m torch.distributed.run` so the launcher is pinned to the python
  that owns the install.
- NIXL-EP coverage today is smoke + multirank + mocked unit tests only; the
  correctness/mega targets are NCCL-EP-only.

### NCCL-EP low-latency device-kernel limits

Two constraints of the `nccl.ep` LL device kernel (probed empirically on
nccl4py 0.3.1; not enforced by `validate_fleet_params`, so they surface as
device-side aborts):

- **Per-token row widths are whitelisted**: LL dispatch accepts bf16 rows of
  {2048, 2560, 4096, 5120, 6144, 7168, 8192} elements only — 3072, sub-2048
  widths, and all 1-byte payload dtypes are rejected with
  `low_latency.cu 'Unsupported hidden'`. The sent row may be narrower than
  `FleetParams.token_hidden_size` (recv buffers mirror the sent row), which
  is what the split path's packed-MXFP8 dispatch relies on.
- **top-k is capped at 8** (`numTopk <= kNumMaxTopK`, `low_latency.cu`):
  top-10 models (e.g. Qwen3.5) abort on NCCL-EP LL; use the HT algorithm or
  NIXL-EP (which handles top-10 at LL).

### SM90 mega token sweep

Hopper-only (`sm90_fp8_fp8_bf16_pull_cutedsl`) correctness targets run in their own pytest
process (the SM90/SM100 kernel trees are mutually exclusive per process):
`bash tests/moe_ep/run_tests.sh oracle_sm90` (1 GPU) and
`bash tests/moe_ep/run_tests.sh mega_sm90` (4 GPUs).

The perf microbenchmark reproduces the kernel drop's Hopper P03 multirank
token sweep (`moe_hopper_fp8/run_token_sweep_benchmark.py`, DSV4 geometry:
topk 6, 384 experts EP4, hidden 7168, intermediate 3072 post-SwiGLU, tokens
per rank 512..32768) through the FI `MoEEpLayer` mega path, on 4×H100:

```bash
torchrun --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py
```

Rank 0 prints one `BENCH_CSV` row per (scale_mode, layout, tokens) point;
each row names the matching drop reference CSV
(`moe_hopper_fp8/benchmark_data/20260720/...`) so comparison is one grep
away. The `compute_*_us` columns map to the drop's per-rank
`mega_us + topk_us`; `e2e_*_us` adds FI staging/validation/output-copy.
Axes: `--scale-mode {per_tensor,blockwise,both}`, `--swap-ab`/`--no-swap-ab`
(default both layouts at the shim default tiles: non-swap M64 N128, swap-AB
M256 N32), `--mma-tiler M,N`, `--tokens`, `--kind`. See the module docstring
for the full timing/mapping notes. Measured results, comparison caveats,
and the reproduce recipe live in
[`kernel_src/sm90/pull_style_cutedsl_megakernel/TUNING.md`](../../flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/TUNING.md).

---

## Benchmarking

Two harnesses live in the companion **`moe_ep_benchmark`** repo (kept next to
the flashinfer checkout — not part of this tree): a single-node
**microbenchmark** (`fi_mega` / `model_shapes`) and a **vLLM 0.25.1
end-to-end** suite (`vllm_e2e/`). Clone it beside your flashinfer checkout:

```bash
export ROOT=/path/to            # parent dir; hold both checkouts here
cd "$ROOT"
git clone https://github.com/mhoqueanik/moe_ep_benchmark.git
# pin the revision the published numbers were run with (vllm-pr branch,
# "Add 2026-07-22 cutlass-dsl 4.5.2 validation results" — scripts + result CSVs)
git -C moe_ep_benchmark checkout c8aefda
# $ROOT now has both flashinfer-2/flashinfer-moe_ep and moe_ep_benchmark/
```

All numbers below were measured on **4× GB200 (SM100)** at the tip of this
tree's `4_5_2-perf-fix` flashinfer branch, on **nvidia-cutlass-dsl 4.5.2**
(vLLM 0.25.1's own pin) — the measured and supported baseline. 4.6.1 appears
below only as a parity *reference*: the MR!27 mainloop WAR brings 4.5.2 to
4.6.1 parity, so 4.5.2 is the runtime floor and versions below it are
unsupported (4.5.0 fails at `cute.compile`) — see
[`../../flashinfer/moe_ep/kernel_src/cutedsl_megamoe/TUNING.md`](../../flashinfer/moe_ep/kernel_src/cutedsl_megamoe/TUNING.md).

### 1. Microbenchmark

Every table below comes from one sweep — `model_shapes/run_model_shapes.sh` —
which runs five variants (the table columns) across the MoE geometries of real
models (`model_shapes/shapes.tsv`). The `deepseek_v3` geometry
(7168 / 2048 / 256 / top-8) **is** the default-geometry table.

| column          | variant          | backend          | env |
|-----------------|------------------|------------------|-----|
| `dg`            | `fi_dg`          | `sm100_fp8_fp4_bf16_deepgemm` | — |
| `nvfp4 bf16`    | `fi_fp4`         | `sm100_nvfp4_nvfp4_bf16_cutedsl`  | — |
| `+ikr`          | `fi_ikr`         | `sm100_nvfp4_nvfp4_bf16_cutedsl`  | `MEGA_IKR=1` (in-kernel fc2 reduce) |
| `+combine_nvfp4`| `fi_combine_fp4` | `sm100_nvfp4_nvfp4_bf16_cutedsl`  | `MEGA_COMBINE_DTYPE=nvfp4` (16·e2m1 + bf16/16 wire) |
| `+combine_mxfp8`| `fi_combine_fp8` | `sm100_nvfp4_nvfp4_bf16_cutedsl`  | `MEGA_COMBINE_DTYPE=mxfp8` (32·e4m3 + e8m0/32 wire) |

Inside the flashinfer-EP container (editable install per "Build & test
environment" above), pin the DSL and run the sweep:

```bash
# pin the DSL the numbers were measured on ($ROOT set at clone time above)
python -m pip install "nvidia-cutlass-dsl[cu13]==4.5.2"
python -c "import cutlass; assert cutlass.__version__=='4.5.2', cutlass.__version__"

cd "$ROOT/moe_ep_benchmark"
# all six shapes × five variants; e2e_pipelined timing is the sweep default
SEQ_LENS="8 64 512 1024 2048 4096 8192" GPUS=4 \
  bash model_shapes/run_model_shapes.sh
# just the default-geometry table:
SHAPES=deepseek_v3 SEQ_LENS="8 64 512 1024 2048 4096 8192" \
  bash model_shapes/run_model_shapes.sh
# render markdown (speedup-vs-dg columns) from the CSV(s):
python model_shapes/make_tables.py model_shapes/results/model_shapes_*.csv
```

- `MEGA_TIMING=e2e_pipelined` (steady-state, iters back-to-back enqueued) is the
  default and the methodology of these tables — don't override it.
- First-time `cute.compile` scales with experts/rank (~12 min per kernel config
  at 64 experts/rank; `qwen3_5_397b` at 128/rank is the worst case). Compiles
  cache under `~/.cache/flashinfer`, so resubmits are cheap.

#### Microbenchmark results (2026-07-22, `e2e_pipelined` p50 µs)

Default geometry (7168 hidden / 2048 inter / 256 experts / top-8), heuristic
knobs, speedup vs `sm100_fp8_fp4_bf16_deepgemm` in parens:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 215.0  | 222.2 (0.97x)  | 234.0 (0.92x)  | 228.3 (0.94x)      | 228.8 (0.94x)  |
| 64       | 288.8  | 287.7 (1.00x)  | 314.8 (0.92x)  | 304.2 (0.95x)      | 308.2 (0.94x)  |
| 512      | 346.1  | 359.3 (0.96x)  | 362.5 (0.95x)  | 328.7 (1.05x)      | 334.8 (1.03x)  |
| 1024     | 474.1  | 428.6 (1.11x)  | 431.8 (1.10x)  | **375.8 (1.26x)**  | 384.0 (1.23x)  |
| 2048     | 822.3  | 621.5 (1.32x)  | 615.4 (1.34x)  | **545.8 (1.51x)**  | 573.4 (1.43x)  |
| 4096     | 1501.7 | 1001.5 (1.50x) | 989.2 (1.52x)  | **931.3 (1.61x)**  | 947.5 (1.58x)  |
| 8192     | 3072.9 | 1896.4 (1.62x) | 1878.3 (1.64x) | **1655.3 (1.86x)** | 1752.0 (1.75x) |

Shape of the curve: **parity with dg through ~512 tok/rank, crossover between
512 and 1024, win growing to 1.62x at 8192** (1.86x with the fp4 combine wire).
The small-batch regime is weight-load bound and fp4-vs-fp4 there is a wash.

**Real-model geometry sweep (2026-07-21)** — same recipe/session/node; pattern
holds everywhere (dg-parity below ~512 tok/rank, fp4 combine-wire best at large
tokens, 1.6-1.9x on 7168-hidden shapes). `e2e_pipelined` p50 µs, speedup vs
`sm100_fp8_fp4_bf16_deepgemm` in parens.

_deepseek_v3_ — hidden 7168, inter 2048, 256 experts, top-8 (independent
same-session re-run of the default table; matches within run noise):

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 211.1  | 220.5 (0.96x)  | 233.9 (0.90x)  | 228.3 (0.92x)      | 227.2 (0.93x)  |
| 64       | 286.7  | 285.7 (1.00x)  | 314.4 (0.91x)  | 303.2 (0.95x)      | 306.0 (0.94x)  |
| 512      | 348.2  | 357.4 (0.97x)  | 361.5 (0.96x)  | 326.7 (1.07x)      | 332.8 (1.05x)  |
| 2048     | 830.0  | 612.2 (1.36x)  | 609.2 (1.36x)  | **539.6 (1.54x)**  | 566.2 (1.47x)  |
| 8192     | 3129.4 | 1942.5 (1.61x) | 1916.9 (1.63x) | **1728.0 (1.81x)** | 1784.3 (1.75x) |

_deepseek_v4_flash_ — hidden 4096, inter 2048, 256 experts, top-6:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4    | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|------------------:|---------------:|
| 8        | 127.8  | 142.1 (0.90x)  | 148.5 (0.86x)  | 148.3 (0.86x)     | 151.6 (0.84x)  |
| 64       | 176.2  | 190.3 (0.93x)  | 204.7 (0.86x)  | 195.2 (0.90x)     | 197.6 (0.89x)  |
| 512      | 206.8  | 231.1 (0.89x)  | 229.2 (0.90x)  | 216.0 (0.96x)     | 220.2 (0.94x)  |
| 2048     | 401.4  | 344.6 (1.16x)  | 344.1 (1.17x)  | **303.1 (1.32x)** | 318.5 (1.26x)  |
| 8192     | 1294.9 | 1070.1 (1.21x) | 1061.6 (1.22x) | **794.1 (1.63x)** | 901.1 (1.44x)  |

_deepseek_v4_pro_ — hidden 7168, inter 3072, 384 experts, top-6:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 286.7  | 304.2 (0.94x)  | 310.4 (0.92x)  | 309.2 (0.93x)      | 312.8 (0.92x)  |
| 64       | 540.7  | 562.2 (0.96x)  | 575.5 (0.94x)  | 566.0 (0.96x)      | 570.1 (0.95x)  |
| 512      | 603.1  | 621.6 (0.97x)  | 621.8 (0.97x)  | 601.1 (1.00x)      | 607.2 (0.99x)  |
| 2048     | 921.1  | 760.8 (1.21x)  | 758.1 (1.22x)  | **695.3 (1.32x)**  | 715.7 (1.29x)  |
| 8192     | 3093.5 | 1897.4 (1.63x) | 1880.4 (1.65x) | **1773.6 (1.74x)** | 1836.5 (1.68x) |

_kimi_k2_6_ — hidden 7168, inter 2048, 384 experts, top-8:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 247.8  | 253.0 (0.98x)  | 275.4 (0.90x)  | 261.2 (0.95x)      | 263.2 (0.94x)  |
| 64       | 409.6  | 401.2 (1.02x)  | 440.2 (0.93x)  | 416.8 (0.98x)      | 419.8 (0.98x)  |
| 512      | 472.2  | 469.0 (1.01x)  | 470.4 (1.00x)  | 442.4 (1.07x)      | 449.0 (1.05x)  |
| 2048     | 834.5  | 668.8 (1.25x)  | 662.2 (1.26x)  | **603.1 (1.38x)**  | 619.5 (1.35x)  |
| 8192     | 3188.3 | 2000.3 (1.59x) | 1982.9 (1.61x) | **1701.9 (1.87x)** | 1894.4 (1.68x) |

_qwen3_5_397b_ — hidden 4096, inter 1024, 512 experts, top-10:

| tok/rank | dg     | nvfp4 bf16     | +ikr           | +combine_nvfp4     | +combine_mxfp8 |
|---------:|-------:|---------------:|---------------:|-------------------:|---------------:|
| 8        | 129.0  | 145.7 (0.89x)  | 167.8 (0.77x)  | 157.7 (0.82x)      | 161.8 (0.80x)  |
| 64       | 194.6  | 213.0 (0.91x)  | 264.3 (0.74x)  | 234.5 (0.83x)      | 238.6 (0.82x)  |
| 512      | 233.5  | 253.9 (0.92x)  | 256.0 (0.91x)  | 238.6 (0.98x)      | 243.8 (0.96x)  |
| 2048     | 495.6  | 459.0 (1.08x)  | 461.8 (1.07x)  | **361.6 (1.37x)**  | 391.0 (1.27x)  |
| 8192     | 1772.8 | 1506.0 (1.18x) | 1544.7 (1.15x) | **1100.9 (1.61x)** | 1270.8 (1.40x) |

_gpt_oss_120b_ — hidden 2880, inter 2880, 128 experts, top-4. No `dg` column:
2880 fails deep_gemm's hard-%128 wire alignment, so this geometry runs only via
the cutedsl %64 relaxation (fp4 variants only):

| tok/rank | dg | nvfp4 bf16 | +ikr   | +combine_nvfp4 | +combine_mxfp8 |
|---------:|:--:|-----------:|-------:|---------------:|---------------:|
| 8        | —  | 111.6      | 113.9  | 115.1          | 116.7          |
| 64       | —  | 125.9      | 132.5  | 132.1          | 135.6          |
| 512      | —  | 150.5      | 150.5  | 150.5          | 150.0          |
| 2048     | —  | 261.1      | 261.0  | **252.7**      | 259.1          |
| 8192     | —  | 697.3      | 678.9  | **551.9**      | 609.3          |

### 2. End-to-end vLLM benchmark

The backend integrated into vLLM 0.25.1 (integration lands as a separate PR),
driven from `vllm_e2e/` — see `vllm_e2e/RUNBOOK.md` for the full recipe and
`vllm_e2e/FINDINGS.md` for the measured numbers. Backend is selected per-run by
env (all runs pass `--moe-backend deep_gemm_mega_moe`; the fi path is env-gated):

| config   | env |
|----------|-----|
| native   | `FI_MOE_EP=0` |
| fi_dg    | `FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=sm100_fp8_fp4_bf16_deepgemm` |
| fi_nvfp4 | `FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=sm100_nvfp4_nvfp4_bf16_cutedsl` |

```bash
W=$ROOT/moe_ep_benchmark/vllm_e2e
# 1) hold a 4-GPU node; run every command through the container helper:
JOBID=$(sbatch --parsable -A coreai_libraries_cudnn -p batch -N1 \
    --ntasks-per-node=1 --time=04:00:00 -J fi.vllm_e2e.hold \
    --output=$W/logs/hold_%j.log --wrap "sleep 14400")
# 2) one-time venv: vllm==0.25.1 + editable fi branch + patch_0251 (keeps
#    vllm's own cutlass-dsl 4.5.2 pin):
JOBID=$JOBID bash $W/in_container.sh 'bash setup_container.sh'
# 3) throughput A/B (backend env-only):
JOBID=$JOBID bash $W/in_container.sh 'bash bench_throughput.sh'
# 4) GSM8K fairness gate (both checkpoints must land in the same band):
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && FI_MOE_EP=0 python eval_gsm8k.py --tag native --out results/gsm8k_native.json'
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=sm100_nvfp4_nvfp4_bf16_cutedsl python eval_gsm8k.py --tag fi_nvfp4 --out results/gsm8k_fi_nvfp4.json'
```

Reproducing the **headline cells** (not `bench_throughput.sh`'s defaults):
- **Prefill, 8k-token chunks**: `WORKLOADS="prefill:8192:1" MAX_BATCHED_TOKENS=8192 bash bench_throughput.sh`.
- **Decode @ 1024-seq concurrency**: run the decode workload with CUDA-graph
  capture covering the prefill chunk shapes (vLLM
  `max_cudagraph_capture_size` >= the chunk size). With the default small
  capture list the chunks fall to eager, the extra inter-rank launch skew is
  absorbed as collective spin, and decode reads ~4 % **below** native — a config
  artifact, not a kernel property.
- **Per-role offline knob caches**: build prefill- and decode-tuned caches with
  `python -m flashinfer.moe_ep.tune` before the timed runs.
- **Two-checkpoint fairness**: `fi_nvfp4` consumes the prequantized NVFP4
  checkpoint directly via the weight-pack path; native and `fi_dg` run the mxfp4
  checkpoint. The GSM8K gate must show both in the same accuracy band before a
  throughput delta is apples-to-apples. **Band definition**: both variants run
  the identical 200-question set with greedy decoding, and the gate passes when
  `fi` accuracy ≥ native accuracy − 0.02 (4 questions of 200, ≈2 standard
  errors of the binomial noise at ~0.97 accuracy). The published 0.975 (`fi`)
  vs 0.965 (native) passes with `fi` above native.
- Repeat each cell ≥3× and use medians — prefill-heavy cells showed ±35 %
  cross-restart variance; decode/mixed were stable to ~2 %.

#### End-to-end results (vLLM 0.25.1, 2026-07-20)

DeepSeek-V4-Flash (4096 hidden / 2048 inter / 256 experts / top-6), 4× GB200
TP4/EP4, CUDA graphs capturing all recurring step shapes (incl. the 4096-token
prefill chunks), per-role offline knob caches.

| Workload                                     | native vLLM | fi_dg           | fi_nvfp4            |
|----------------------------------------------|------------:|----------------:|--------------------:|
| Prefill, 8k-token chunks (tok/s)             | 45,701      | 47,534 (1.04x)  | **53,962 (1.18x)**  |
| Decode @ 1024-seq concurrency (output tok/s) | 21,049      | 21,540 (1.02x)  | **22,614 (1.07x)**  |
| GSM8K (200q, greedy)                         | 0.965       | 0.965           | **0.975**           |

- `fi_dg` is the same deep_gemm kernel routed through this integration layer —
  parity-or-better shows the layer itself costs nothing under graphs; the nvfp4
  deltas above it are kernel-side.
- 2026-07-22 revalidation on nvidia-cutlass-dsl 4.5.2 (vLLM 0.25.1's own pin):
  the prefill-8k cell reproduces on the same metric — native 45,582 / fi_nvfp4
  53,623 tok/s = 1.176x, within 0.3 % of the headline cells. The decode-1k
  check was recorded as **total tok/s** (prompt + output: native 32,086 /
  fi_nvfp4 34,263 = 1.068x), a *different metric* from the headline row's
  output tok/s — it corroborates the 1.07x speedup ratio, not the absolute
  output-tok/s cells.

---

## Adding a new mega-kernel backend

A mega kernel owns fused comm + local MoE. To wire a new one, add a subpackage
under `flashinfer/moe_ep/backends/mega/kernel/sm<arch>/<act>_<weight>_<out>_<style>/`. Kernel-team drops are
vendored per architecture under `flashinfer/moe_ep/kernel_src/<arch>/`:

- `kernel_src/cutedsl_megamoe/` — Blackwell (NVFP4 + MXFP8 kernels)
- `kernel_src/sm90/pull_style_cutedsl_megakernel/` — Hopper pull-style FP8
  (a fork of the same kernel repo)
- `kernel_src/sm90/push_style_megamoe/` — Hopper push-style FP8 (raw CUDA,
  JIT-compiled; vendored from flashinfer PR #4069, see its VENDOR.md)
- `kernel_src/sm120/swapab_cutedsl_megakernel/` — Blackwell-consumer
  (sm_120/sm_121) swap-AB MXFP8 (another fork snapshot of the same repo)

Each tree exposes its kernels through its own package public API (e.g. the
sm100 tree's `mxfp8_mega_moe`, `get_symm_buffer_for_mxfp8_mega_moe`). The
trees duplicate the shared kernel-repo runtime (`common`, `src`, …) at their
own drop revision and are **process-exclusive** — the top-level kernel module
names collide, so each tree's `shim/_paths.py` refuses to bootstrap when the
sibling tree's modules are already imported (a process runs on one
architecture anyway). Use the existing `sm100_mxfp8_mxfp8_bf16_cutedsl` backend as the
reference template.

### 1. Kernel + frontend (the "backend config" it links to)

Every mega kernel exposes exactly **two entry points** through a thin frontend,
and the `MegaKernelBackend` subclass links to nothing else. Keep this contract
stable so new kernels — including future **SM90 (Hopper)** and **SM120
(Blackwell-consumer)** variants — drop in behind the same backend shape without
touching `modes/` or the registry:

**(a) Workspace allocator** — problem sizes first, tuning knobs keyword-only;
returns a symm-buffer object with the staging views the backend fills and a
`.destroy()`. Model it on
`get_symm_buffer_for_mxfp8_mega_moe` / `get_symm_buffer_for_mega_moe`:

```python
def get_symm_buffer_for_<name>_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,        # == fleet_params.max_tokens_per_rank
    num_topk: int,
    hidden: int,                # fleet_params.token_hidden_size
    intermediate: int,          # post-SwiGLU width
    rank: int,                  # self.ep_rank
    world_size: int,            # self.ep_world_size
    *,
    kind=...,                   # dtype selector, if applicable
    # ... kernel knobs: clamps, in_kernel_fc2_reduce, token_back_by_dispatch, ...
) -> <Name>SymmBuffer: ...
```

The returned buffer must expose the staging tensors the backend's `stage_inputs`
writes — at minimum `x`, `x_sf` (quantized paths), `topk_idx`, `topk_weights` —
plus `destroy()`. Expert weights are **not** owned by the workspace; they are
passed to the compute call each launch.

**(b) Compute entry** — output tensor first, then the two kernel-ready
`(weight, scale)` weight tuples, the workspace, and keyword-only knobs. Model it
on `mxfp8_mega_moe` / `nvfp4_mega_moe`:

```python
def <name>_mega_moe(
    y: torch.Tensor,                  # bf16 [num_tokens, hidden] output
    transformed_l1,                   # (w13, w13_scale) kernel-ready fc1
    transformed_l2,                   # (w2, w2_scale)  kernel-ready fc2
    symm_buffer: <Name>SymmBuffer,
    *,
    num_tokens: int | None = None,
    # ... clamps, fast_math (accept for API parity even if a no-op), ...
) -> None: ...
```

`compute` fuses dispatch + fc1 + fc2 + combine and writes `y[:num_tokens]`. The
caller (the backend's `stage_inputs`) must have filled `symm_buffer.x` and the
routing slices first.

Add both functions under the owning tree's `shim/` — e.g.
`kernel_src/cutedsl_megamoe/shim/` for Blackwell kernels (alongside
`nvfp4.py` / `mxfp8.py`), `kernel_src/sm90/pull_style_cutedsl_megakernel/shim/`
for Hopper — and re-export them from that package's `__init__.py` (or point at
your own kernel module). Raw kernel sources live under the tree's `src/` — see
the tree's `SKILL.md` for how to update that directory when the kernel team
ships a new drop. The kernel-specific tuning knobs
(intermediate size, top_k, clamps, dtype `kind`, fast-math, reduce/dispatch
flags) live on the **config** dataclass in step 2 and are threaded through to
these two calls by the backend in step 4 — so an SM90/SM120 kernel that needs
different knobs only changes its own config + these two signatures, not the
`MegaKernelBackend` plumbing.

### 2. `config.py` — the user-facing config dataclass

```python
@dataclass
class MyMegaMoeConfig:
    intermediate_size: int          # post-SwiGLU width
    top_k: int
    kernel_name: str = "my_mega"    # MUST match the @register_mega_kernel name
    # ... kernel-specific knobs (dtype kind, clamps, fast_math, ...)
```

`kernel_name` is how the registry resolves a config to a backend
(`_kernel_name()` reads this attribute) — it must be a non-empty string equal to
the registration name.

### 3. `weights.py` — weight transform + validation

Provide `preprocess_mega_weights(weights: MoEWeightPack, ...) -> Transformed...`
that turns canonical bf16 (or pre-quantized) `w13`/`w2` into the kernel-ready
layout, and a `validate_transformed_mega_weights(...)` for the
`preprocess_weights=False` path (user supplies `MegaConfig.transformed_weights`).

### 4. `backend.py` — subclass `MegaKernelBackend` + register

```python
from .....core.kernel.base import MegaKernelBackend
from .....core.kernel.registry import register_mega_kernel

@register_mega_kernel("my_mega")           # == config.kernel_name
class MyMegaKernelBackend(MegaKernelBackend):
    @classmethod
    def kernel_name(cls) -> str:
        return "my_mega"

    # Required abstracts:
    def _allocate_workspace(self, fleet_params): ...   # call frontend allocator
    def compute(self, workspace, transformed_weights, *, output): ...  # call frontend kernel

    # Common overrides:
    def runtime_requirements(self, bootstrap): ...     # add "nvshmem" if needed
    def validate_init(self, bootstrap, fleet_params): ...
    def preprocess_weights(self, weights, fleet_params): ...
    def validate_transformed_weights(self, tw, bootstrap, fleet_params): ...
    def validate_forward(self, t, fleet_params, *, quantize_input): ...
    def stage_inputs(self, t, workspace, *, quantize_input): ...  # copy/quantize acts
    def destroy(self, workspace): ...
```

EP rank/world/comm are available via `self.ep_rank`, `self.ep_world_size`,
`self.ep_comm_group` (bound by `bind_ep_bootstrap`, resolved lazily once dist is
up). If the kernel needs NVSHMEM, return it from `runtime_requirements()` (see
`mxfp8_cutedsl_runtime_requirements`).

### 5. Register imports + export

- `<name>/__init__.py`: export the backend, config, transformed-weights type, and
  `preprocess_mega_weights`.
- `backends/mega/kernel/__init__.py`: add `<name>` to the `from . import ...` so
  `@register_mega_kernel` runs on import (registration is import-triggered).
- `flashinfer/moe_ep/__init__.py`: re-export the config (e.g.
  `MyMegaMoeConfig`) and any `preprocess_*` helper for user imports.

### 6. Use it

```python
from flashinfer.moe_ep import MoEEpLayer, MegaConfig, MyMegaMoeConfig

layer = MoEEpLayer(
    bootstrap=..., fleet_params=...,
    weights=...,                       # canonical MoEWeightPack, required
    backend=MegaConfig(megakernel=MyMegaMoeConfig(intermediate_size=1024, top_k=4)),
)
out = layer.forward(tensors)
layer.destroy()
```

The raw megakernel config must be wrapped in `MegaConfig` — `MoEEpLayer` routes
`MegaConfig` → `MoEEpMegaLayer` → `create_mega_kernel(cfg)`, which looks up
`cfg.kernel_name` in `_MEGA_KERNEL_REGISTRY`.

## Fault tolerance

Enable with `FleetAlgoKnobFaultTolerance()` in `fleet_knobs`. Check support
first — it needs more than the backend being built:

```python
from flashinfer.moe_ep import supports_fault_tolerance
supports_fault_tolerance("nccl_ep")   # needs nccl4py with GroupConfig.enable_mask
                                      # AND a libnccl_ep exporting ncclEpMask*
supports_fault_tolerance("nixl_ep")   # true whenever the backend is staged
```

If `nccl_ep` returns False, upgrade the nccl4py wheel that ships
`libnccl_ep.so` and confirm it is the one actually loaded
(`python -m nccl show_versions`). The probe never raises, and the Fleet
constructor fails with the same diagnosis rather than waiting for a real fault.

### Recovery state machine

```
HEALTHY ──query_fault()──> FAULT_DETECTED      (local; quiesce in-flight combine/complete)
   ──reconcile_active_mask()──> RECONCILING     (store-collective, tolerates dead ranks)
   ──clear_faults(readmit=False)──> DEGRADED    (serving continues; dead ranks' tokens dropped)

DEGRADED ──peer came back──> clear_faults(readmit=True)    [COLLECTIVE, blocking] ──> HEALTHY
DEGRADED ──peer gone for good──> update_topology(...)      [COLLECTIVE, blocking] ──> HEALTHY
```

### Ordering rules (all load-bearing)

0. **The steady state is read-only.** The transport discovers the fault: its
   kernel times out on the peer and masks it. The application's job is to
   *notice* — `query_fault()`, then `query_active_mask()`. `set_active_mask()`
   is **not** a normal-path call; it exists so that reconciliation can impose an
   agreed vector once survivors have compared views, and most callers should
   only reach it indirectly through `reconcile_active_mask()`.

1. **If you do write the mask, write it only between iterations.** Both
   transports read it *live* from the dispatch/combine kernels, so changing it
   while a collective is in flight is a race that can produce a half-masked
   dispatch. Retire the iteration's `combine` (and `Handle.complete()` under
   `HandleAlgoKnobSplitOperation`) first.

2. **All survivors must reconcile in the same iteration slot** — they must pass
   the same `active_mask_epoch`. Make the host's fault poll a synchronous
   decision point.

3. **`clear_faults(readmit=True)` and `update_topology()` are alternatives, not
   a sequence.**

   * `clear_faults(readmit=True)` runs `ncclEpMaskClean` on the *same*
     communicator: it re-admits a rank that was merely delayed.
   * `update_topology()` destroys the group and builds a **new communicator**
     (a fresh `ncclComm_t`, unless you adopted one via
     `BootstrapConfig.nccl_comm`). That is the only way to add or replace a
     process — matching `nccl_ep.h`, which notes that rank replacement needs
     `ncclCommGrow` and a new EP group.

   Re-admitting after a rebuild is meaningless (the group it targeted is gone)
   and before one is wasted work. If you genuinely do both, `MaskClean` must
   come first, since it needs a live handle on the *current* group.

4. **No FT call may be issued during CUDA-graph capture** — but not for the
   reason you might expect. Neither transport compacts the surviving ranks'
   data layout, so **dispatch and combine are safe to capture**: they re-read
   the mask on every replay, and a rank that fails later is still skipped.
   Data from a masked rank is simply left in an unknown state with the error
   flag raised.

   What must not be captured is a *decision about* the fault state:

   * `query_fault()` is a host-side read, not stream work, so it cannot be
     captured at all. Inside a capture region it returns the capture-time
     answer, and the branch taken on it is frozen into the graph's structure —
     a graph that ignores faults forever because none had happened when it was
     recorded. This is the quietest failure of the three, which is why it is
     guarded even though it touches no stream.
   * `set_active_mask()` bakes the rank and value into the captured operation,
     so every replay re-applies that one mask.
   * `query_active_mask()` can only be consumed on the host, which needs a sync
     that capture forbids.

   All four raise on capture with a per-operation explanation. Call them from
   the host between replays.

### Backend asymmetries worth knowing

* **`nixl_ep` cannot evict a rank from the middle.** `disconnect_ranks`
  requires the removed ranks to form a *suffix*, so masked-and-degraded is the
  terminal state for a mid-list failure; only a highest-numbered failure can be
  shrunk away with `update_topology`.
* **`nccl_ep` re-admission under `EXPERT_MAJOR` warns.** `ncclEpMaskClean`
  computes its buffer-reset offsets assuming `RANK_MAJOR`, so re-admission may
  leave stale bytes in the LL staging buffer. Prefer degraded serving or a full
  `update_topology` rebuild there. (Degraded serving itself is sound under
  `EXPERT_MAJOR`.)
* **Growing past the topology capacity is rejected.** Every per-rank array is
  sized to the capacity at construction; pass
  `FleetAlgoKnobTopologyCapacity(n=<max ranks>)` up front.

### Dropped tokens — no renormalization

A masked rank's experts contribute nothing and combine does **not** renormalize
`topk_weights`. An affected token therefore comes out scaled by the surviving
weight fraction:

```
y_degraded[t] = y_healthy[t] * sum(topk_weights[t][k] for k where the owning rank is alive)
```

This is deliberate. Renormalizing implicitly would add an unconditional kernel
to every forward for a case that is almost never active, hide a serving-quality
event the operator must see, and divide by zero for a token whose entire top-k
landed on dead ranks.

**FlashInfer does not re-home the dead rank's experts either.** That is an
EPLB-style job and belongs to the framework: re-route the routing map away from
the failed rank, leave the rank masked, and keep serving on a partial mask —
no `update_topology()` and no new communicator required. The sequence is

```
query_fault() → reconcile_active_mask() → clear_faults(readmit=False) → keep going
```

where `clear_faults(readmit=False)` is purely "re-arm fault detection"
(`ncclEpErrorClear`); it does not touch the mask. FlashInfer's contribution is
to surface an agreed mask and the formula below; what you do with the routing is
yours.

To preserve magnitude yourself:

```python
alive = fleet.query_active_mask()                       # [world], 1 = active
owner = torch.arange(num_experts, device=d) // (num_experts // world_size)
ok = (topk_ids >= 0) & alive[owner[topk_ids.clamp_min(0)]].bool()
surviving_w = (topk_weights * ok).sum(-1)               # [num_tokens]
out.div_(surviving_w.clamp_min(1e-6).unsqueeze(-1))     # or drop requests below a threshold
```

### Being evicted

`reconcile_active_mask()` raises `MoEEpRankEvictedError` when the survivors
agreed *this* rank is dead — it stalled long enough for their kernels to give
up on it. It cannot apply that decision (a rank may not mask itself) and must
not keep serving (its peers stopped sending it tokens). Tear the worker down,
or rejoin through a fresh Fleet after the survivors call
`clear_faults(readmit=True)`.

### Running the FT tests

```bash
bash tests/moe_ep/run_tests.sh ft          # both backends, gated on support
```

Or directly (needs ≥4 GPUs):

```bash
# stalled-rank pytest half — every process survives
torchrun --nproc_per_node=4 -m pytest \
  tests/moe_ep/test_moe_ep_fault_tolerance_multirank.py -v \
  -m "nvep and gpu_4" --backend=nccl_ep      # then --backend=nixl_ep

# hard-kill half — a rank really dies, so judge by SMOKE_RESULT count, not exit code
torchrun --nproc_per_node=4 --max-restarts=0 \
  tests/moe_ep/smoke_ft_ep.py --backend nixl_ep
```

The kill half is a script rather than a pytest test on purpose: torchrun
reports the victim's non-zero exit, which would fail the survivors' pytest
session even when they behaved correctly. `run_tests.sh ft` counts
`SMOKE_RESULT:` lines (expects `nproc - 1`) and is **not** part of `run_all`.

The host-only protocol tests need no GPU at all:

```bash
pytest tests/moe_ep/test_fault_tolerance_reconcile.py \
       tests/moe_ep/test_fault_tolerance_api.py \
       tests/moe_ep/nccl_ep/test_mask_ffi.py -q
```
