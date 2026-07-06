# Runbook — NCCL-EP (FlashInfer) vs DeepEP on Pre-Nyx (pyxis/enroot, no Docker)

Linear, copy-paste steps to build the container images **with pyxis (no Docker daemon)** and run
the **NCCL-EP vs DeepEP** comparison in vLLM, on **1 node (8 GPU)** and **2 nodes (16 GPU)**.

- **NCCL-EP** = the `nccl.ep`-backed vLLM all2all backends `flashinfer_ep_low_latency` /
  `flashinfer_ep_high_throughput` (FlashInfer branch **`feat/nvep-default`** ≥ `fa09bc46` +
  vLLM branch **`feat/flashinfer-ep-all2all`** ≥ `ab1415e` — these commits carry the HT
  token-cap clamp, the HT recv-trim, the batched-DP cap membership and the fleet host-path
  caches; older refs reproduce the pre-optimization 2–6× gap). The standalone raw-`nccl.ep`
  backend is **not** in upstream vLLM (GitLab-fork only), so it's out of scope here.
- **DeepEP** = `deepep_low_latency` / `deepep_high_throughput`.
- Model: `Qwen/Qwen3-30B-A3B` (128 experts, bf16). Base: `nvcr.io/nvidia/pytorch:26.05-py3`
  (CUDA 13.2 — required; older stacks abort cross-node HT at `nccl_ep.cc:2884`).

Everything runs from a shared-FS work dir mounted `/host` inside the container.

```bash
# --- run once per shell ---
RW=/lustre/fsw/coreai_libraries_cudnn/agopal/agopal-moe-ep-verif   # a FRESH shared-FS work dir
ACCT=coreai_libraries_cudnn ; PART=batch
mkdir -p $RW/logs
```

Starting **from scratch** in a fresh `$RW`? Run §1 (clone) → §2 (build all three images,
nothing is reused) → §3 (single-node runs) → §4 (multi-node). Everything below is written to be
copy-pasted top-to-bottom into a shell that has already `export`ed `RW`/`ACCT`/`PART` above.

---

## 0. Why pyxis, not Docker

Pre-Nyx login/compute nodes have **no Docker daemon**. Container images are **enroot squashfs
(`.sqsh`) files**, built by running the install steps *inside* a container under `srun` and
snapshotting it with `--container-save`:

- `--container-image="nvcr.io#nvidia/<img>"` — enroot registry syntax (note the **`#`**, not
  `/`). A local image is just a path: `--container-image=$RW/foo.sqsh`.
- `--container-writable` — **required** so `apt`/`pip`/build outputs are captured by the save.
- `--container-save=$RW/out.sqsh` — writes the resulting image.
- `--container-mounts=$RW:/host` — the shared FS shows up at `/host` in the container.
- Whole-node allocation only — **do not pass `--gres`** (rejected on this cluster).
- A `--container-name` does **not** persist across separate `srun` jobs; always pass
  `--container-image=<...>.sqsh`.

---

## 1. Prerequisites — create the fresh dir and clone the repos

```bash
# (RW/ACCT/PART already exported above — e.g. RW=.../agopal-moe-ep-verif)
mkdir -p $RW/logs
# Pinned refs — these carry ALL the DP-EP fixes/optimizations the reference numbers were
# measured with (HT clamp+trim, batched-DP cap membership, fleet host-path caches):
git clone -b feat/nvep-default           https://github.com/Anerudhan/flashinfer.git $RW/flashinfer
git clone -b feat/flashinfer-ep-all2all  https://github.com/Anerudhan/vllm.git       $RW/vllm
git -C $RW/flashinfer submodule update --init --recursive   # cutlass, cccl, spdlog, nccl
# sanity: the perf-critical commits must be present
git -C $RW/flashinfer merge-base --is-ancestor fa09bc46 HEAD && echo "flashinfer ref OK"
git -C $RW/vllm       merge-base --is-ancestor ab1415e  HEAD && echo "vllm ref OK"
```

---

## 2. Build the three images with pyxis

> **Rebuilds vs `git pull`:** the images install flashinfer and vLLM as *editable* installs
> pointing at `/host/flashinfer` / `/host/vllm` — i.e. at **your `$RW` clones, resolved at
> runtime**. Python-only changes (all the perf fixes in the pinned refs are Python) take effect
> by just updating the clones (§1); **no image rebuild needed**. Rebuild only for dependency/
> native changes (e.g. a new nccl4py/nvidia-nccl pin).

### 2a. FlashInfer-EP base — `flashinfer-ep-pt2605.sqsh`
Runs `docker/install/build_flashinfer_ep_pytorch.sh` (pins nvidia-nccl-cu13 2.30.7 / nccl4py
0.3.1 / cuda-core 1.0.1 / cuda-bindings 13.2.0, then `BUILD_NCCL_EP=1 pip install -e .[nvep]`).

```bash
srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=03:00:00 \
  --container-image="nvcr.io#nvidia/pytorch:26.05-py3" --container-writable \
  --container-save=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
  bash -lc 'cd /host/flashinfer && bash docker/install/build_flashinfer_ep_pytorch.sh'
# sanity:
srun -A $ACCT -p $PART -N1 --container-image=$RW/flashinfer-ep-pt2605.sqsh --container-mounts=$RW:/host \
  bash -lc "python -c \"from flashinfer.moe_ep import available_backends; print(available_backends())\""
#   expect: ['nccl_ep']
```

### 2b. vLLM from source — `vllm-flashinfer-ep.sqsh`
Create `$RW/build_vllm.sh`:
```bash
cat > $RW/build_vllm.sh <<'EOS'
#!/bin/bash
set -eo pipefail
cd /host/vllm
python use_existing_torch.py                                  # strip torch==2.11 pin -> use NGC torch
TORCH_VER=$(python -c 'import torch;print(torch.__version__.split("+")[0])'); echo "torch==$TORCH_VER" > /tmp/tc.txt
command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"                          # vLLM bundles a Rust crate
PIP_CONSTRAINT=/tmp/tc.txt pip install --no-cache-dir -r requirements/build/cuda.txt
MAX_JOBS=32 VLLM_USE_PRECOMPILED=0 PIP_CONSTRAINT=/tmp/tc.txt \
  pip install --no-cache-dir --no-build-isolation -e . -v
# vLLM's deps pull `flashinfer-python` from PyPI, which UNINSTALLS/shadows our branch editable
# (the PyPI moe_ep lacks EpLayout / FleetAlgoKnobAllocator). Restore the branch editable so
# `import flashinfer` resolves to /host/flashinfer at runtime.
pip install --no-cache-dir --no-build-isolation --no-deps -e /host/flashinfer
python -c 'import vllm, flashinfer; from flashinfer.moe_ep import EpLayout, FleetAlgoKnobAllocator; \
  print("vllm", vllm.__version__, "| flashinfer", flashinfer.__file__)'
EOS
```
Build:
```bash
srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=03:00:00 \
  --container-image=$RW/flashinfer-ep-pt2605.sqsh --container-writable \
  --container-save=$RW/vllm-flashinfer-ep.sqsh --container-mounts=$RW:/host \
  bash /host/build_vllm.sh
```

### 2c. DeepEP — `vllm-fi-ep-deepep.sqsh`
Four things to get right on CUDA 13.2: (1) `uv` must be installed (the vLLM installer runs
`uv pip install --system`); (2) `UV_BREAK_SYSTEM_PACKAGES=1` (PEP-668); (3)
**`TORCH_CUDA_ARCH_LIST=10.0a`** (else DeepEP builds `sm_75` and `ptxas` rejects its
`elect`/`mbarrier`/`cp.async.bulk` kernels); and (4) **torch must load the 2.30.7 wheel
libnccl** — DeepEP asserts *torch nccl == the nvidia-nccl wheel*, but the NGC image also has a
system libnccl 2.30.4 that torch loads by default, so the script forces the wheel onto
`LD_LIBRARY_PATH` and bakes it into `/etc/profile.d`. (Symptom if skipped:
`AssertionError: Invalid NCCL versions: ...2.30.4 (loaded) v.s. ...nvidia/nccl/lib/libnccl.so.2
(expected)`.) Create `$RW/build_deepep.sh`:
```bash
cat > $RW/build_deepep.sh <<'EOS'
#!/bin/bash
export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_SYSTEM_PYTHON=1
set -eo pipefail
export TORCH_CUDA_ARCH_LIST="10.0a"                           # B200 / sm_100
command -v uv >/dev/null 2>&1 || pip install -q uv           # installer calls `uv pip install --system`
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"          # ensure `uv` is on PATH
TORCH_VER=$(python -c 'import torch;print(torch.__version__.split("+")[0])'); echo "torch==$TORCH_VER" > /tmp/tc.txt
PIP_CONSTRAINT=/tmp/tc.txt UV_CONSTRAINT=/tmp/tc.txt \
  bash /host/vllm/tools/ep_kernels/install_python_libraries.sh --workspace /host/ep_kernels_workspace
# guard: ensure the branch flashinfer editable is still the active install (nothing shadowed it)
pip install --no-cache-dir --no-build-isolation --no-deps -e /host/flashinfer
# DeepEP asserts torch's loaded NCCL == the nvidia-nccl wheel (2.30.7). The NGC image also has a
# system libnccl 2.30.4 that torch loads by default -> mismatch. Force the wheel's libnccl first,
# and bake it into /etc/profile.d so every `bash -lc` runtime run picks it up.
NCCL_LIB=$(python -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')
echo "export LD_LIBRARY_PATH=\"$NCCL_LIB:\${LD_LIBRARY_PATH:-}\"" > /etc/profile.d/zz_nccl_wheel.sh
export LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}"
# `import deep_ep` runs DeepEP's NCCL assert (loaded libnccl must == the wheel); if it prints OK
# the correct lib is loaded. NOTE: torch.cuda.nccl.version() reports torch's BUILD-time NCCL
# (cosmetic) — to see the actually-loaded runtime lib use ncclGetVersion on /proc/self/maps.
python -c 'import ctypes, torch; torch.cuda.init(); import deep_ep; \
  m=sorted({l.split()[-1] for l in open("/proc/self/maps") if "libnccl.so" in l}); \
  lib=ctypes.CDLL(m[0]); v=ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); \
  print("deep_ep OK; loaded libnccl", m, "runtime ncclGetVersion", v.value)'
EOS
```
Build:
```bash
srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=02:00:00 \
  --container-image=$RW/vllm-flashinfer-ep.sqsh --container-writable \
  --container-save=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
  bash /host/build_deepep.sh
```

> The DeepEP image contains vLLM + FlashInfer-EP + DeepEP, so you can run **all four backends
> from `vllm-fi-ep-deepep.sqsh`**. (The FI-EP image lacks DeepEP.)

### Warm the caches once (avoids a 30-min first-run JIT/cubin storm, esp. multi-node)
The very first vLLM forward JIT-compiles the CUTLASS MoE kernels (~267 units) and downloads
cubins. Persist them to `/host` so every later run (and every worker) reuses them:
```
export FLASHINFER_WORKSPACE_BASE=/host/fi_cache   # JIT cache
export FLASHINFER_CUBIN_DIR=/host/fi_cubins       # cubin cache
```
(These are already in the run commands below. Run one single-node throughput first to populate.)

---

## 3. Single node (8 GPU) — the comparison

Common per-run env (put at the top of each `bash -lc '...'`):
```
export HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 \
       FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins
```

### ⚠ 3.0 CRITICAL — you MUST use **data-parallel EP** or the all-to-all backend is a no-op

The `--all2all-backend` flag ONLY takes effect when vLLM selects the **modular EP** dispatch/combine
path. That selection is gated in `vllm/model_executor/layers/fused_moe/config.py`:

```python
@property
def use_all2all_kernels(self):
    return self.dp_size > 1 and self.use_ep          # ← dp_size MUST be > 1
```

and every backend predicate (`use_flashinfer_ep_ll_kernels`, `use_deepep_ll_kernels`, …) is
`use_all2all_kernels and all2all_backend == "<name>"`. With **pure `--tensor-parallel-size 8`
(dp_size = 1)** `use_all2all_kernels` is `False`, so `maybe_make_prepare_finalize()`
(`all2all_utils.py:148`) returns the **monolithic** `MoEPrepareAndFinalizeNoDPEPMonolithic` — the
experts run locally per rank and are reconciled with the ordinary **TP all-reduce**. The dispatch/
combine transport (nccl.ep for FlashInfer-EP, NVSHMEM for DeepEP) is **never launched**, and the
two backends produce byte-identical communication kernels. This is exactly what our first nsys
capture showed (both logged `Using MoEPrepareAndFinalizeNoDPEPMonolithic`).

**Correct config for a single 8-GPU node** (EP=8 across the 8 ranks, all2all engaged):
`--data-parallel-size 8 --enable-expert-parallel` (with TP=1, so DP×TP = 8 GPUs). Qwen3-30B-A3B in
bf16 fits per-GPU on B200/GB200 (non-expert weights replicated per DP rank; the 128 experts shard
16-per-rank). When this engages, the log shows the expert backend flip to a *batched* one
(`Using BATCHED_TRITON …` or `FlashInfer CUTLASS`) and `Using FlashInferEPLLPrepareAndFinalize`.

**⚠ Offline `vllm bench throughput` cannot take `--data-parallel-size` directly** — it errors
`Data parallel is only supported with external launcher mode with synchronous engine in offline
benchmark` (`benchmarks/throughput.py:914`). You must launch it under **`torchrun` (one process per
DP rank) with `--distributed-executor-backend external_launcher`**. Create a tiny driver once
(the `sys.path` scrub is required — torchrun prepends the script dir, and a sibling `vllm/` repo
dir would otherwise shadow the installed `vllm` package → `ModuleNotFoundError:
vllm.benchmarks.throughput`):
```bash
mkdir -p $RW/dprun
cat > $RW/dprun/driver.py <<'PY'
import sys
sys.path = [p for p in sys.path if p not in ("", "/host", "/host/dprun")]
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.benchmarks.throughput import add_cli_args, main
p = FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())
PY
```
Then every offline throughput/nsys run below uses `cd /tmp && torchrun --nproc_per_node=8
/host/dprun/driver.py <same args> --data-parallel-size 8 --distributed-executor-backend
external_launcher …` (each rank prints its own `Throughput:`; the deployment total ≈ sum of the 8).
`lm_eval` (GSM8K, §3b) does its own dataset-sharding DP and does **not** build a unified EP group,
so it stays an *accuracy* check on the monolithic path — the transport is validated by §3a
`--validate` and the §3e nsys dispatch/combine capture, not by GSM8K. To exercise the transport
end-to-end with GSM8K, use the **server** path instead (`vllm serve --data-parallel-size 8
--enable-expert-parallel --all2all-backend <B>` + `lm_eval --model local-completions
--model_args base_url=http://127.0.0.1:8000/v1/completions,...`; needs `pip install lm-eval[api]`).

**HT backends (`*_high_throughput`) require `--max-num-batched-tokens 8192`.** nccl_ep HT hard-caps
`max_dispatch_tokens_per_rank` at `MAX_SUPPORTED_TOKENS_PER_RANK=8192`; the flashinfer fleet clamps
to it, but a single forward with more than 8192 tokens per rank raises at dispatch — so cap the
scheduler to match. (DeepEP-LL, conversely, *rejects* `--max-num-batched-tokens 8192`, so only pass
it for the HT runs.)

**LL backends: do NOT pass `--max-num-batched-tokens` at all.** `flashinfer_ep_low_latency` is in
vLLM's `use_batched_dp_moe` set (with `deepep_low_latency`/`nixl_ep`), so when the flag is unset the
scheduler auto-caps to the 256-token batched-DP budget the BatchedExperts format needs — the padded
`[local_experts, max_tokens×world, N]` workspaces (and the LL transport slot buffers) are sized from
it, and an explicit large value silently makes every fill/activation/GEMM pad 32× (this was the
original 2–5× perf gap vs DeepEP; see results doc §1.1f).

**Always verify the path** after every run — the log must NOT say `Monolithic`:
```bash
grep -h "Using .*PrepareAndFinalize" $RW/logs/<log>            # expect one of:
#   Using FlashInferEPLLPrepareAndFinalize   (flashinfer_ep_low_latency)
#   Using FlashInferEPHTPrepareAndFinalize   (flashinfer_ep_high_throughput)
#   Using DeepEPLLPrepareAndFinalize         (deepep_low_latency)
#   Using DeepEPHTPrepareAndFinalize         (deepep_high_throughput)
# ✗ BUG (transport NOT exercised): Using MoEPrepareAndFinalizeNoDPEPMonolithic
```
(vLLM logs this via the oracle at `oracle/unquantized.py:332` — `logger.info_once("Using %s", …)`.)

### 3a. (optional) Correctness — EP dispatch/combine `--validate` @ world=8
```bash
srun -A $ACCT -p $PART -N1 --ntasks-per-node=8 --time=00:25:00 \
  --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
  bash -lc 'EP_SYNC=/host/sync_ht NCCL_GIN_TYPE=3 FLASHINFER_DISABLE_VERSION_CHECK=1 \
    bash /host/flashinfer/benchmarks/run_ep_matrix_one_pt.sh \
      --algorithm ht --layout fl --tokens 4096 --hidden 7168 --top-k 8 --experts 256 --validate'
# LL: --algorithm ll --layout em --tokens 128 --validate        (expect "... dispatch+combine OK")
```

### 3b. GSM8K accuracy (5-shot) — loop the 4 backends
```bash
for B in flashinfer_ep_low_latency flashinfer_ep_high_throughput deepep_low_latency deepep_high_throughput; do
  srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=01:00:00 \
    --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
    bash -lc "export HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins; \
      python -m pip install -q lm_eval; \
      lm_eval --model vllm --tasks gsm8k --num_fewshot 5 --batch_size auto \
        --model_args pretrained=Qwen/Qwen3-30B-A3B,tensor_parallel_size=8,enable_expert_parallel=True,all2all_backend=$B,trust_remote_code=True,max_model_len=4096,enforce_eager=True" \
    > $RW/logs/gsm8k_${B}.log 2>&1 &
done; wait
grep -H "flexible-extract\|strict-match" $RW/logs/gsm8k_*.log
```
> **GSM8K is an accuracy gate only — it does NOT exercise the all2all transport.** lm_eval's own
> `data_parallel_size` spins up independent replica engines (each `dp_size=1` internally ⇒
> monolithic path), so there is no single cross-rank EP group to dispatch through. Keep
> `tensor_parallel_size=8` here; it confirms end-to-end model accuracy is correct with the backend
> selected. The **transport** is validated by §3a (`--validate`), §3b′ below, and the §3e nsys capture.

### 3b′. GSM8K THROUGH the transport (real DP-EP server) — the one that exercises dispatch/combine
Serve a genuine DP-EP engine, then point lm_eval's OpenAI-compatible client at it. Unlike §3b this
runs GSM8K over the actual all2all transport. One srun starts the server, waits for `/health`, runs
the eval, and tears down. HT backends need `--max-num-batched-tokens 8192` (see §3.0).
```bash
for B in flashinfer_ep_low_latency flashinfer_ep_high_throughput; do
  case $B in *high_throughput*) CAP="--max-num-batched-tokens 8192";; *) CAP="";; esac
  srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=01:00:00 \
    --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
    bash -lc "export HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins; \
      python -m pip install -q lm_eval tenacity; \
      vllm serve Qwen/Qwen3-30B-A3B --port 8000 --data-parallel-size 8 --enable-expert-parallel \
        --all2all-backend $B $CAP --trust-remote-code --max-model-len 4096 --enforce-eager \
        > /host/logs/serve_${B}.log 2>&1 & SP=\$!; \
      for i in \$(seq 1 120); do curl -sf http://127.0.0.1:8000/health && break; \
        kill -0 \$SP || { echo SERVER_DIED; tail -30 /host/logs/serve_${B}.log; exit 1; }; sleep 10; done; \
      lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 --batch_size 1 \
        --model_args model=Qwen/Qwen3-30B-A3B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=128,tokenized_requests=False; \
      kill \$SP" \
    > $RW/logs/gsm8k_dpep_${B}.log 2>&1
  grep -H "flexible-extract\|strict-match\|Using .*PrepareAndFinalize" $RW/logs/gsm8k_dpep_${B}.log
done
# Expect (transport-exercised, at the pinned refs): LL 0.856/0.898, HT 0.857/0.898
# (flex/strict; ±0.01 run-to-run). Log says FlashInferEPLL/HTPrepareAndFinalize (NOT
# Monolithic). Needs lm-eval[api] (tenacity).
```

### 3c. Throughput — 4 backends × 3 ISL/OSL shapes
**Must pass `--dataset-name random`** or vLLM silently uses the `sonnet` default (~1024/128) and
ignores `--input-len`/`--output-len`.
Launched under `torchrun` + `external_launcher` (see §3.0) so DP-EP engages. Each of the 8 ranks
prints its own `Throughput:`; sum them (or ×8 the mean) for the deployment total.
```bash
for SHAPE in "128 128" "2048 128" "128 2048"; do set -- $SHAPE; ISL=$1; OSL=$2
 for B in flashinfer_ep_low_latency flashinfer_ep_high_throughput deepep_low_latency deepep_high_throughput; do
  # HT backends need the 8192 cap (nccl_ep HT limit); LL backends must NOT get it
  # (deepep_low_latency rejects --max-num-batched-tokens 8192). See §3.0.
  case $B in *high_throughput*) CAP="--max-num-batched-tokens 8192";; *) CAP="";; esac
  srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=01:15:00 \
    --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
    bash -lc "export HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins; cd /tmp; \
      torchrun --nproc_per_node=8 /host/dprun/driver.py --model Qwen/Qwen3-30B-A3B \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL --num-prompts 256 \
        --data-parallel-size 8 --distributed-executor-backend external_launcher \
        --enable-expert-parallel --all2all-backend $B $CAP \
        --trust-remote-code --max-model-len 4096 --enforce-eager" \
    > $RW/logs/tp_${B}_${ISL}x${OSL}.log 2>&1 &
 done; wait   # (or drop `wait` to run shapes concurrently if you have the nodes)
done
# Deployment total = SUM of the 8 per-rank Throughput lines; this prints it per run:
for f in $RW/logs/tp_*.log; do
  tot=$(grep -a "Throughput:" "$f" | grep -oE "[0-9.]+ total tokens/s" | grep -oE "^[0-9.]+" \
        | awk '{s+=$1} END{printf "%.0f", s}')
  pf=$(grep -a "Using .*PrepareAndFinalize" "$f" | head -1 | grep -oE "Using \w+")
  echo "$(basename $f)  sum_total_tok/s=$tot  [$pf]"   # [..] must be FlashInferEP…/DeepEP…, NOT Monolithic
done
```
> `--num-prompts 256` matches the reference matrix below exactly; larger NP (e.g. 1000) runs
> longer/steadier but shifts absolute numbers — keep it fixed when comparing backends.
> Requires the `$RW/dprun/driver.py` from §3.0. Plain `vllm bench throughput --data-parallel-size 8`
> does **not** work offline (it errors and tells you to use external launcher / serving).
> These are **eager-mode DP-EP** numbers (transport on the critical path); they are far below the
> CUDA-graph monolithic numbers — see the measured values + interpretation in
> `vllm_moe_ep_results_prenyx.md` §1.1e.

### 3d. (optional) Memory footprint
Same as 3c but `--num-prompts 8`, and grep the KV-cache line from the **full** stream (add
`--max-num-batched-tokens 8192` for HT backends, see §3.0):
```bash
srun ... bash -lc "... cd /tmp; torchrun --nproc_per_node=8 /host/dprun/driver.py \
  --model Qwen/Qwen3-30B-A3B --dataset-name random \
  --random-input-len 128 --random-output-len 128 --num-prompts 8 --data-parallel-size 8 \
  --distributed-executor-backend external_launcher \
  --enable-expert-parallel --all2all-backend $B $CAP --gpu-memory-utilization 0.9 \
  --trust-remote-code --max-model-len 4096 --enforce-eager 2>&1 \
  | grep -iE 'Available KV cache|GPU KV cache size|Maximum concurrency|Using .*PrepareAndFinalize'"
```

### 3e. Capture the launched kernels with nsys, and list the all-to-all kernels

Profile a **short** run under Nsight Systems, then dump the GPU kernel summary and filter for
the EP dispatch/combine (all-to-all) kernels. Use `--enforce-eager` (already set) so kernels are
launched individually (not hidden inside CUDA graphs), and a tiny `--num-prompts` so the report
is small. `nsys profile` follows the vLLM worker child processes, so all ranks' GPU kernels
land in one `.nsys-rep`.

> **Use the DP-EP `torchrun` form** (see §3.0). With `--tensor-parallel-size 8` the run takes the
> monolithic path and the `.nsys-rep` will contain **no** dispatch/combine kernels — which defeats
> the purpose. `nsys` wraps `torchrun`; `--trace-fork-before-exec=true` makes it follow the 8
> external-launcher rank processes so all ranks' kernels land in one `.nsys-rep`.

```bash
for B in flashinfer_ep_low_latency flashinfer_ep_high_throughput deepep_low_latency deepep_high_throughput; do
  case $B in *high_throughput*) CAP="--max-num-batched-tokens 8192";; *) CAP="";; esac  # see §3.0
  srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=00:45:00 \
    --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
    bash -lc "export HF_HOME=/host/hf_cache NCCL_GIN_TYPE=3 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins; cd /tmp; \
      nsys profile -t cuda,nvtx,nccl --force-overwrite true --sample=none --cpuctxsw=none \
        --trace-fork-before-exec=true -o /host/logs/nsys_dpep_${B} \
        torchrun --nproc_per_node=8 /host/dprun/driver.py --model Qwen/Qwen3-30B-A3B \
          --dataset-name random --random-input-len 128 --random-output-len 128 --num-prompts 64 \
          --data-parallel-size 8 --distributed-executor-backend external_launcher \
          --enable-expert-parallel --all2all-backend $B $CAP \
          --trust-remote-code --max-model-len 4096 --enforce-eager" \
    > $RW/logs/nsys_dpep_${B}.log 2>&1
done
# GATE: confirm every run actually took the modular EP path before trusting the kernel dump.
grep -H "Using .*PrepareAndFinalize" $RW/logs/nsys_dpep_*.log
#   want: FlashInferEPLL/HT... or DeepEPLL/HT...PrepareAndFinalize   ✗ reject: ...Monolithic
```

Dump the per-kernel GPU-time summary and **filter to the all-to-all / EP kernels**. `nsys` is
only inside the container (not on the login node), so run the parsing under `srun` too — it
needs no GPU, just reads the `.nsys-rep`. The `kern_*.txt` land in `$RW/logs/` (via `/host`), so
you can grep them afterward on the login node.
```bash
srun -A $ACCT -p $PART -N1 --ntasks-per-node=1 --time=00:15:00 \
  --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
  bash -lc 'for B in flashinfer_ep_low_latency flashinfer_ep_high_throughput deepep_low_latency deepep_high_throughput; do
    echo "############ $B — GPU kernel summary ############"
    nsys stats --report cuda_gpu_kern_sum --format table /host/logs/nsys_dpep_${B}.nsys-rep \
      | tee /host/logs/kern_dpep_${B}.txt | head -40
    echo "---- all-to-all / EP dispatch-combine kernels only ----"
    grep -iE "nccl.?ep|gin|gdaki|hybridep|deep_?ep|nvshmem|intranode|internode|dispatch|combine|moe.?ep" \
      /host/logs/kern_dpep_${B}.txt | tee /host/logs/kern_a2a_${B}.txt
    [ -s /host/logs/kern_a2a_${B}.txt ] || echo "  (NONE — check §3.0: run took the monolithic path?)"
  done'
# If nsys is not on PATH in the image: NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys | head -1); use "$NSYS" stats ...
```

**What to expect** — the exact symbols observed on this stack (Qwen3-30B-A3B, 8×DP-EP, 64 prompts,
eager):
- **FlashInfer-EP LL** (`nccl.ep`): `nccl_ep::internode_ll::dispatch` + `nccl_ep::internode_ll::combine`.
- **FlashInfer-EP HT** (`nccl.ep`, JIT): `nccl_ep_jit_ht_dispatch_kernel` + `nccl_ep_jit_ht_combine_kernel`
  + `nccl_ep_jit_ht_scan_kernel` (FLAT metadata) + `nccl_ep::hybridep::{dense_to_sparse_prob,
  convert_topk_to_routing_map,sparse_to_dense_prob}` helpers.
- **DeepEP LL** (`deep_ep`+NVSHMEM): `deep_ep::legacy::internode_ll::dispatch` +
  `…::internode_ll::combine` + one-time `nvshmemi_init_array_kernel`.
- **DeepEP HT** (`deep_ep` intranode): `deep_ep::legacy::intranode::{notify_dispatch,dispatch,
  combine,cached_notify_combine}` + `…::layout::get_dispatch_layout`.
- Both also show the shared **expert GEMM** and attention/norm kernels — those are *not*
  all-to-all; the grep above narrows to transport.

> **Reading the times (important):** the LL dispatch/combine kernels **busy-wait on the network**,
> so `Total`/`Avg`/`Max` are spin-dominated (multi-second `Max`, huge `StdDev`) — use the **median**
> per-launch. And **launch counts differ by design**: DeepEP LL issues ~2× the launches of
> FlashInfer-EP LL because DeepEP low-latency splits each dispatch/combine into a **send kernel +
> a deferred receive "hook"** (`low_latency_dispatch(return_recv_hook=True)` then `hook()`, for
> compute/comm overlap), whereas nccl.ep issues one fused kernel per call (`handle.dispatch()` +
> a `handle.complete()` stream-sync, not a 2nd launch). So compare DeepEP's (send+recv) sum vs
> FlashInfer's single launch, not launch-for-launch. HT is 1 launch each → directly comparable.
> Measured medians and the full table are in `vllm_moe_ep_results_prenyx.md` §1.1b; for a
> spin-free per-op latency use the standalone comm benchmark below.
- **If `kern_a2a_<B>.txt` is empty for a backend, the transport did not run** — the summary will
  instead be dominated by `multimem_all_reduce_kernel` / `vllm::cross_device_reduce_*` (TP
  all-reduce). That means the run fell back to the monolithic path (§3.0); fix the DP flag and
  re-capture. **This is the exact failure our first capture hit** (both backends identical, only
  TP all-reduce, no dispatch/combine).

Notes:
- **`nsys` lives only inside the container**, not on the Pre-Nyx login node — run *both*
  `nsys profile` and `nsys stats` under `srun --container-image=...`. `nsys stats` needs no GPU
  (it only reads the `.nsys-rep`), so it's a cheap short job.
- **Cleaner isolation (NCCL-EP only):** to see *just* the dispatch/combine kernels with no model
  noise, profile the standalone comm benchmark instead of vLLM:
  ```bash
  srun ... --ntasks-per-node=8 bash -lc 'EP_SYNC=/host/sync_ns NCCL_GIN_TYPE=3 \
    nsys profile -t cuda,nvtx,nccl -o /host/logs/nsys_epcomm_r${SLURM_PROCID} \
      bash /host/flashinfer/benchmarks/run_ep_matrix_one_pt.sh \
        --algorithm ht --layout fl --tokens 4096 --hidden 7168 --top-k 8 --experts 256 --iters 20'
  # then: nsys stats --report cuda_gpu_kern_sum /host/logs/nsys_epcomm_r0.nsys-rep
  ```
- If the `.nsys-rep` has no GPU kernels, nsys didn't follow the workers — re-run adding
  `--trace-fork-before-exec=true`, or use the per-rank standalone form above (`-o ..._r${SLURM_PROCID}`).
- Copy `$RW/logs/*.nsys-rep` locally to open the timeline in the Nsight Systems GUI if you want
  to see the dispatch→expert-GEMM→combine sequence visually.

---

## 4. Multi-node (2 nodes / 16 GPU)

Multi-node needs a **Ray cluster across both nodes' containers** + `--data-parallel-size 16
--enable-expert-parallel --distributed-executor-backend ray` + `NCCL_MNNVL_ENABLE=1` for the
cross-node EP fabric. **Use data-parallel EP, not `--tensor-parallel-size 16`** — same reason as
§3.0: TP-only ⇒ `dp_size=1` ⇒ monolithic path ⇒ `all2all_backend` ignored (and the cross-node
all-to-all you're trying to measure never runs).
Create `$RW/run_2node.sh` (rank-0 starts the Ray head + runs the bench; rank-1 joins and blocks):

```bash
cat > $RW/run_2node.sh <<'EOS'
#!/bin/bash
set -o pipefail
export HF_HOME=/host/hf_cache FLASHINFER_WORKSPACE_BASE=/host/fi_cache FLASHINFER_CUBIN_DIR=/host/fi_cubins
export NCCL_GIN_TYPE=3 NCCL_MNNVL_ENABLE=1
# force the 2.30.7 wheel libnccl (DeepEP needs torch nccl == wheel; runs via bash, not -lc)
NCCL_LIB=$(python -c 'import nvidia.nccl,os;print(os.path.join(list(nvidia.nccl.__path__)[0],"lib"))')
export LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}"
B=${BACKEND:-flashinfer_ep_low_latency}; ISL=${ISL:-128}; OSL=${OSL:-128}
python -m pip install -q ray 2>/dev/null || true
HEADF=/host/ray_head.$SLURM_JOB_ID; TMP=/host/raylog_${SLURM_JOB_ID}_${SLURM_NODEID}
if [ "${SLURM_NODEID:-0}" = "0" ]; then
  hostname -I | awk '{print $1}' > $HEADF
  ray start --head --port=6379 --num-gpus=8 --disable-usage-stats --temp-dir=$TMP
  for i in $(seq 1 60); do
    n=$(python -c 'import ray;ray.init(address="auto");print(int(ray.cluster_resources().get("GPU",0)))' 2>/dev/null|tail -1)
    [ "$n" = "16" ] && break; sleep 5; done; echo "cluster GPUs=$n"
  vllm bench throughput --model Qwen/Qwen3-30B-A3B --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL --num-prompts 1000 \
    --data-parallel-size 16 --enable-expert-parallel --all2all-backend $B \
    --distributed-executor-backend ray --trust-remote-code --max-model-len 4096 --enforce-eager
  ray stop; rm -f $HEADF
else
  for i in $(seq 1 60); do [ -f $HEADF ] && break; sleep 3; done; sleep 8
  ray start --address=$(cat $HEADF):6379 --num-gpus=8 --disable-usage-stats --temp-dir=$TMP --block
fi
EOS
```
Run (one task per node; warm `/host/fi_cubins` from a single-node run first):
```bash
for B in flashinfer_ep_low_latency flashinfer_ep_high_throughput deepep_low_latency deepep_high_throughput; do
  srun -A $ACCT -p $PART -N2 --ntasks-per-node=1 --time=01:00:00 \
    --container-image=$RW/vllm-fi-ep-deepep.sqsh --container-mounts=$RW:/host \
    --export=ALL,BACKEND=$B,ISL=128,OSL=128 \
    bash /host/run_2node.sh > $RW/logs/tp2n_${B}.log 2>&1
  grep -H "Throughput:" $RW/logs/tp2n_${B}.log
done
```

> ⚠ **Known issue (environmental, not the EP code):** in our runs the Ray 16-GPU cluster forms
> and vLLM launches, but engine-core init **stalls** — and a plain `--tensor-parallel-size
> 16` run **without** EP stalls identically. So it's the cluster's cross-node vLLM/NCCL bring-up,
> not FlashInfer-EP/DeepEP. If you hit it, debug the cross-node fabric first, e.g. export before
> the bench: `NCCL_DEBUG=INFO`, and set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` to the node's IB
> interfaces (`ibdev2netdev` / `ibv_devices`); verify a plain 2-node NCCL all-reduce works.

> ⚠ **Offline-DP caveat also applies here.** `vllm bench throughput --data-parallel-size 16` with
> the Ray backend hits the same offline-DP guard as §3.0 (it needs `external_launcher`, not `ray`).
> For a working 2-node DP-EP throughput run once the fabric is up, either launch the §3.0 driver
> under `torchrun --nnodes=2 --nproc_per_node=8 --distributed-executor-backend external_launcher`,
> or use the **server** path (`vllm serve --data-parallel-size 16 --enable-expert-parallel` +
> `vllm bench serve`). The Ray script above is kept as the cluster-bring-up reference.

---

## Expected reference numbers (8-GPU, Qwen3-30B-A3B; full detail in `vllm_moe_ep_results_prenyx.md`)

**DP-EP, transport-exercised** (the numbers to trust for a backend comparison; §1.1c–f).
Measured at the pinned refs (flashinfer `fa09bc46`, vLLM `ab1415e`), §3c command verbatim
(NP=256, eager, HT capped at 8192, LL uncapped):

| Backend | 128/128 | 2048/128 | 128/2048 |
|---|---|---|---|
| `flashinfer_ep_low_latency`  | **9,088** | **23,106** | **5,825** |
| `deepep_low_latency`         | 10,116 | 24,013 | 6,595 |
| `flashinfer_ep_high_throughput` | **6,797** | **45,224** | 3,795 |
| `deepep_high_throughput`     | 5,736 | 35,623 | 4,539 |

(total tok/s, sum of 8 ranks; expect ±3–5% run-to-run.) **FI-HT is ahead of DeepEP-HT by
19–27%** on 128/128 and 2048/128; FI-LL within 4–12% of DeepEP-LL; decode-heavy within
12–16% for both modes.
- **GSM8K over a real DP-EP server** (§3b′, flex/strict): FI-EP **LL 0.856/0.898**,
  **HT 0.857/0.898** — both ≥0.80, on par with the ~0.88 reference. Transport genuinely exercised.
- ⚠ These numbers **require the pinned refs** (§1) — the HT token clamp + recv-trim, the
  batched-DP cap membership and the fleet host-path caches. On older refs FI-EP lands 2–6×
  behind DeepEP and HT SIGABRTs at group-create (root causes + fixes: results doc §1.1c/§1.1f).
- Also required: HT runs get `--max-num-batched-tokens 8192`; LL runs get **no** such flag.

**Monolithic (`--tensor-parallel-size 8`, dp_size=1) — NOT a transport comparison** (§1.2, retained
for reference only): GSM8K strict ~0.89 all four; throughput 32.5k · 141k · 18.5k (FI) ≈ DeepEP
(both ran the identical TP-all-reduce path, so the ~1–2% closeness is an artifact); memory identical
150.45 GiB KV cache. The all2all backend had **no effect** here.
