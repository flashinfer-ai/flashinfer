<h1 align="center">
FlashInfer+ROCm: An AMD ROCm port of FlashInfer
</h1>

FlashInfer+ROCm is a port of the [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
library to add support for AMD GPUs. The project is in its early
active stage of development and does not yet support all functionalities
implemented upstream. The feature support matrix lists currently supported
features.

To determine which upstream version a specific FlashInfer+ROCm release is
based on, please refer to the release tag. The versioning convention,
`<upstream_version>+rocm`, directly links each of the FlashInfer+ROCm releases
to a corresponding upstream tag. For example, the `0.2.5+rocm` release is
synchronized with the upstream `v0.2.5` tag.


**Feature Support Matrix**
| Kernel Type | FP16 / BF16 | FP8 (E4M3, E5M2) | Notes |
| :--- | :---: | :---: | :--- |
| **Decode Attention** | ✅ | WIP | Supports MHA, GQA, and MQA variants. |
| **Prefill Attention** | WIP | WIP | MHA/GQA/MQA support is a work in progress. |
| **Cascade** | WIP | WIP | not yet ported. |
| **MLA** | TBD | TBD | not yet ported. |
| **POD** | TBD | TBD | not yet ported. |
| **Positional Encoding** | TBD | TBD | LLaMA RoPE is supported. |
| **Sampling** | TBD | TBD | Top-K/Top-P sampling is not yet ported. |
| **Normalization** | TBD | TBD | RMS-Norm/Layer-Norm is not yet ported. |


**GPU Support**
| Model | Architecture |
|---|---|
| MI300x | CDNA3 |

**ROCm Support**

6.3.2, 6.4.1

**Docker image compatibility**

| Docker Image | ROCm | Flashinfer | PyTorch |
|---|---|---|---|
| TBD | 6.4.1 | 0.2.5 | 2.7.1


Table of Contents
=================

   * [Development Setup Inside a Pre-built Docker Container](#manual-setup)
      * [Step 0: Docker pull](#step-0-docker-pull)
      * [Step 1: Setting up micromamba](#step-1-setting-up-micromamba)
      * [Step 2: Installing Flashinfer](#step-2-installing-flashinfer)
      * [Step3: Verifying Installation](#step3-verifying-installation)
      * [Configure C++ Tests](#configure-c-tests)
      * [Configure PyTest](#configure-pytest)
   * [Using ROCm Dockerfile](#using-rocm-dockerfile)
      * [Step 1: Clone the repository](#step-1-clone-the-repository)
      * [Step 2: Docker build](#step-2-docker-build)
      * [Step 3: Using the container](#step-3-using-the-container)

## Development Setup Inside a Pre-built Docker Container
The recommended docker image to setup a development environment for
FlashInfer+ROCm is `rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1`

### Step 0: Docker pull
```bash
docker pull rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
```
```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
--group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--ipc=host --shm-size 128G --name=<container name> rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
```

### Step 1: Setting up micromamba
Run the following command to install micromamba and set it up inside `bash`
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```
After installing micromamba a custom environment for FlashInfer+ROCm
development should be setup. The micromamba environment is used to only manage 
the Python version, rest of the dependencies are installed using `pip`.

```bash
# Create a micromamba env for Python 3.12
micromamba create -n <environment_name> python=3.12 -c conda-forge --override-channels
# Activate the environment
micromamba activate <environment_name>
# Install added dependencies using pip
pip install setuptools-scm scikit-build-core pytest numpy cmake ninja pybind11
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```
### Step 2: Installing Flashinfer into the Docker container

Clone the latest trunk from https://github.com/ROCm/flashinfer.

```bash
git clone https://github.com/ROCm/flashinfer
cd flashinfer/
```
The Flashinfer+ROCm library can be built in two ways: with ahead-of-time (AOT) 
compiled kernels and without any AOT kernels.

Building the library with AOT kernels will take more time and local disk space 
as several common configurations of the core Flashinfer kernels are built 
during installation.

When building without AOT compilation, every kernel will be just-in-time (JIT) 
compiled at the time of first use.

* Instructions to build with AOT are as follows:
```bash
FLASHINFER_HIP_ARCHITECTURES=gfx942 FLASHINFER_AOT_TORCH_EXTS=ON python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
cd dist
pip install flashinfer-*.whl
```
* Instructions to build using JIT requires setting the FLASHINFER_AOT_TORCH_EXTS build flag to OFF.

```bash
FLASHINFER_HIP_ARCHITECTURES=gfx942 python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
cd dist
pip install flashinfer-*.whl
```
**Note:** The `--no-deps` flags assumes that all require dependencies are already available in the build environment. Otherwise, refer the earlier steps to install required packages. If building without first installing all Python and build dependencies, the `--no-deps` flag should be omitted. In that case, the build step will download all needed dependencies.

Development mode or editable installs (PEP 660) is supported and can be used
with both AOT and JIT builds of the package. To setup an editable install, follow these steps:

```bash
FLASHINFER_HIP_ARCHITECTURES=gfx942 python -m pip install --no-build-isolation -ve.
```
### Step3: Verifying Installation
A convenience script is provided inside the example directory that runs two HIPified kernels from Flashinfer: `SingleDecodeWithKVCache` and `BatchDecodeWithKVCache` and verifies the correctness of the generated results.

Following are the instructions to run the script:
```bash
cd examples/
python test_batch_decode_example.py
```
If Flashinfer+ROCm was installed without AOT kernels, the output should look as follows:
```
Failed to import __aot_prebuilt_uris__: No module named 'flashinfer.__aot_prebuilt_uris__'
2025-07-23 21:45:24,657 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
PASS
```
### Configure C++ Tests
Flashinfer+ROCm provides a C++ test suite to test all HIP kernels and C++ code. Installing Flashinfer does not automatically install the tests, instead these have to be configured separately.

1. To configure the rest of the test suite
```bash
cd flashinfer/libflashinfer/tests/hip
mkdir build && cd build/
cmake -DCMAKE_CXX_COMPILER:PATH=/opt/rocm/bin/amdclang++ -DFLASHINFER_INCLUDE_DIRS=<path to flashinfer includ dirs> ..
ninja
```
2. To run individual tests
```bash
./test_<target_test_name>
```
3. To run all tests
```bash
ctest
```
The output should look something like this
```
Test project /root/flashinfer/libflashinfer/tests/hip/build
    Start 1: MathTest
1/6 Test #1: MathTest .........................   Passed    3.40 sec
    Start 2: PosEncTest
2/6 Test #2: PosEncTest .......................   Passed    3.40 sec
    Start 3: CascadeTest
3/6 Test #3: CascadeTest ......................   Passed  985.27 sec
    Start 4: PageTest
4/6 Test #4: PageTest .........................   Passed  112.40 sec
    Start 5: SingleDecodeTest
5/6 Test #5: SingleDecodeTest .................   Passed   35.46 sec
    Start 6: BatchDecodeTest
6/6 Test #6: BatchDecodeTest ..................   Passed  556.81 sec

100% tests passed, 0 tests failed out of 6
```
### Configure PyTest
To run pytests, run the following helper script from the project root directory:

```bash
cd scripts/
./run_hip_tests.sh
```

## Using ROCm Dockerfile

### Step 1: Clone the repository
```bash
git clone https://github.com/ROCm/flashinfer
```

### Step 2: Docker build

```bash
docker build -f docker/Dockerfile.rocm_ci --target flashinfer_base -t <docker-image-tag> .
```
```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
--group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--ipc=host --shm-size 128G --name=<container name> <docker-image-tag>
```

### Step 3: Using the container
The docker container will come with micromamba pre-installed. It also builds
and pre-installs Flashinfer AOT version. To use Flashiner, first activate the
environment and then use the `flashinfer` package from Python.


```bash
micromamba activate flashinfer-py3.12-torch2.7.1-rocm6.4.1
```

```python
import torch
import flashinfer

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

# decode attention

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly
```
