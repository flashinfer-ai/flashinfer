#!/usr/bin/env python3
"""Deterministically rebuild the validated generated-program CUDA artifacts with NVRTC."""
import argparse
import hashlib
import json
import os
from pathlib import Path

UNITS = json.loads('[{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"003dad2b4b8c9c153bf4b463d5670cbe170946ee77255cc9bfdb6bf136b4379d","expected_cubin_size_bytes":46432,"id":"cuda-module-cubin-000","output":"generated_program/sm100a/cuda/modules/module-000/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-000/kernel.cu","source_sha256":"065811d37c28744f4a21dcd389e5a82d11b3714118f523f318a1829ad06aa2e3"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a696d955a4623d922ee6986cfc24c8d4dfaa5eef18936258992f6aae90609a9f","expected_cubin_size_bytes":41080,"id":"cuda-module-cubin-001","output":"generated_program/sm100a/cuda/modules/module-001/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-001/kernel.cu","source_sha256":"1387c46dfbabe52a31f112e000a69eaf718156492c535495954ce1a5df5171db"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"f814937c072b4b5daec3c2b2cb69e3b17bbf9fbeb4aaad0e2f0baf50935833d1","expected_cubin_size_bytes":84440,"id":"cuda-module-cubin-002","output":"generated_program/sm100a/cuda/modules/module-002/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-002/kernel.cu","source_sha256":"ccdb1dd4bf4b9ec93e457fdef800db37256bac4ce695f975f1a2af29a5ea7467"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"f9ec08dbb778d534fce9af83c15123f2687e1b7f5fc3eebe0f0f76b56cd2f3ab","expected_cubin_size_bytes":90552,"id":"cuda-module-cubin-003","output":"generated_program/sm100a/cuda/modules/module-003/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-003/kernel.cu","source_sha256":"18db0727338b4980643b9060930d68ac93031d51481154b728a8628bbc66013c"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"df68bd8a8bfed24559818d3db297180acb3f560f32d098271d1e21e530511bdb","expected_cubin_size_bytes":96608,"id":"cuda-module-cubin-004","output":"generated_program/sm100a/cuda/modules/module-004/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-004/kernel.cu","source_sha256":"fcbdd6b2e284a1e61b438d1c264b1cdb64eb61701ea5cdd66357d1223cac626a"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"6fcb1a7b3d601baf0a78bc2e7ea33f1e2ed9abe517db62d43a3efa7828500206","expected_cubin_size_bytes":97680,"id":"cuda-module-cubin-005","output":"generated_program/sm100a/cuda/modules/module-005/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-005/kernel.cu","source_sha256":"4c70cb07e6a55ab1c7e670df516a28e8a34fb3aba57831eaa51d21eae1058a5c"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"40403edc8a19c5f1efc5e23450b520a6201bdcb56a1890eb1a63090c2dc20d74","expected_cubin_size_bytes":86040,"id":"cuda-module-cubin-006","output":"generated_program/sm100a/cuda/modules/module-006/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-006/kernel.cu","source_sha256":"a89f830b90f7ae9fceb00beea092677805149b3058c2b05e5c1df653b846ca6b"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"5ea870a1ba08643472503b72ce5df0d7fd608f8a02fc1112eaeaa39203a870e5","expected_cubin_size_bytes":101184,"id":"cuda-module-cubin-007","output":"generated_program/sm100a/cuda/modules/module-007/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-007/kernel.cu","source_sha256":"e0cd0be9ab0eefaedc6df4982c1991e5c8e47621486bde7ed999785de8272d0b"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ecc5f98e0836602d393d5418838b7f801d65d7f5e0b6f26c51f42278d2456811","expected_cubin_size_bytes":99504,"id":"cuda-module-cubin-008","output":"generated_program/sm100a/cuda/modules/module-008/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-008/kernel.cu","source_sha256":"31199183bbe045a282b9f76bbedb9a587ce2347448380044ff816a237b6cf1a4"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2bb5266a67d6f81682c9c3dd41c9fdbdf48084ff1952aada5ef8f37091984c42","expected_cubin_size_bytes":89480,"id":"cuda-module-cubin-009","output":"generated_program/sm100a/cuda/modules/module-009/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-009/kernel.cu","source_sha256":"34a21af250ca6a5471e83a7e6c75e51b8694936bb582622f279c6411215af17f"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"dd95c2dee79e4fbe9b729c13e4986e9890fe2c0b53404ad39d53cc0939c6dd5e","expected_cubin_size_bytes":91032,"id":"cuda-module-cubin-010","output":"generated_program/sm100a/cuda/modules/module-010/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-010/kernel.cu","source_sha256":"aff7fd9576c6983bc1b359858429cfe9aa1cb4a5dd52ce230b9261e5b2f35c9d"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2f90d924f8f49f0ad99e5f2f92c5aeb34c7829028b990506212e03215e082731","expected_cubin_size_bytes":92128,"id":"cuda-module-cubin-011","output":"generated_program/sm100a/cuda/modules/module-011/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-011/kernel.cu","source_sha256":"d65ac94f9fe1bc17576a547f00ef2c3570bb58e1692e107e3fe3cf2dba2e6789"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"00a9fed25e253e95a13245c4b326819842bef9ee7c1e277a8fa0a7612b60f533","expected_cubin_size_bytes":92176,"id":"cuda-module-cubin-012","output":"generated_program/sm100a/cuda/modules/module-012/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-012/kernel.cu","source_sha256":"52572c74280676608760ef8bc8ea07a34eec4d4f8f16133740da4d3a9d417a2b"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"53fe42fdd7fd705a5a1d456761b28915d2656929b5bcdec9e01f0d45903490e7","expected_cubin_size_bytes":92104,"id":"cuda-module-cubin-013","output":"generated_program/sm100a/cuda/modules/module-013/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-013/kernel.cu","source_sha256":"fe786c3a41aaee4231b9526cb7062d1e1039476e9abae22f961465390088a210"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"d1700fc49ca2f8773eb62f1f15ab877c676c48e665bb67037dcc10e7b7d802da","expected_cubin_size_bytes":64304,"id":"cuda-module-cubin-014","output":"generated_program/sm100a/cuda/modules/module-014/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-014/kernel.cu","source_sha256":"8e815b6df2e15b68804aa56ba0c27951fc41977f6f417c8f824e59afa0ed6c01"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a58d9fcc8efa05c2afc46a00ce20926336e04288f7b5150cacbe3be8cc7890a3","expected_cubin_size_bytes":96544,"id":"cuda-module-cubin-015","output":"generated_program/sm100a/cuda/modules/module-015/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-015/kernel.cu","source_sha256":"2e3d7ad4296e1d13f7ec2b3fb6af1ead1fec553da65454cdfcd602af2dc447b4"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"4fcb6a3825df357b1e935df2af6e2f3207ca19ec24af1d0c0dc9a1c9d5d863c3","expected_cubin_size_bytes":101336,"id":"cuda-module-cubin-016","output":"generated_program/sm100a/cuda/modules/module-016/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-016/kernel.cu","source_sha256":"d4ed2d0667209ec14c0a91e6a69f365c3d4eb6230f1f6cebd338d4aaba64d60d"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ae4578965256b9299fe088c504988b825ba6f87e3383ecee36d7f8aaeced5b62","expected_cubin_size_bytes":35776,"id":"cuda-module-cubin-017","output":"generated_program/sm100a/cuda/modules/module-017/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-017/kernel.cu","source_sha256":"f0908ebbbc660ffbc106c05d6814a82fbd6ed3792406133f90315e9d72cec505"}]')
EXPECTED_TOOLCHAIN_IDENTITY = 'sha256:14f5d1246407924bbde1408b3c885e9be263d212c05e513aaca7f1d1941de0da'

def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

def sha256(payload):
    return hashlib.sha256(payload).hexdigest()

def check(code, label):
    if code != 0:
        raise RuntimeError(f"{label} failed with NVRTC code {code}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--include-dir", required=True, action="append", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    from cuda.bindings import nvrtc

    err, major, minor = nvrtc.nvrtcVersion()
    check(err, "nvrtcVersion")
    libraries = []
    mapped = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        raw = line.rpartition(" ")[2]
        if "libnvrtc" in Path(raw).name:
            mapped.add(Path(raw).resolve(strict=True))
    if not mapped:
        raise RuntimeError("cannot resolve the loaded NVRTC binary")
    for path in sorted(mapped):
        payload = path.read_bytes()
        libraries.append({"name": path.name, "sha256": sha256(payload), "size_bytes": len(payload)})
    libraries.sort(key=canonical)
    include_dirs = [str(path.resolve(strict=True)) for path in args.include_dir]
    toolchain = {
        "kind": "flashinfer.nvrtc_toolchain_identity",
        "nvrtc_version": [int(major), int(minor)],
        "loaded_libraries": libraries,
    }
    identity = "sha256:" + sha256(canonical(toolchain))
    if identity != EXPECTED_TOOLCHAIN_IDENTITY:
        raise RuntimeError(f"toolchain identity drifted: {identity}")

    outputs = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for unit in UNITS:
        source_path = args.source_root / unit["source"]
        source = source_path.read_bytes()
        if sha256(source) != unit["source_sha256"]:
            raise RuntimeError(f"source drifted: {source_path}")
        err, program = nvrtc.nvrtcCreateProgram(source, b"kernel.cu", 0, [], [])
        check(err, "nvrtcCreateProgram")
        try:
            options = [
                f"--gpu-architecture={unit['architecture']}",
                "-std=c++17",
                "-default-device",
            ]
            for include in include_dirs:
                options.append(f"-I{include}")
                cccl = Path(include) / "cccl"
                if (cccl / "cuda" / "std").exists():
                    options.append(f"-I{cccl}")
            options.extend(unit["compile_options"])
            encoded = [option.encode() for option in options]
            (err,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
            if err != 0:
                _, size = nvrtc.nvrtcGetProgramLogSize(program)
                log = b"\0" * size
                nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(log.decode(errors="replace").rstrip("\0"))
            err, size = nvrtc.nvrtcGetCUBINSize(program)
            check(err, "nvrtcGetCUBINSize")
            cubin = b"\0" * size
            (err,) = nvrtc.nvrtcGetCUBIN(program, cubin)
            check(err, "nvrtcGetCUBIN")
        finally:
            nvrtc.nvrtcDestroyProgram(program)
        if sha256(cubin) != unit["expected_cubin_sha256"] or len(cubin) != unit["expected_cubin_size_bytes"]:
            raise RuntimeError(f"rebuilt cubin identity drifted: {unit['id']}")
        destination = args.output_root / unit["output"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(cubin)
        outputs.append({"id": unit["id"], "path": unit["output"], "sha256": sha256(cubin), "size_bytes": len(cubin)})
    report = {"kind": "flashinfer.generated_program_cuda_build_report", "schema_version": 1, "toolchain": toolchain, "toolchain_identity": identity, "outputs": outputs, "passed": True}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical(report) + b"\n")

if __name__ == "__main__":
    main()
