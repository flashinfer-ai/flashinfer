#!/usr/bin/env python3
"""Deterministically rebuild the validated generated-program CUDA artifacts with NVRTC."""
import argparse
import hashlib
import json
import os
from pathlib import Path

UNITS = json.loads('[{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"003dad2b4b8c9c153bf4b463d5670cbe170946ee77255cc9bfdb6bf136b4379d","expected_cubin_size_bytes":46432,"id":"cuda-module-cubin-000","output":"generated_program/sm100a/cuda/modules/module-000/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-000/kernel.cu","source_sha256":"8ea3a770ef0e0b1eeeff59e2837b611e08f34a9eb53b83b8974f9ebf37445bad"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a696d955a4623d922ee6986cfc24c8d4dfaa5eef18936258992f6aae90609a9f","expected_cubin_size_bytes":41080,"id":"cuda-module-cubin-001","output":"generated_program/sm100a/cuda/modules/module-001/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-001/kernel.cu","source_sha256":"faf2c9699aaa471a8924581e3bda8238c035f4d56ae88559d79ec26c31b044bf"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"f814937c072b4b5daec3c2b2cb69e3b17bbf9fbeb4aaad0e2f0baf50935833d1","expected_cubin_size_bytes":84440,"id":"cuda-module-cubin-002","output":"generated_program/sm100a/cuda/modules/module-002/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-002/kernel.cu","source_sha256":"110966dd1aefda3054b43e60c95f8892a09b67e6a2532dcdc9efd1976f94b10c"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"00a9fed25e253e95a13245c4b326819842bef9ee7c1e277a8fa0a7612b60f533","expected_cubin_size_bytes":92176,"id":"cuda-module-cubin-003","output":"generated_program/sm100a/cuda/modules/module-003/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-003/kernel.cu","source_sha256":"eda6274866b3092e93ea0b2eac66b890d4480f7b33f81dbb8f17b9c2fb866bdf"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2bb5266a67d6f81682c9c3dd41c9fdbdf48084ff1952aada5ef8f37091984c42","expected_cubin_size_bytes":89480,"id":"cuda-module-cubin-004","output":"generated_program/sm100a/cuda/modules/module-004/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-004/kernel.cu","source_sha256":"0c665a2b3451084f5a7681f38fc9a9ccd4143c1c952c6fcbf0292babd3550a0b"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"f9ec08dbb778d534fce9af83c15123f2687e1b7f5fc3eebe0f0f76b56cd2f3ab","expected_cubin_size_bytes":90552,"id":"cuda-module-cubin-005","output":"generated_program/sm100a/cuda/modules/module-005/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-005/kernel.cu","source_sha256":"fa94640bf3ee4be7decd67b9bceb2844519e859deca87219c85f5b46d6113160"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2f90d924f8f49f0ad99e5f2f92c5aeb34c7829028b990506212e03215e082731","expected_cubin_size_bytes":92128,"id":"cuda-module-cubin-006","output":"generated_program/sm100a/cuda/modules/module-006/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-006/kernel.cu","source_sha256":"cd6f5978ef8afcb68820396434763dd70134d7af69a2a7b9605982c6745c386d"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"6fcb1a7b3d601baf0a78bc2e7ea33f1e2ed9abe517db62d43a3efa7828500206","expected_cubin_size_bytes":97680,"id":"cuda-module-cubin-007","output":"generated_program/sm100a/cuda/modules/module-007/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-007/kernel.cu","source_sha256":"c3f2291b92282b3676f730e07a8919ace0ea2ce4e81cccfaccd2f342ce55c452"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"40403edc8a19c5f1efc5e23450b520a6201bdcb56a1890eb1a63090c2dc20d74","expected_cubin_size_bytes":86040,"id":"cuda-module-cubin-008","output":"generated_program/sm100a/cuda/modules/module-008/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-008/kernel.cu","source_sha256":"601959ba75b51afd39bea58fffaf8f3b117345f32719d1caffa471a37b1de741"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"5ea870a1ba08643472503b72ce5df0d7fd608f8a02fc1112eaeaa39203a870e5","expected_cubin_size_bytes":101184,"id":"cuda-module-cubin-009","output":"generated_program/sm100a/cuda/modules/module-009/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-009/kernel.cu","source_sha256":"c7439df59c510ff3b127ef89406cf41135e4a0234edf3cc9eafa601597d458ec"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"df68bd8a8bfed24559818d3db297180acb3f560f32d098271d1e21e530511bdb","expected_cubin_size_bytes":96608,"id":"cuda-module-cubin-010","output":"generated_program/sm100a/cuda/modules/module-010/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-010/kernel.cu","source_sha256":"a70981955cdf820ead5d5118e140c043369ff6f773255f6e8f8f6bd1fb8947f5"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"dd95c2dee79e4fbe9b729c13e4986e9890fe2c0b53404ad39d53cc0939c6dd5e","expected_cubin_size_bytes":91032,"id":"cuda-module-cubin-011","output":"generated_program/sm100a/cuda/modules/module-011/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-011/kernel.cu","source_sha256":"71b662db3ba97d34bb25d3e72fb0e6bf33c031096667ee49b3586041a0cad972"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"53fe42fdd7fd705a5a1d456761b28915d2656929b5bcdec9e01f0d45903490e7","expected_cubin_size_bytes":92104,"id":"cuda-module-cubin-012","output":"generated_program/sm100a/cuda/modules/module-012/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-012/kernel.cu","source_sha256":"a0326e7b6b767b6196a3be8422b8053908b5ae6049bfb5ff0b05c7484a6f7599"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ecc5f98e0836602d393d5418838b7f801d65d7f5e0b6f26c51f42278d2456811","expected_cubin_size_bytes":99504,"id":"cuda-module-cubin-013","output":"generated_program/sm100a/cuda/modules/module-013/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-013/kernel.cu","source_sha256":"31d1ca889c7b01d1ad669b6663018a946d539f9628fef289800c2e82e18a82a8"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"d1700fc49ca2f8773eb62f1f15ab877c676c48e665bb67037dcc10e7b7d802da","expected_cubin_size_bytes":64304,"id":"cuda-module-cubin-014","output":"generated_program/sm100a/cuda/modules/module-014/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-014/kernel.cu","source_sha256":"d786344ae12fc6fa9cd289a51a7858f671b81e2611657e5506b01de43d2805d6"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a58d9fcc8efa05c2afc46a00ce20926336e04288f7b5150cacbe3be8cc7890a3","expected_cubin_size_bytes":96544,"id":"cuda-module-cubin-015","output":"generated_program/sm100a/cuda/modules/module-015/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-015/kernel.cu","source_sha256":"9478fb1b499b09b4770a4e333a8e7ed56cfcfe36e7b636f5ca97d1f0d9652dee"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"4fcb6a3825df357b1e935df2af6e2f3207ca19ec24af1d0c0dc9a1c9d5d863c3","expected_cubin_size_bytes":101336,"id":"cuda-module-cubin-016","output":"generated_program/sm100a/cuda/modules/module-016/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-016/kernel.cu","source_sha256":"559e2020ecaa997e03d593852e21e6075078ffe98a105201f19cd9557e5a0081"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ae4578965256b9299fe088c504988b825ba6f87e3383ecee36d7f8aaeced5b62","expected_cubin_size_bytes":35776,"id":"cuda-module-cubin-017","output":"generated_program/sm100a/cuda/modules/module-017/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-017/kernel.cu","source_sha256":"f0908ebbbc660ffbc106c05d6814a82fbd6ed3792406133f90315e9d72cec505"}]')
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
