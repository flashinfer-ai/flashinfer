#!/usr/bin/env python3
"""Deterministically rebuild the validated generated-program CUDA artifacts with NVRTC."""
import argparse
import hashlib
import json
import os
from pathlib import Path

UNITS = json.loads('[{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"003dad2b4b8c9c153bf4b463d5670cbe170946ee77255cc9bfdb6bf136b4379d","expected_cubin_size_bytes":46432,"id":"cuda-module-cubin-000","output":"generated_program/sm100a/cuda/modules/module-000/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-000/kernel.cu","source_sha256":"8ea3a770ef0e0b1eeeff59e2837b611e08f34a9eb53b83b8974f9ebf37445bad"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a696d955a4623d922ee6986cfc24c8d4dfaa5eef18936258992f6aae90609a9f","expected_cubin_size_bytes":41080,"id":"cuda-module-cubin-001","output":"generated_program/sm100a/cuda/modules/module-001/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-001/kernel.cu","source_sha256":"faf2c9699aaa471a8924581e3bda8238c035f4d56ae88559d79ec26c31b044bf"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"49ef24e8d3925f298de5870c07f3ca78bf99678e0e47e8b9bcadd275ca4b5295","expected_cubin_size_bytes":84424,"id":"cuda-module-cubin-002","output":"generated_program/sm100a/cuda/modules/module-002/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-002/kernel.cu","source_sha256":"4e1f645e53475be09bb1d0b8208fa1133923f4aeed912cd26eea247d9c2d9e40"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"1e198157e64d4579a4f854368b02e9352639bc05444f9287f8ca7e583df3b799","expected_cubin_size_bytes":92192,"id":"cuda-module-cubin-003","output":"generated_program/sm100a/cuda/modules/module-003/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-003/kernel.cu","source_sha256":"ca8e0f5072c3e50a4d681f76992cc1e7641c1a7c7603257ca01093f8c0b3fcdd"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"da16e0beab33d8078477d902621f69b6d5b95c7a0f1031c595d3a66732349b27","expected_cubin_size_bytes":89480,"id":"cuda-module-cubin-004","output":"generated_program/sm100a/cuda/modules/module-004/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-004/kernel.cu","source_sha256":"a37a9ded66b7cc7eced3114ea1ecbb38d56580a2131ff108e7a977415a7b723c"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"549549c3d271265048a63ace06bff496c4d0e9e4bcf68af1cd6e806c701481d1","expected_cubin_size_bytes":90560,"id":"cuda-module-cubin-005","output":"generated_program/sm100a/cuda/modules/module-005/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-005/kernel.cu","source_sha256":"bcdef60f95de526016e61ba87804379f7e87b0a0dfb3abaf660da2f2b218344f"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a1231eabe9902892ec02ed11de47765da217466aeb75a08de4cadb841f1dbfde","expected_cubin_size_bytes":92128,"id":"cuda-module-cubin-006","output":"generated_program/sm100a/cuda/modules/module-006/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-006/kernel.cu","source_sha256":"7414b5d8f4aa6cb6c0d49b07d6f9643374e5b007e7784d342df39a140fd17796"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"5368db1a793350b7e12f69ae71e0ee7c08133a1d35d37ac9be54c6778235a970","expected_cubin_size_bytes":97696,"id":"cuda-module-cubin-007","output":"generated_program/sm100a/cuda/modules/module-007/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-007/kernel.cu","source_sha256":"2a5cbcb6c2a5e2368f0043e5fff5397647dab2c59b6a75272df6303b73dbfa31"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"40403edc8a19c5f1efc5e23450b520a6201bdcb56a1890eb1a63090c2dc20d74","expected_cubin_size_bytes":86040,"id":"cuda-module-cubin-008","output":"generated_program/sm100a/cuda/modules/module-008/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-008/kernel.cu","source_sha256":"601959ba75b51afd39bea58fffaf8f3b117345f32719d1caffa471a37b1de741"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2a7292fff492e2f84dee67ff28e0dfe80b8c1ccd27119dbd96c7399b633fb635","expected_cubin_size_bytes":101200,"id":"cuda-module-cubin-009","output":"generated_program/sm100a/cuda/modules/module-009/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-009/kernel.cu","source_sha256":"5abd8c153cebd257ff3dca77a498440be40c38897fcacd2be1c05014d17d4630"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"7dda8f5564e966838b4d84e68401ba98a7ff26837e9ef960747b5f53d54733f1","expected_cubin_size_bytes":96608,"id":"cuda-module-cubin-010","output":"generated_program/sm100a/cuda/modules/module-010/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-010/kernel.cu","source_sha256":"c9256a3a94b79d84089dd80a39d71d60d412978fe167023a443c70af891b9fe4"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"31df3757808e401373e7db70f95e6f85422fc13e73d2beea21c991bfe4dab1ec","expected_cubin_size_bytes":91032,"id":"cuda-module-cubin-011","output":"generated_program/sm100a/cuda/modules/module-011/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-011/kernel.cu","source_sha256":"d58cf2afd08b89b800f4b3d1aa0023aed5ef3dd56b15b83aadbe9607b2ba984c"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"583efbe6f8366db955d04dbbfa919678b8054310962f48867014aa92af330e97","expected_cubin_size_bytes":92120,"id":"cuda-module-cubin-012","output":"generated_program/sm100a/cuda/modules/module-012/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-012/kernel.cu","source_sha256":"1a3d383f39b43096faf599d7dc0ed9c636e98761cf4218dcda4600ec2460349e"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ecc5f98e0836602d393d5418838b7f801d65d7f5e0b6f26c51f42278d2456811","expected_cubin_size_bytes":99504,"id":"cuda-module-cubin-013","output":"generated_program/sm100a/cuda/modules/module-013/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-013/kernel.cu","source_sha256":"31d1ca889c7b01d1ad669b6663018a946d539f9628fef289800c2e82e18a82a8"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"d1700fc49ca2f8773eb62f1f15ab877c676c48e665bb67037dcc10e7b7d802da","expected_cubin_size_bytes":64304,"id":"cuda-module-cubin-014","output":"generated_program/sm100a/cuda/modules/module-014/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-014/kernel.cu","source_sha256":"d786344ae12fc6fa9cd289a51a7858f671b81e2611657e5506b01de43d2805d6"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"6b831e3c6160b696697fb16ac4c08829e8b1453b9f8e0f6e981639247ab3ea5d","expected_cubin_size_bytes":96528,"id":"cuda-module-cubin-015","output":"generated_program/sm100a/cuda/modules/module-015/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-015/kernel.cu","source_sha256":"2f98db2d132cd4ce42c8d37ee481c98d62833ce475d7c9a4f54aa4f57f962ce1"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"4d41d24b18c53b7c1ac1af247bd1bdd759305395e8a105ab7be3f713d733778d","expected_cubin_size_bytes":101304,"id":"cuda-module-cubin-016","output":"generated_program/sm100a/cuda/modules/module-016/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-016/kernel.cu","source_sha256":"ec4358d51cc3ed9c6f7353d18b3b179b4b242ad79f16707c60988d92582cd89d"},{"architecture":"sm_100a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ae4578965256b9299fe088c504988b825ba6f87e3383ecee36d7f8aaeced5b62","expected_cubin_size_bytes":35776,"id":"cuda-module-cubin-017","output":"generated_program/sm100a/cuda/modules/module-017/kernel.cubin","source":"generated_program/sm100a/cuda/modules/module-017/kernel.cu","source_sha256":"f0908ebbbc660ffbc106c05d6814a82fbd6ed3792406133f90315e9d72cec505"}]')
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
