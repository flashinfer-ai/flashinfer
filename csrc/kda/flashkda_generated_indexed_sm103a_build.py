#!/usr/bin/env python3
"""Deterministically rebuild the validated generated-program CUDA artifacts with NVRTC."""
import argparse
import hashlib
import json
import os
from pathlib import Path

UNITS = json.loads('[{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"aeffa91149fa9e6dc979d12e15fe7792e6490f953082aa3556022f0d4b117046","expected_cubin_size_bytes":47480,"id":"cuda-module-cubin-000","output":"generated_program/sm103a/cuda/modules/module-000/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-000/kernel.cu","source_sha256":"8ea3a770ef0e0b1eeeff59e2837b611e08f34a9eb53b83b8974f9ebf37445bad"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"d254464c3034bd449597f328e537373ce8546651adc3a073cd5f66e9d8dce5c2","expected_cubin_size_bytes":41112,"id":"cuda-module-cubin-001","output":"generated_program/sm103a/cuda/modules/module-001/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-001/kernel.cu","source_sha256":"faf2c9699aaa471a8924581e3bda8238c035f4d56ae88559d79ec26c31b044bf"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"1e513b5834edfd3b7bcd37a690d383eca039a8ed740f0251561922d6f744dc5d","expected_cubin_size_bytes":84448,"id":"cuda-module-cubin-002","output":"generated_program/sm103a/cuda/modules/module-002/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-002/kernel.cu","source_sha256":"4e1f645e53475be09bb1d0b8208fa1133923f4aeed912cd26eea247d9c2d9e40"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"b3a0545ac2b0699a03596bcc9abbea543d843a4e43b362c80b06c153203cd82f","expected_cubin_size_bytes":89776,"id":"cuda-module-cubin-003","output":"generated_program/sm103a/cuda/modules/module-003/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-003/kernel.cu","source_sha256":"ca8e0f5072c3e50a4d681f76992cc1e7641c1a7c7603257ca01093f8c0b3fcdd"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"0d3f441ee74a462081c59d32ec6f9605ca8706c49398b735c5f0c43b9730336e","expected_cubin_size_bytes":88104,"id":"cuda-module-cubin-004","output":"generated_program/sm103a/cuda/modules/module-004/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-004/kernel.cu","source_sha256":"a37a9ded66b7cc7eced3114ea1ecbb38d56580a2131ff108e7a977415a7b723c"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"e19b824c57c58953a7f8555e7bc557886d6bedca75fcb2b65a673664715aa3af","expected_cubin_size_bytes":88144,"id":"cuda-module-cubin-005","output":"generated_program/sm103a/cuda/modules/module-005/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-005/kernel.cu","source_sha256":"bcdef60f95de526016e61ba87804379f7e87b0a0dfb3abaf660da2f2b218344f"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"3bb10663922c2f1abbc3251f078105adc52bbb49e70eafb15274ac2a6d9e1f74","expected_cubin_size_bytes":89728,"id":"cuda-module-cubin-006","output":"generated_program/sm103a/cuda/modules/module-006/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-006/kernel.cu","source_sha256":"7414b5d8f4aa6cb6c0d49b07d6f9643374e5b007e7784d342df39a140fd17796"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ef9b920452716b1f598decda747fc38cae64fcf1c349a6241888952278ca0bae","expected_cubin_size_bytes":95360,"id":"cuda-module-cubin-007","output":"generated_program/sm103a/cuda/modules/module-007/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-007/kernel.cu","source_sha256":"2a5cbcb6c2a5e2368f0043e5fff5397647dab2c59b6a75272df6303b73dbfa31"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"e47472e5745da51f0c3df891003d41b1b1dfcd010293f7252401f5dde2bf0627","expected_cubin_size_bytes":88720,"id":"cuda-module-cubin-008","output":"generated_program/sm103a/cuda/modules/module-008/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-008/kernel.cu","source_sha256":"601959ba75b51afd39bea58fffaf8f3b117345f32719d1caffa471a37b1de741"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"489e683b9faa498dced24c4d99481fd37b138708115878bfa95836cc8f32855f","expected_cubin_size_bytes":98224,"id":"cuda-module-cubin-009","output":"generated_program/sm103a/cuda/modules/module-009/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-009/kernel.cu","source_sha256":"66707413331ca365dd8cf8ccba517c1e5ac8cb02bbfaa5b3b613ab05e3d80518"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ff1d2683adb9781887277ac22bd5bffde147ae8dded09e6904bc2158011dc234","expected_cubin_size_bytes":100872,"id":"cuda-module-cubin-010","output":"generated_program/sm103a/cuda/modules/module-010/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-010/kernel.cu","source_sha256":"5abd8c153cebd257ff3dca77a498440be40c38897fcacd2be1c05014d17d4630"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"17bb266b9ebb63c8a5431de46e4c8c2571943f0172d113d35189db45301a37fa","expected_cubin_size_bytes":95296,"id":"cuda-module-cubin-011","output":"generated_program/sm103a/cuda/modules/module-011/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-011/kernel.cu","source_sha256":"c9256a3a94b79d84089dd80a39d71d60d412978fe167023a443c70af891b9fe4"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"25c3e9b4cf8c22c9ebfc469f8042cb894d5380306b22165794a65f0dc109a4b2","expected_cubin_size_bytes":89656,"id":"cuda-module-cubin-012","output":"generated_program/sm103a/cuda/modules/module-012/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-012/kernel.cu","source_sha256":"d58cf2afd08b89b800f4b3d1aa0023aed5ef3dd56b15b83aadbe9607b2ba984c"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"bafa8563b111323b4ecd355c3586fa8c8f89dcda659a2f12ec3a20675e807145","expected_cubin_size_bytes":89696,"id":"cuda-module-cubin-013","output":"generated_program/sm103a/cuda/modules/module-013/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-013/kernel.cu","source_sha256":"1a3d383f39b43096faf599d7dc0ed9c636e98761cf4218dcda4600ec2460349e"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2e4665b127a38a4961712232e79dce60471158c938c26b87c868037c276514f7","expected_cubin_size_bytes":98272,"id":"cuda-module-cubin-014","output":"generated_program/sm103a/cuda/modules/module-014/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-014/kernel.cu","source_sha256":"31d1ca889c7b01d1ad669b6663018a946d539f9628fef289800c2e82e18a82a8"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"92230af4f8af931668bb49cf3dcf71c3bbf89eacc895d82e5b37ff83db191eeb","expected_cubin_size_bytes":72872,"id":"cuda-module-cubin-015","output":"generated_program/sm103a/cuda/modules/module-015/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-015/kernel.cu","source_sha256":"cfb4b91d17f439741a619d08e70d04e503646b26840e063357b1da6fda0db624"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"6ed6eb85af81b463fcc657bfe3e790a60ecdfcd49c19379d568e16a7077644d7","expected_cubin_size_bytes":66304,"id":"cuda-module-cubin-016","output":"generated_program/sm103a/cuda/modules/module-016/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-016/kernel.cu","source_sha256":"b6704253ab64f31161e3216f4ddd2d99d687bfd3527a810436b388b680bcfd11"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a2c6d7dbeb64873bc0620bae8c67f9f288f91551aad6afbf40c747f02e0bb9f0","expected_cubin_size_bytes":64320,"id":"cuda-module-cubin-017","output":"generated_program/sm103a/cuda/modules/module-017/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-017/kernel.cu","source_sha256":"d786344ae12fc6fa9cd289a51a7858f671b81e2611657e5506b01de43d2805d6"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"18771c814170941af02393446a6606a0a9e87cec42daefa4b0e0fe866edaf84f","expected_cubin_size_bytes":35776,"id":"cuda-module-cubin-018","output":"generated_program/sm103a/cuda/modules/module-018/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-018/kernel.cu","source_sha256":"f0908ebbbc660ffbc106c05d6814a82fbd6ed3792406133f90315e9d72cec505"}]')
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
