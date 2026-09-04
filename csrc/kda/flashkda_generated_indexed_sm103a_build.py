#!/usr/bin/env python3
"""Deterministically rebuild the validated generated-program CUDA artifacts with NVRTC."""
import argparse
import hashlib
import json
import os
from pathlib import Path

UNITS = json.loads('[{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"aeffa91149fa9e6dc979d12e15fe7792e6490f953082aa3556022f0d4b117046","expected_cubin_size_bytes":47480,"id":"cuda-module-cubin-000","output":"generated_program/sm103a/cuda/modules/module-000/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-000/kernel.cu","source_sha256":"8ea3a770ef0e0b1eeeff59e2837b611e08f34a9eb53b83b8974f9ebf37445bad"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"d254464c3034bd449597f328e537373ce8546651adc3a073cd5f66e9d8dce5c2","expected_cubin_size_bytes":41112,"id":"cuda-module-cubin-001","output":"generated_program/sm103a/cuda/modules/module-001/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-001/kernel.cu","source_sha256":"faf2c9699aaa471a8924581e3bda8238c035f4d56ae88559d79ec26c31b044bf"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"4bb894c2a99a27f5350b96d0ca713c46c927bca1d759d01dda5763ab75f64017","expected_cubin_size_bytes":84344,"id":"cuda-module-cubin-002","output":"generated_program/sm103a/cuda/modules/module-002/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-002/kernel.cu","source_sha256":"110966dd1aefda3054b43e60c95f8892a09b67e6a2532dcdc9efd1976f94b10c"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"40aff46dac6eb1319bb58960345291448eb281e173fda47e453544485c945a73","expected_cubin_size_bytes":89776,"id":"cuda-module-cubin-003","output":"generated_program/sm103a/cuda/modules/module-003/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-003/kernel.cu","source_sha256":"eda6274866b3092e93ea0b2eac66b890d4480f7b33f81dbb8f17b9c2fb866bdf"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"d01662c003b000f8d441fa2bdc4b6dca81939e27b0a0e47e317aa6ad7dad8cb8","expected_cubin_size_bytes":88104,"id":"cuda-module-cubin-004","output":"generated_program/sm103a/cuda/modules/module-004/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-004/kernel.cu","source_sha256":"0c665a2b3451084f5a7681f38fc9a9ccd4143c1c952c6fcbf0292babd3550a0b"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"3923e3724c510d38c0edbec18720ae7c36fc1a13865770ae702589745007b2cd","expected_cubin_size_bytes":88144,"id":"cuda-module-cubin-005","output":"generated_program/sm103a/cuda/modules/module-005/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-005/kernel.cu","source_sha256":"fa94640bf3ee4be7decd67b9bceb2844519e859deca87219c85f5b46d6113160"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"e9e61c217bff926a050fe4702f6b73cdf60d1ce56578854ec113bb4cc48cf6af","expected_cubin_size_bytes":89728,"id":"cuda-module-cubin-006","output":"generated_program/sm103a/cuda/modules/module-006/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-006/kernel.cu","source_sha256":"cd6f5978ef8afcb68820396434763dd70134d7af69a2a7b9605982c6745c386d"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"0bb4a13c726fb00e2d97aaf014e5dcc75b38bfe4506477c475cb76296cc79122","expected_cubin_size_bytes":95344,"id":"cuda-module-cubin-007","output":"generated_program/sm103a/cuda/modules/module-007/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-007/kernel.cu","source_sha256":"c3f2291b92282b3676f730e07a8919ace0ea2ce4e81cccfaccd2f342ce55c452"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"e47472e5745da51f0c3df891003d41b1b1dfcd010293f7252401f5dde2bf0627","expected_cubin_size_bytes":88720,"id":"cuda-module-cubin-008","output":"generated_program/sm103a/cuda/modules/module-008/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-008/kernel.cu","source_sha256":"601959ba75b51afd39bea58fffaf8f3b117345f32719d1caffa471a37b1de741"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2251e378361e016dab3831fcc2f54ac4878a4e7fe05d8636a5dcb2ba728225c0","expected_cubin_size_bytes":98224,"id":"cuda-module-cubin-009","output":"generated_program/sm103a/cuda/modules/module-009/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-009/kernel.cu","source_sha256":"1dd8074be70d39a6ca24fa4c99a8a4e0b85e6e00254d3659a2e528acfe3378af"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"93388e7988dd70ead1a67bcc3fce53fb92b9952d376e31324bbbdef0a2048a2a","expected_cubin_size_bytes":100864,"id":"cuda-module-cubin-010","output":"generated_program/sm103a/cuda/modules/module-010/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-010/kernel.cu","source_sha256":"c7439df59c510ff3b127ef89406cf41135e4a0234edf3cc9eafa601597d458ec"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"ff169d5b7945207aeeec61b567ce144cfb42e1f4781ff95a127c006e018d8b47","expected_cubin_size_bytes":95296,"id":"cuda-module-cubin-011","output":"generated_program/sm103a/cuda/modules/module-011/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-011/kernel.cu","source_sha256":"a70981955cdf820ead5d5118e140c043369ff6f773255f6e8f8f6bd1fb8947f5"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"77d6b40c856e3d0095574d73bdd45752bd4db366117f781578dff83d8f7d5734","expected_cubin_size_bytes":89656,"id":"cuda-module-cubin-012","output":"generated_program/sm103a/cuda/modules/module-012/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-012/kernel.cu","source_sha256":"71b662db3ba97d34bb25d3e72fb0e6bf33c031096667ee49b3586041a0cad972"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"c0d44177225f71b89f9e6143c8a75b9938041ced0676396f8c82f4e6db860619","expected_cubin_size_bytes":89696,"id":"cuda-module-cubin-013","output":"generated_program/sm103a/cuda/modules/module-013/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-013/kernel.cu","source_sha256":"a0326e7b6b767b6196a3be8422b8053908b5ae6049bfb5ff0b05c7484a6f7599"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"2e4665b127a38a4961712232e79dce60471158c938c26b87c868037c276514f7","expected_cubin_size_bytes":98272,"id":"cuda-module-cubin-014","output":"generated_program/sm103a/cuda/modules/module-014/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-014/kernel.cu","source_sha256":"31d1ca889c7b01d1ad669b6663018a946d539f9628fef289800c2e82e18a82a8"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"92230af4f8af931668bb49cf3dcf71c3bbf89eacc895d82e5b37ff83db191eeb","expected_cubin_size_bytes":72872,"id":"cuda-module-cubin-015","output":"generated_program/sm103a/cuda/modules/module-015/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-015/kernel.cu","source_sha256":"cfb4b91d17f439741a619d08e70d04e503646b26840e063357b1da6fda0db624"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"6ed6eb85af81b463fcc657bfe3e790a60ecdfcd49c19379d568e16a7077644d7","expected_cubin_size_bytes":66304,"id":"cuda-module-cubin-016","output":"generated_program/sm103a/cuda/modules/module-016/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-016/kernel.cu","source_sha256":"b6704253ab64f31161e3216f4ddd2d99d687bfd3527a810436b388b680bcfd11"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"a2c6d7dbeb64873bc0620bae8c67f9f288f91551aad6afbf40c747f02e0bb9f0","expected_cubin_size_bytes":64320,"id":"cuda-module-cubin-017","output":"generated_program/sm103a/cuda/modules/module-017/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-017/kernel.cu","source_sha256":"d786344ae12fc6fa9cd289a51a7858f671b81e2611657e5506b01de43d2805d6"},{"architecture":"sm_103a","compile_options":["--use_fast_math"],"expected_cubin_sha256":"18771c814170941af02393446a6606a0a9e87cec42daefa4b0e0fe866edaf84f","expected_cubin_size_bytes":35776,"id":"cuda-module-cubin-018","output":"generated_program/sm103a/cuda/modules/module-018/kernel.cubin","source":"generated_program/sm103a/cuda/modules/module-018/kernel.cu","source_sha256":"f0908ebbbc660ffbc106c05d6814a82fbd6ed3792406133f90315e9d72cec505"}]')
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
