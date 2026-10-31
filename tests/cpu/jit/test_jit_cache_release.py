import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def cuda_config_module():
    name = "_test_validate_cuda_versions"
    module = _load_module(name, REPO_ROOT / "ci" / "validate_cuda_versions.py")
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def wheel_index_module():
    name = "_test_update_whl_index"
    module = _load_module(name, REPO_ROOT / "scripts" / "update_whl_index.py")
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def release_verifier_module():
    scripts_dir = REPO_ROOT / "scripts"
    name = "_test_verify_jit_cache_provider_release"
    sys.path.insert(0, str(scripts_dir))
    module = _load_module(name, scripts_dir / "verify_jit_cache_provider_release.py")
    try:
        yield module
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop("jit_cache_provider_validation", None)
        sys.path.remove(str(scripts_dir))


@pytest.mark.cpu_only
def test_provider_release_matrices_are_explicit(cuda_config_module):
    config = json.loads((REPO_ROOT / "ci" / "cuda-versions.json").read_text())

    cuda_config_module.validate_cuda_config(config, REPO_ROOT)
    provider_matrix = cuda_config_module.build_jit_cache_provider_matrix(config)
    shim_matrix = cuda_config_module.build_jit_cache_shim_matrix(config)

    assert "jit_cache_wheel_format" not in config
    assert len(provider_matrix) == 42
    assert len(shim_matrix) == 6
    assert all(entry["provider_tag"].startswith("sm") for entry in provider_matrix)
    assert {
        (entry["cuda_label"], entry["cpu_architecture"])
        for entry in provider_matrix
        if entry["provider_tag"] == "sm121a"
    } == {("cu130", "aarch64"), ("cu134", "aarch64")}
    assert all(entry["provider_tag"] != "sm86" for entry in provider_matrix)


@pytest.mark.cpu_only
def test_provider_release_matrix_rejects_duplicate_architectures(
    cuda_config_module,
):
    config = json.loads((REPO_ROOT / "ci" / "cuda-versions.json").read_text())
    invalid_config = deepcopy(config)
    invalid_config["jit_cache"][0]["x86_64_provider_architectures"].append("8.0")

    with pytest.raises(
        cuda_config_module.ConfigError, match="contains duplicate architectures"
    ):
        cuda_config_module.validate_cuda_config(invalid_config, REPO_ROOT)


@pytest.mark.cpu_only
def test_jit_cache_python_setup_hardens_incomplete_downloads(tmp_path):
    common_script = REPO_ROOT / "scripts" / "jit_cache_build_common.sh"
    fake_python = tmp_path / "python"
    invocation_log = tmp_path / "python-invocations"
    fake_python.write_text(
        "#!/bin/bash\n"
        f'printf \'%s|%s|%s\\n\' "$*" "${{PIP_CONSTRAINT-unset}}" '
        f'"${{PIP_BUILD_CONSTRAINT-unset}}" >> "{invocation_log}"\n'
        'if [ "${1:-}" = "-c" ]; then\n'
        "  printf '%s\\n' 'torch==2.9.0+cu130'\n"
        "fi\n"
    )
    fake_python.chmod(0o755)
    script = f"""
set -euo pipefail
source "{common_script}"
export PIP_CONSTRAINT=/tmp/original-constraint
export PIP_BUILD_CONSTRAINT=/tmp/original-build-constraint
unset PIP_RETRIES PIP_RESUME_RETRIES PIP_DEFAULT_TIMEOUT
setup_jit_cache_python_build "{fake_python}" 13.0 cu130
test "${{PIP_RETRIES}}" = 10
test "${{PIP_RESUME_RETRIES}}" = 10
test "${{PIP_DEFAULT_TIMEOUT}}" = 120
cleanup_jit_cache_python_build
test "${{PIP_CONSTRAINT}}" = /tmp/original-constraint
test "${{PIP_BUILD_CONSTRAINT}}" = /tmp/original-build-constraint
"""

    subprocess.run(["bash", "-c", script], check=True)

    assert invocation_log.read_text().splitlines()[0] == (
        "-m pip install --upgrade pip>=26.2.1 build|unset|unset"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("cuda_version", ["13.0", "130", "13.4", "134", "14.0"])
def test_sccache_uses_release_for_all_cuda_versions(cuda_version):
    common_script = REPO_ROOT / "scripts" / "jit_cache_build_common.sh"
    script = f"""
set -euo pipefail
source "{common_script}"
install_released_sccache() {{ printf '%s %s' "$1" "$2"; }}
export CUDA_VERSION={cuda_version}
install_sccache "${{SCCACHE_VERSION}}" x86_64
printf ' %s %s' "${{FLASHINFER_SCCACHE_INSTALL_SOURCE}}" "${{FLASHINFER_SCCACHE_REVISION}}"
"""

    result = subprocess.run(
        ["bash", "-c", script], check=True, capture_output=True, text=True
    )

    assert result.stdout == "0.18.0 x86_64 github-release v0.18.0"


@pytest.mark.cpu_only
def test_pr_aot_matrix_waits_for_published_cuda_image_tag():
    workflow = (REPO_ROOT / ".github" / "workflows" / "pr-test.yml").read_text()

    assert 'grep -q "^flashinfer/flashinfer-ci-${cuda}:" ci/docker-tags.yml' in workflow
    assert "AOT image not published" in workflow


@pytest.mark.cpu_only
def test_release_verifier_rejects_duplicate_wheel_distributions(
    monkeypatch, tmp_path, release_verifier_module
):
    wheel_type = release_verifier_module.Wheel
    for filename in ("shim-one.whl", "shim-two.whl"):
        (tmp_path / filename).touch()

    def fake_open(path):
        return wheel_type(
            path=path,
            distribution="flashinfer-jit-cache",
            version="0.6.16+cu130",
            requirements=(),
            contents=(),
            metadata_path="flashinfer_jit_cache-0.6.16.dist-info/METADATA",
        )

    monkeypatch.setattr(release_verifier_module.Wheel, "open", staticmethod(fake_open))

    with pytest.raises(ValueError, match="Duplicate wheel distribution"):
        release_verifier_module.load_wheels(tmp_path)


@pytest.mark.cpu_only
def test_release_verifier_requires_exact_provider_set(release_verifier_module):
    shim = release_verifier_module.Wheel(
        path=Path(
            "flashinfer_jit_cache-0.6.16+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
        ),
        distribution="flashinfer-jit-cache",
        version="0.6.16+cu130",
        requirements=(
            "flashinfer-jit-cache-sm80==0.6.16+cu130",
            "flashinfer-jit-cache-sm90a==0.6.16+cu130",
        ),
        contents=("flashinfer_jit_cache-0.6.16.dist-info/METADATA",),
        metadata_path="flashinfer_jit_cache-0.6.16.dist-info/METADATA",
    )

    release_verifier_module.validate_shim(
        shim,
        "0.6.16+cu130",
        {"sm80", "sm90a"},
        "manylinux_2_28_x86_64",
    )
    with pytest.raises(ValueError, match="do not match"):
        release_verifier_module.validate_shim(
            shim,
            "0.6.16+cu130",
            {"sm80", "sm90a", "sm120f"},
            "manylinux_2_28_x86_64",
        )

    with pytest.raises(ValueError, match="does not use manylinux_2_28_aarch64"):
        release_verifier_module.validate_shim(
            shim,
            "0.6.16+cu130",
            {"sm80", "sm90a"},
            "manylinux_2_28_aarch64",
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("filename", "expected_package"),
    [
        (
            "flashinfer_jit_cache_sm90a-0.6.16+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
            "flashinfer-jit-cache-sm90a",
        ),
        (
            "flashinfer_jit_cache_sm121a-0.6.16.dev20260909+cu134-cp39-abi3-manylinux_2_28_aarch64.whl",
            "flashinfer-jit-cache-sm121a",
        ),
    ],
)
def test_wheel_index_recognizes_provider_distributions(
    wheel_index_module, filename, expected_package
):
    info = wheel_index_module.get_package_info(Path(filename))

    assert info == {
        "package": expected_package,
        "version": filename.split("-")[1],
        "cuda": "130" if "+cu130" in filename else "134",
    }


@pytest.mark.parametrize("nightly", [False, True])
@pytest.mark.parametrize("untested", [False, True])
@pytest.mark.parametrize(
    ("filename", "index_path"),
    [
        ("flashinfer_python-0.7.2-py3-none-any.whl", "flashinfer-python"),
        ("flashinfer_cubin-0.7.2-py3-none-any.whl", "flashinfer-cubin"),
        (
            "flashinfer_jit_cache-0.7.2+cu130-py3-none-any.whl",
            "cu130/flashinfer-jit-cache",
        ),
        (
            "flashinfer_jit_cache_sm120f-0.7.2+cu130-py3-none-any.whl",
            "cu130/flashinfer-jit-cache-sm120f",
        ),
    ],
)
def test_wheel_index_adds_upload_times_without_duplicate_links(
    tmp_path, wheel_index_module, nightly, untested, filename, index_path
):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / filename).write_bytes(b"wheel")
    assets = tmp_path / "release-assets.json"
    timestamp = "2026-10-01T12:34:56Z"
    assets.write_text(json.dumps([{"name": filename, "created_at": timestamp}]))
    output = tmp_path / "whl"
    index = output / ("nightly" if nightly else "") / index_path / "index.html"
    index.parent.mkdir(parents=True)
    digest = hashlib.sha256(b"wheel").hexdigest()
    url = f"https://example.com/v0.7.2/{filename}#sha256={digest}"
    existing_link = '<a href="https://example.com/old.whl">old.whl</a><br>\n'
    dated_link = (
        '<a href="https://example.com/dated.whl" data-upload-time="2026-09-01T00:00:00Z">'
        "dated.whl</a><br>\n"
    )
    index.write_text(
        existing_link + dated_link + f'<a href="{url}">{filename}</a><br>\n'
    )

    kwargs = dict(
        release_assets=str(assets),
        dist_dir=str(dist),
        output_dir=str(output),
        base_url="https://example.com",
        release_tag="v0.7.2",
        nightly=nightly,
        untested=untested,
    )
    wheel_index_module.update_index(**kwargs)

    content = index.read_text()
    assert existing_link in content
    assert dated_link in content
    marker = ' data-untested="true"' if untested else ""
    assert (
        f'<a href="{url}" data-upload-time="{timestamp}"{marker}>{filename}</a>'
        in content
    )
    assert content.count(f'href="{url}"') == 1

    wheel_index_module.update_index(**kwargs)
    assert index.read_text() == content


@pytest.mark.parametrize(
    "assets",
    [
        [],
        [{"name": "flashinfer_python-0.7.2-py3-none-any.whl", "created_at": "invalid"}],
        [
            {
                "name": "flashinfer_python-0.7.2-py3-none-any.whl",
                "created_at": "2026-10-01T12:34:56",
            }
        ],
    ],
)
def test_wheel_index_rejects_missing_or_invalid_upload_times(
    tmp_path, wheel_index_module, assets
):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "flashinfer_python-0.7.2-py3-none-any.whl").touch()
    metadata = tmp_path / "release-assets.json"
    metadata.write_text(json.dumps(assets))
    output = tmp_path / "whl"

    with pytest.raises(ValueError):
        wheel_index_module.update_index(
            release_assets=str(metadata), dist_dir=str(dist), output_dir=str(output)
        )

    assert not output.exists()
