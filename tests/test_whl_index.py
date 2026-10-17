"""Tests for the wheel index generator used by the release workflows.

`scripts/update_whl_index.py` is shared by `release.yml` (stable) and
`nightly-release.yml` (nightly). The nightly workflow publishes artifacts even
when the test suite fails and passes `--untested` in that case, so these tests
pin both the untested marker and the unchanged default output the stable
release path depends on.
"""

import importlib.util
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "update_whl_index.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("update_whl_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def whl_index():
    return _load_module()


WHEEL = "flashinfer_python-0.6.18.dev20260915-cp39-abi3-manylinux_2_28_x86_64.whl"
JIT_WHEEL = (
    "flashinfer_jit_cache-0.6.18.dev20260915+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
)


def _make_dist(tmp_path, names=(WHEEL,)):
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in names:
        (dist / name).write_bytes(b"wheel-content-" + name.encode())
    return dist


def _index_for(out_dir, package="flashinfer-python", cuda=None):
    base = pathlib.Path(out_dir) / "nightly"
    if cuda:
        base = base / f"cu{cuda}"
    return base / package / "index.html"


def _run(whl_index, dist, out_dir, untested=False):
    whl_index.update_index(
        dist_dir=str(dist),
        output_dir=str(out_dir),
        base_url="https://github.com/flashinfer-ai/whl/releases/download",
        release_tag="nightly",
        nightly=True,
        untested=untested,
    )


def test_default_row_has_no_marker(whl_index, tmp_path):
    """The stable release path must keep producing plain anchor rows."""
    dist = _make_dist(tmp_path)
    out = tmp_path / "whl"
    _run(whl_index, dist, out)

    content = _index_for(out).read_text()
    assert f">{WHEEL}</a><br>" in content
    assert "data-untested" not in content
    assert "(untested)" not in content


def test_untested_row_is_marked(whl_index, tmp_path):
    dist = _make_dist(tmp_path)
    out = tmp_path / "whl"
    _run(whl_index, dist, out, untested=True)

    content = _index_for(out).read_text()
    assert 'data-untested="true"' in content
    assert "</a> (untested)<br>" in content
    # The href itself must be untouched so installers resolve it normally.
    assert re.search(
        rf'<a href="[^"]*/{re.escape(WHEEL)}#sha256=[0-9a-f]{{64}}"', content
    )


def test_untested_entry_is_still_a_single_parseable_link(whl_index, tmp_path):
    """An untested wheel stays installable: one anchor, unchanged href."""
    dist = _make_dist(tmp_path)
    out = tmp_path / "whl"
    _run(whl_index, dist, out, untested=True)

    content = _index_for(out).read_text()
    hrefs = re.findall(r'<a href="([^"]+)"', content)
    assert len(hrefs) == 1
    assert WHEEL in hrefs[0]
    assert "#sha256=" in hrefs[0]


def test_rerun_replaces_row_instead_of_duplicating(whl_index, tmp_path):
    """Re-publishing the same wheel as tested must not leave two rows."""
    dist = _make_dist(tmp_path)
    out = tmp_path / "whl"

    _run(whl_index, dist, out, untested=True)
    _run(whl_index, dist, out, untested=False)

    content = _index_for(out).read_text()
    # One anchor, not two: the untested row was rewritten, not appended to.
    assert content.count("<a href=") == 1
    assert "data-untested" not in content
    assert "(untested)" not in content


def test_preexisting_plain_rows_are_preserved(whl_index, tmp_path):
    """Rows written before the marker existed must survive a new run."""
    dist = _make_dist(tmp_path)
    out = tmp_path / "whl"
    index_file = _index_for(out)
    index_file.parent.mkdir(parents=True)
    old = "flashinfer_python-0.6.17.dev20260914-cp39-abi3-manylinux_2_28_x86_64.whl"
    index_file.write_text(
        "<!DOCTYPE html>\n<html>\n"
        "<head><title>Links for flashinfer-python</title></head>\n"
        "<body>\n<h1>Links for flashinfer-python</h1>\n"
        f'<a href="https://example/{old}#sha256=deadbeef">{old}</a><br>\n'
        "</body>\n</html>\n"
    )

    _run(whl_index, dist, out, untested=True)

    content = index_file.read_text()
    assert old in content
    assert WHEEL in content
    # Only the newly added wheel carries the marker.
    assert content.count('data-untested="true"') == 1


def test_jit_cache_wheel_lands_in_cuda_subdir(whl_index, tmp_path):
    dist = _make_dist(tmp_path, names=(JIT_WHEEL,))
    out = tmp_path / "whl"
    _run(whl_index, dist, out, untested=True)

    content = _index_for(out, package="flashinfer-jit-cache", cuda="130").read_text()
    assert JIT_WHEEL in content
    assert 'data-untested="true"' in content
