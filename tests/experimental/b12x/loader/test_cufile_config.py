import json

from b12x.loader import _gds_native

NVIDIA_STYLE = """{
    // NOTE : Application can override custom configuration via export CUFILE_ENV_PATH_JSON=<filepath>
    "logging": {
        // log directory, if not enabled will create log file under current working directory
        "dir": "/home/<xxxx>", /* inline block */
        "level": "ERROR"
    },
    "properties": {
        "url": "http://example//not-a-comment",
        "max_direct_io_size_kb": 16384
    }
}
"""


def test_cufile_config_accepts_comments(tmp_path, monkeypatch):
    source = tmp_path / "cufile.json"
    source.write_text(NVIDIA_STYLE)
    monkeypatch.setenv("CUFILE_ENV_PATH_JSON", str(source))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _gds_native._configure_cufile()
    written = json.loads(open(_gds_native.os.environ["CUFILE_ENV_PATH_JSON"]).read())
    assert written["properties"]["allow_compat_mode"] is True
    assert written["properties"]["url"] == "http://example//not-a-comment"
    assert written["logging"] == {"dir": "/home/<xxxx>", "level": "ERROR"}
