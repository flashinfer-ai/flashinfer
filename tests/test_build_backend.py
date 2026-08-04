from unittest import mock

import build_backend


def test_build_hooks_do_not_invoke_package_installers():
    with (
        mock.patch.object(build_backend, "_build_nvep_if_enabled"),
        mock.patch.object(build_backend, "_create_data_dir"),
        mock.patch.object(build_backend, "_data_dir") as data_dir,
        mock.patch.object(build_backend.subprocess, "run") as run,
    ):
        data_dir.exists.return_value = False
        build_backend._prepare_for_editable()
    run.assert_not_called()

    with (
        mock.patch.object(build_backend, "_BUILD_NCCL_EP", True),
        mock.patch.object(build_backend, "_BUILD_NIXL_EP", False),
        mock.patch.object(build_backend.subprocess, "run") as run,
    ):
        build_backend._build_nvep_if_enabled()
    run.assert_not_called()
