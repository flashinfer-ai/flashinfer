"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# flashinfer-cli
import click
from tabulate import tabulate  # type: ignore[import-untyped]

from .artifacts import (
    ArtifactPath,
    download_artifacts,
    clear_cubin,
    get_artifacts_status,
)
from .jit import clear_cache_dir
from .jit.cubin_loader import FLASHINFER_CUBINS_REPOSITORY
from .jit.env import FLASHINFER_CACHE_DIR, FLASHINFER_CUBIN_DIR
from .jit.core import current_compilation_context
from .jit.cpp_ext import get_cuda_path, get_cuda_version


def _download_cubin():
    """Helper function to download cubin"""
    try:
        download_artifacts()
        click.secho("✅ All cubin download tasks completed successfully.", fg="green")
    except Exception as e:
        click.secho(f"❌ Cubin download failed: {e}", fg="red")


@click.group(invoke_without_command=True)
@click.option(
    "--download-cubin", "download_cubin_flag", is_flag=True, help="Download artifacts"
)
@click.pass_context
def cli(ctx, download_cubin_flag):
    """FlashInfer CLI"""
    if download_cubin_flag:
        _download_cubin()
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# list of environment variables
env_variables = {
    "FLASHINFER_CACHE_DIR": FLASHINFER_CACHE_DIR,
    "FLASHINFER_CUBIN_DIR": FLASHINFER_CUBIN_DIR,
    "CUDA_HOME": get_cuda_path(),
    "CUDA_VERSION": get_cuda_version(),
    "FLASHINFER_CUDA_ARCH_LIST": current_compilation_context.TARGET_CUDA_ARCHS,
    "FLASHINFER_CUBINS_REPOSITORY": FLASHINFER_CUBINS_REPOSITORY,
}


@cli.command("show-config")
def show_config_cmd():
    """Show configuration"""
    import torch

    # Section: Torch Version Info
    click.secho("=== Torch Version Info ===", fg="yellow")
    click.secho("Torch version:", fg="magenta", nl=False)
    click.secho(f" {torch.__version__}", fg="cyan")
    click.secho("", fg="white")

    # Section: Environment Variables
    click.secho("=== Environment Variables ===", fg="yellow")
    for name, value in env_variables.items():
        click.secho(f"{name}:", fg="magenta", nl=False)
        click.secho(f" {value}", fg="cyan")
    click.secho("", fg="white")

    # Section: Artifact path
    click.secho("=== Artifact Path ===", fg="yellow")
    # list all artifact paths
    for name, path in ArtifactPath.__dict__.items():
        if not name.startswith("__"):
            click.secho(f"{name}:", fg="magenta", nl=False)
            click.secho(f" {path}", fg="cyan")
    click.secho("", fg="white")

    # Section: Downloaded Cubins
    click.secho("=== Downloaded Cubins ===", fg="yellow")

    status = get_artifacts_status()
    num_downloaded = sum(1 for _, _, exists in status if exists)
    total_cubins = len(status)

    click.secho(f"Downloaded {num_downloaded}/{total_cubins} cubins", fg="cyan")


@cli.command("list-cubins")
def list_cubins_cmd():
    """List downloaded cubins"""
    status = get_artifacts_status()
    table_data = []
    for name, extension, exists in status:
        status_str = "Downloaded" if exists else "Missing"
        color = "green" if exists else "red"
        table_data.append([f"{name}{extension}", click.style(status_str, fg=color)])

    click.echo(tabulate(table_data, headers=["Cubin", "Status"], tablefmt="github"))
    click.secho("", fg="white")


@cli.command("download-cubin")
def download_cubin_cmd():
    """Download artifacts"""
    _download_cubin()


@cli.command("clear-cache")
def clear_cache_cmd():
    """Clear cache"""
    try:
        clear_cache_dir()
        click.secho("✅ Cache cleared successfully.", fg="green")
    except Exception as e:
        click.secho(f"❌ Cache clear failed: {e}", fg="red")


@cli.command("clear-cubin")
def clear_cubin_cmd():
    """Clear cubin"""
    try:
        clear_cubin()
        click.secho("✅ Cubin cleared successfully.", fg="green")
    except Exception as e:
        click.secho(f"❌ Cubin clear failed: {e}", fg="red")


if __name__ == "__main__":
    cli()
