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

from .artifacts import (
    download_artifacts,
    clear_cubin,
    download_artifacts_status,
    get_cubin_file_list,
)
from .jit import clear_cache_dir


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


@cli.command("show-config")
def show_config_cmd():
    """Show configuration"""
    import torch

    # Section: Torch Version Info
    click.secho("=== Torch Version Info ===", fg="yellow")
    click.secho(f"Torch version: {torch.__version__}", fg="cyan")
    click.secho("", fg="white")

    # Section: Environment Variables
    click.secho("=== Environment Variables ===", fg="yellow")

    # Section: Downloaded Cubins
    click.secho("=== Downloaded Cubins ===", fg="yellow")
    num_downloaded, total = download_artifacts_status()
    click.secho(f"Downloaded {num_downloaded} out of {total} cubins", fg="green")
    click.secho("", fg="white")

    # Section: Compiled Kernels
    click.secho("=== Compiled Kernels ===", fg="yellow")
    # TODO: List compiled kernels


@cli.command("list-cubins")
def list_cubins_cmd():
    """List downloaded cubins"""
    cubin_files = get_cubin_file_list()
    for name, extension in cubin_files:
        click.secho(f"{name}{extension}", fg="cyan")


@cli.command("list-compiled-kernels")
def list_compiled_kernels_cmd():
    """List compiled kernels"""
    # TODO(Zihao)


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
