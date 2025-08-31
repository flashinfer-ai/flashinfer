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

from .artifacts import download_artifacts, clear_cubin
from .jit import clear_cache_dir


@click.group()
def cli():
    """FlashInfer CLI"""
    pass


@cli.command("download-cubin")
def download_cubin_cmd():
    """Download artifacts"""
    try:
        download_artifacts()
        click.secho("✅ All cubin download tasks completed successfully.", fg="green")
    except Exception as e:
        click.secho(f"❌ Cubin download failed: {e}", fg="red")


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
