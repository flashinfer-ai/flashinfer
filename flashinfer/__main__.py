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
import os
import click
from tabulate import tabulate  # type: ignore[import-untyped]

from .artifacts import (
    ArtifactPath,
    download_artifacts,
    clear_cubin,
    get_artifacts_status,
)
from .jit import clear_cache_dir, jit_spec_registry
from .jit.cubin_loader import FLASHINFER_CUBINS_REPOSITORY
from .jit.env import FLASHINFER_CACHE_DIR, FLASHINFER_CUBIN_DIR
from .jit.core import current_compilation_context
from .jit.cpp_ext import get_cuda_path, get_cuda_version

# Import __version__ from centralized version module
from .version import __version__


def _download_cubin():
    """Helper function to download cubin"""
    try:
        download_artifacts()
        click.secho("✅ All cubin download tasks completed successfully.", fg="green")
    except Exception as e:
        click.secho(f"❌ Cubin download failed: {e}", fg="red")


def _ensure_modules_registered():
    """Helper function to ensure modules are registered"""
    statuses = jit_spec_registry.get_all_statuses()
    if not statuses:
        click.secho("No modules found. Registering default modules...", fg="yellow")
        try:
            from .aot import register_default_modules

            num_registered = register_default_modules()
            click.secho(f"✅ Registered {num_registered} modules", fg="green")
            statuses = jit_spec_registry.get_all_statuses()
        except Exception as e:
            click.secho(f"❌ Module registration failed: {e}", fg="red")
    return statuses


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
    "FLASHINFER_CUDA_ARCH_LIST": current_compilation_context.TARGET_CUDA_ARCHS,
    "FLASHINFER_CUDA_VERSION": get_cuda_version(),
    "FLASHINFER_CUBINS_REPOSITORY": FLASHINFER_CUBINS_REPOSITORY,
    "CUDA_VERSION": get_cuda_version(),
}
try:
    env_variables["CUDA_HOME"] = get_cuda_path()
    found_nvcc = os.path.isfile(os.path.join(env_variables["CUDA_HOME"], "bin", "nvcc"))
except Exception:
    env_variables["CUDA_HOME"] = ""
    found_nvcc = False


@cli.command("show-config")
def show_config_cmd():
    """Show configuration"""

    click.secho("=== Version Info ===", fg="yellow")
    click.secho("FlashInfer version:", fg="magenta", nl=False)
    click.secho(f" {__version__}", fg="cyan")

    # Check for additional packages
    try:
        import importlib.metadata

        try:
            cubin_version = importlib.metadata.version("flashinfer-cubin")
            click.secho("flashinfer-cubin version:", fg="magenta", nl=False)
            click.secho(f" {cubin_version}", fg="cyan")
        except importlib.metadata.PackageNotFoundError:
            click.secho("flashinfer-cubin:", fg="magenta", nl=False)
            click.secho(" Not installed", fg="red")

        try:
            jit_cache_version = importlib.metadata.version("flashinfer-jit-cache")
            click.secho("flashinfer-jit-cache version:", fg="magenta", nl=False)
            click.secho(f" {jit_cache_version}", fg="cyan")
        except importlib.metadata.PackageNotFoundError:
            click.secho("flashinfer-jit-cache:", fg="magenta", nl=False)
            click.secho(" Not installed", fg="red")
    except Exception as e:
        click.secho(f"Error checking package versions: {e}", fg="yellow")

    # Section: Torch Version Info
    import torch

    click.secho("=== Torch Version Info ===", fg="yellow")
    click.secho("Torch version:", fg="magenta", nl=False)
    click.secho(f" {torch.__version__}", fg="cyan")
    click.secho("CUDA runtime available:", fg="magenta", nl=False)
    if torch.cuda.is_available():
        click.secho(" Yes", fg="green")
    else:
        click.secho(" No", fg="red")
    click.secho("", fg="white")

    # Section: Environment Variables
    click.secho("=== Environment Variables ===", fg="yellow")
    for name, value in env_variables.items():
        click.secho(f"{name}:", fg="magenta", nl=False)
        click.secho(f" {value}", fg="cyan")
    click.secho("NVCC found:", fg="magenta", nl=False)
    if found_nvcc:
        click.secho(" Yes", fg="green")
    else:
        click.secho(" No", fg="red")
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
    num_downloaded = sum(1 for _, exists in status if exists)
    total_cubins = len(status)

    click.secho(f"Downloaded {num_downloaded}/{total_cubins} cubins", fg="cyan")
    click.secho("", fg="white")

    # Section: Module Status
    click.secho("=== Module Status ===", fg="yellow")

    module_statuses = _ensure_modules_registered()
    if module_statuses:
        stats = jit_spec_registry.get_stats()
        click.secho(f"Total modules: {stats['total']}", fg="cyan")
        click.secho(f"compiled: {stats['compiled']}", fg="magenta")
        click.secho(f"Not compiled: {stats['not_compiled']}", fg="red")


@cli.command("list-cubins")
def list_cubins_cmd():
    """List downloaded cubins"""
    status = get_artifacts_status()
    table_data = []

    for file_name, exists in status:
        status_str = "Downloaded" if exists else "Missing"
        color = "green" if exists else "red"
        table_data.append([file_name, click.style(status_str, fg=color)])

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


@cli.command("module-status")
@click.option("--detailed", is_flag=True, help="Show detailed information")
@click.option(
    "--filter",
    type=click.Choice(["all", "aot", "jit", "compiled", "not-compiled"]),
    default="all",
    help="Filter modules by compilation type or status",
)
def module_status_cmd(detailed, filter):
    """Show module compilation status"""
    statuses = _ensure_modules_registered()
    if not statuses:
        return

    # Apply filter
    filter_map = {
        "compiled": lambda s: s.is_compiled,
        "not-compiled": lambda s: not s.is_compiled,
    }
    if filter in filter_map:
        statuses = [s for s in statuses if filter_map[filter](s)]

    # Sort by name for consistent output
    statuses.sort(key=lambda x: x.name)

    if detailed:
        # Detailed view
        for status in statuses:
            click.secho(f"Module: {status.name}", fg="cyan", bold=True)
            click.secho(
                f"  Status: {click.style(status.status, fg='green' if status.is_compiled else 'red')}"
            )
            if status.library_path:
                click.secho(f"  Library: {status.library_path}", fg="white")
            click.secho(f"  Sources: {len(status.sources)} file(s)", fg="white")
            if status.needs_device_linking:
                click.secho("  Device Linking: Required", fg="yellow")
            click.secho(
                f"  Created: {status.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                fg="white",
            )
            click.echo()
    else:
        # Table view
        table_data = []
        for status in statuses:
            status_color = "green" if status.is_compiled else "red"
            table_data.append(
                [
                    status.name,
                    click.style(status.status, fg=status_color),
                    len(status.sources),
                    "Yes" if status.needs_device_linking else "No",
                ]
            )

        headers = ["Module Name", "Type", "Status", "Sources", "Device Linking"]
        click.echo(tabulate(table_data, headers=headers, tablefmt="github"))

    # Show summary statistics
    stats = jit_spec_registry.get_stats()
    click.echo()
    click.secho("=== Summary ===", fg="yellow")
    click.secho(f"Total modules: {stats['total']}", fg="cyan")
    click.secho(f"Compiled: {stats['compiled']}", fg="magenta")
    click.secho(f"Not compiled: {stats['not_compiled']}", fg="red")


@cli.command("list-modules")
@click.argument("module_name", required=False)
def list_modules_cmd(module_name):
    """List or inspect compilation modules"""
    # Register default modules if none exist
    statuses = _ensure_modules_registered()
    if not statuses:
        return

    if module_name:
        # Show specific module
        status = jit_spec_registry.get_spec_status(module_name)
        if not status:
            click.secho(f"Module '{module_name}' not found.", fg="red")
            return

        click.secho(f"Module: {status.name}", fg="cyan", bold=True)
        click.secho(
            f"Status: {click.style(status.status, fg='green' if status.is_compiled else 'red')}"
        )
        if status.library_path:
            click.secho(f"Library Path: {status.library_path}", fg="white")
        click.secho(
            f"Created: {status.created_at.strftime('%Y-%m-%d %H:%M:%S')}", fg="white"
        )
        click.secho(
            f"Device Linking: {'Required' if status.needs_device_linking else 'Not required'}",
            fg="white",
        )
        click.secho("Source Files:", fg="white")
        for i, source in enumerate(status.sources, 1):
            click.secho(f"  {i}. {source}", fg="white")
    else:
        # List all modules
        statuses = _ensure_modules_registered()
        if not statuses:
            return

        statuses.sort(key=lambda x: x.name)
        click.secho("Available compilation modules:", fg="cyan", bold=True)
        for status in statuses:
            status_color = "green" if status.is_compiled else "red"
            click.secho(
                f"  {status.name} - {click.style(status.status, fg=status_color)}"
            )


@cli.command("export-compile-commands")
@click.argument("path", required=False, default="compile_commands.json")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (overrides PATH argument)",
)
def export_compile_commands_cmd(path, output):
    """Export compile commands to compile_commands.json

    PATH: Output file path (default: compile_commands.json)
    """
    import json

    # --output option overrides PATH argument
    output_path = output if output is not None else path

    # Register default modules if none exist
    _ensure_modules_registered()

    # Get all registered specs
    all_specs = jit_spec_registry.get_all_specs()

    if not all_specs:
        click.secho("No modules found to export.", fg="yellow")
        return

    # Collect all compile commands
    all_compile_commands = []
    for spec in all_specs.values():
        try:
            compile_commands = spec.get_compile_commands()
            all_compile_commands.extend(compile_commands)
        except Exception as e:
            click.secho(
                f"Warning: Failed to generate compile commands for {spec.name}: {e}",
                fg="yellow",
            )

    # Write to output file
    try:
        with open(output_path, "w") as f:
            json.dump(all_compile_commands, f, indent=2)
        click.secho(
            f"✅ Successfully exported {len(all_compile_commands)} compile commands to {output_path}",
            fg="green",
        )
    except Exception as e:
        click.secho(f"❌ Failed to write compile commands: {e}", fg="red")


@cli.command("replay")
@click.option(
    "--dir",
    "dump_dir",
    required=True,
    help="Directory containing dump files (or root directory of session)",
)
def replay_cmd(dump_dir):
    """Replay API calls from dump directory"""
    from .api_logging import replay_sequence, replay_from_dump

    device = "cuda"

    if not os.path.exists(dump_dir):
        click.secho(f"❌ Directory not found: {dump_dir}", fg="red")
        return

    # Check if this is a single dump or a session / sequence root
    is_single_dump = os.path.exists(os.path.join(dump_dir, "metadata.jsonl"))

    try:
        if is_single_dump:
            click.secho(f"Replaying single dump from {dump_dir}...", fg="cyan")
            result = replay_from_dump(
                dump_dir, compare_outputs=True, device=device, run=True
            )
            if result.get("comparison_match"):
                click.secho("✅ Replay passed (outputs matched)", fg="green")
            elif result.get("execution_error"):
                click.secho(
                    f"❌ Execution failed: {result['execution_error']}", fg="red"
                )
            else:
                click.secho("⚠️  Replay finished but outputs did not match", fg="yellow")
        else:
            # Session / sequence replay
            click.secho(f"Replaying session from {dump_dir}...", fg="cyan")
            results = replay_sequence(dump_dir, device=device)

            passed = 0
            failed = 0

            for i, res in enumerate(results):
                dump_name = (
                    os.path.basename(res.get("dump_dir", ""))
                    if "dump_dir" in res
                    else f"call_{i + 1}"
                )
                # If replay_from_dump returned successfully, metadata might have the name
                if "metadata" in res and "function_name" in res["metadata"]:
                    func_name = res["metadata"]["function_name"]
                    dump_name = f"{func_name} ({dump_name})"

                if "error" in res:
                    click.secho(
                        f"[{i + 1}] {dump_name}: ❌ Error: {res['error']}", fg="red"
                    )
                    failed += 1
                elif res.get("comparison_match"):
                    click.secho(f"[{i + 1}] {dump_name}: ✅ Passed", fg="green")
                    passed += 1
                else:
                    click.secho(f"[{i + 1}] {dump_name}: ⚠️  Mismatch", fg="yellow")
                    failed += 1

            click.secho(
                f"\nSummary: {passed} passed, {failed} failed/mismatch", fg="white"
            )

    except Exception as e:
        click.secho(f"❌ Replay failed: {e}", fg="red")


if __name__ == "__main__":
    cli()
