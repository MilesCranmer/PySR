import click

from ..julia_helpers import install


@click.group("pysr")
@click.pass_context
def pysr(context):
    ctx = context


@pysr.command("install", help="Install Julia dependencies for PySR.")
@click.option(
    "-p",
    "julia_project",
    "--project",
    default=None,
    type=str,
    help="Install in a specific Julia project (e.g., a local copy of SymbolicRegression.jl).",
    metavar="PROJECT_DIRECTORY",
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Disable logging.")
@click.option(
    "--precompile",
    "precompile",
    flag_value=True,
    default=None,
    help="Force precompilation of Julia libraries.",
)
@click.option(
    "--no-precompile",
    "precompile",
    flag_value=False,
    default=None,
    help="Disable precompilation.",
)
def _install(julia_project, quiet, precompile):
    install(julia_project, quiet, precompile)
