import warnings

import click

from ..test import (
    get_runtests_cli,
    runtests,
    runtests_jax,
    runtests_torch,
    runtests_warm_start,
)


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
    warnings.warn(
        "This command is deprecated. Julia dependencies are now installed at first import."
    )


TEST_OPTIONS = {"main", "jax", "torch", "cli", "warm_start"}


@pysr.command("test", help="Run PySR test suite.")
@click.argument("tests", nargs=-1)
def _tests(tests):
    """Run part of the PySR test suite.

    Choose from main, jax, torch, cli, and warm_start.
    """
    if len(tests) == 0:
        raise click.UsageError(
            "At least one test must be specified. "
            + "The following are available: "
            + ", ".join(TEST_OPTIONS)
            + "."
        )
    else:
        for test in tests:
            if test in TEST_OPTIONS:
                if test == "main":
                    runtests()
                elif test == "jax":
                    runtests_jax()
                elif test == "torch":
                    runtests_torch()
                elif test == "cli":
                    runtests_cli = get_runtests_cli()
                    runtests_cli()
                elif test == "warm_start":
                    runtests_warm_start()
            else:
                warnings.warn(f"Invalid test {test}. Skipping.")
