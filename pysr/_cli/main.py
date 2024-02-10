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


@pysr.command("install", help="DEPRECATED (dependencies are now installed at import).")
@click.option(
    "-p",
    "julia_project",
    "--project",
    default=None,
    type=str,
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Disable logging.")
@click.option(
    "--precompile",
    "precompile",
    flag_value=True,
    default=None,
)
@click.option(
    "--no-precompile",
    "precompile",
    flag_value=False,
    default=None,
)
def _install(julia_project, quiet, precompile):
    warnings.warn(
        "This command is deprecated. Julia dependencies are now installed at first import."
    )


TEST_OPTIONS = {"main", "jax", "torch", "cli", "warm_start"}


@pysr.command("test")
@click.argument("tests", nargs=1)
def _tests(tests):
    """Run parts of the PySR test suite.

    Choose from main, jax, torch, cli, and warm_start. You can give multiple tests, separated by commas.
    """
    for test in tests.split(","):
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
            elif test == "warm-start":
                runtests_warm_start()
        else:
            warnings.warn(f"Invalid test {test}. Skipping.")
