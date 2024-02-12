import sys
import unittest
import warnings

import click

from ..test import (
    get_runtests_cli,
    runtests,
    runtests_dev,
    runtests_jax,
    runtests_startup,
    runtests_torch,
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


TEST_OPTIONS = {"main", "jax", "torch", "cli", "dev", "startup"}


@pysr.command("test")
@click.argument("tests", nargs=1)
def _tests(tests):
    """Run parts of the PySR test suite.

    Choose from main, jax, torch, cli, dev, and startup. You can give multiple tests, separated by commas.
    """
    test_cases = []
    for test in tests.split(","):
        if test == "main":
            test_cases.extend(runtests(just_tests=True))
        elif test == "jax":
            test_cases.extend(runtests_jax(just_tests=True))
        elif test == "torch":
            test_cases.extend(runtests_torch(just_tests=True))
        elif test == "cli":
            runtests_cli = get_runtests_cli()
            test_cases.extend(runtests_cli(just_tests=True))
        elif test == "dev":
            test_cases.extend(runtests_dev(just_tests=True))
        elif test == "startup":
            test_cases.extend(runtests_startup(just_tests=True))
        else:
            warnings.warn(f"Invalid test {test}. Skipping.")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    runner = unittest.TextTestRunner()
    results = runner.run(suite)
    # Normally unittest would run this, but here we have
    # to do it manually to get the exit code.

    if not results.wasSuccessful():
        sys.exit(1)
