"""CLI for running PySR's test suite."""
import argparse
import os

from . import *

if __name__ == "__main__":
    # Get args:
    parser = argparse.ArgumentParser()
    parser.usage = "python -m pysr.test [tests...]"
    parser.add_argument(
        "test",
        nargs="*",
        help="Test to run. One or more of 'main', 'env', 'jax', 'torch'.",
    )

    # Parse args:
    args = parser.parse_args()
    tests = args.test

    if len(tests) == 0:
        # Raise help message:
        parser.print_help()
        raise SystemExit(1)

    # Run tests:
    for test in tests:
        if test in {"main", "env", "jax", "torch"}:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Running test from {cur_dir}")
            if test == "main":
                runtests()
            elif test == "env":
                runtests_env()
            elif test == "jax":
                runtests_jax()
            elif test == "torch":
                runtests_torch()
        else:
            parser.print_help()
            raise SystemExit(1)
