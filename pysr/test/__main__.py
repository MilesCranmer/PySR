"""CLI for running PySR's test suite."""
import argparse
import os

from . import *

if __name__ == "__main__":
    # Get args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsplits",
        type=int,
        default=5,
        help="Number of splits to split the test suite into, if passing 'main-*'.",
    )
    parser.add_argument(
        "test",
        nargs="*",
        help="Test to run. One or more of 'main', 'main-*', 'env', 'jax', 'torch'.",
    )
    parser.usage = "python -m pysr.test [--nsplits NSPLITS] [TESTS ...]"

    # Parse args:
    args = parser.parse_args()
    tests = args.test
    nsplits = args.nsplits

    if len(tests) == 0:
        # Raise help message:
        parser.print_help()
        raise SystemExit(1)

    # Run tests:
    for test in tests:
        if test in {"env", "jax", "torch"} or test.startswith("main"):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Running test from {cur_dir}")
            if test == "main":
                runtests()
            elif test.startswith("main-"):
                idx = int(test.split("-")[1])
                runtests(idx=idx, nsplits=nsplits)
            elif test == "env":
                runtests_env()
            elif test == "jax":
                runtests_jax()
            elif test == "torch":
                runtests_torch()
        else:
            parser.print_help()
            raise SystemExit(1)
