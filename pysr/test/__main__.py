"""CLI for running PySR's test suite."""
import argparse

from . import *

if __name__ == "__main__":
    # Get args:
    parser = argparse.ArgumentParser()
    parser.usage = "python -m pysr.test [tests...]"
    parser.add_argument(
        "test",
        nargs="*",
        help="DEPRECATED. Use `python -m pysr test [tests...]` instead."
        # help="Test to run. One or more of 'main', 'env', 'jax', 'torch', 'cli'.",
    )
