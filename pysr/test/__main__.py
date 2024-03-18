"""CLI for running PySR's test suite."""

import argparse

from . import *

if __name__ == "__main__":
    # Get args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test",
        nargs="*",
        help="DEPRECATED. Use `python -m pysr test [tests...]` instead.",
    )
