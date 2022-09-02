#!/bin/bash -e

cat README.md | grep -v 'pysr_logo.svg' | grep -E -v '\<.*div.*\>' > docs/index.md
cd docs && python generate_papers.py && cd ..