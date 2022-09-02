#!/bin/bash
cat README.md | grep -v 'pysr_logo.svg' > docs/index.md
pydoc-markdown --build --site-dir build -vv
cp docs/build/content/docs/api*.md docs/ && rm -rf docs/build
for f in docs/api*.md; do mv "$f" "$f.bkup" && cat "$f.bkup" | sed '1,4d' > "$f" && rm "$f.bkup"; done
cd docs && python generate_papers.py && cd ..