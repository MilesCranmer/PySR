#!/bin/bash -e

cat README.md | grep -v 'pysr_logo.svg' | grep -E -v '\<.*div.*\>' > docs/index.md
# Transform "## Test status" to "**Test status**":
cd docs
sed -i.bak 's/\#\#\# Test status/**Test status**/g' index.md
# Change '# ' to '## ':
sed -i.bak '10,$s/^\# /## /g' index.md
python generate_papers.py