#!/bin/bash -e

# Generate home page using README.md:
cat ../README.md | grep -v 'pysr_logo.svg' | grep -E -v '\<.*div.*\>' > index.md

# Transform "### Test status" to "**Test status**":
sed -i.bak 's/\#\#\# Test status/**Test status**/g' index.md
# Change '# ' to '## ':
sed -i.bak '10,$s/^\# /## /g' index.md

# Create papers.md
python generate_papers.py