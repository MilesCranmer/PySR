#!/bin/bash -e

# Generate home page using README.md:
echo '<div style="width:100%;height:0px;position:relative;padding-bottom:56.250%;"><iframe src="https://streamable.com/e/ncvqhy" frameborder="0" width="100%" height="100%" allowfullscreen style="width:100%;height:100%;position:absolute;left:0px;top:0px;overflow:hidden;"></iframe></div>' > index.md
cat ../README.md | grep -v 'user-images' | grep -E -v '\<.*div.*\>' >> index.md

# Transform "### Test status" to "**Test status**":
sed -i.bak 's/\#\#\# Test status/**Test status**/g' index.md
# Change '# ' to '## ':
sed -i.bak '10,$s/^\# /## /g' index.md

# Create papers.md
python generate_papers.py