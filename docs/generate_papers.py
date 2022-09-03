"""This script generates the papers.md file from the papers.yml file."""
import yaml
from pathlib import Path

data_file = "papers.yml"
papers_header = Path("stylesheets") / "papers_header.txt"
output_file = "papers.md"

# Load YAML file:
with open(data_file, "r") as stream:
    papers = yaml.load(stream, Loader=yaml.SafeLoader)["papers"]

# Load header:
with open(papers_header, "r") as stream:
    header = stream.read()

with open(output_file, "w") as f:
    f.write(header)

    # First, we sort the papers by date.
    # This is in the format of "2022-03-15"
    papers = sorted(papers, key=lambda paper: paper["date"], reverse=True)

    snippets = []
    for paper in papers:
        title = paper["title"]
        authors = (
            ", ".join(paper["authors"]).replace("(", "<sup>").replace(")", "</sup>")
        )
        affiliations = ", ".join(
            f"<sup>{num}</sup>{affil}" for num, affil in paper["affiliations"].items()
        )
        link = paper["link"]
        abstract = paper["abstract"]
        image_file = paper["image"]

        # Begin:
        paper_snippet = f"""

<figure markdown>
![](images/{image_file}){{ width="500"}}
<figcaption>
<!-- Large font: -->
<h2>
<a href="{link}">{title}</a>
</h2>
</figcaption>
</figure>

<center>
{authors}
    
<small>{affiliations}</small>
</center>

**Abstract:** {abstract}\n\n
"""
        snippets.append(paper_snippet)

    f.write("\n\n---\n\n".join(snippets))
