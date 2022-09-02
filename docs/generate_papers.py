"""Read papers.yml and generate papers.md

# Here is an example papers.yml:

papers:
  - title: Machine Learning the Gravity Equation for International Trade
    authors:
      - Sergiy Verstyuk (1)
      - Michael R. Douglas (1)
    affiliations:
      1: Harvard University
    link: https://papers.ssrn.com/abstract=4053795
    abstract: Machine learning (ML) is becoming more and more important throughout the mathematical and theoretical sciences. In this work we apply modern ML methods to gravity models of pairwise interactions in international economics. We explain the formulation of graphical neural networks (GNNs), models for graph-structured data that respect the properties of exchangeability and locality. GNNs are a natural and theoretically appealing class of models for international trade, which we demonstrate empirically by fitting them to a large panel of annual-frequency country-level data. We then use a symbolic regression algorithm to turn our fits into interpretable models with performance comparable to state of the art hand-crafted models motivated by economic theory. The resulting symbolic models contain objects resembling market access functions, which were developed in modern structural literature, but in our analysis arise ab initio without being explicitly postulated. Along the way, we also produce several model-consistent and model-agnostic ML-based measures of bilateral trade accessibility.
    image: economic_theory_gravity.png
    date: 2022-03-15

# Corresponding example papers.md:

<div class="row"><div class="image_column">

![](images/economic_theory_gravity.png)

</div><div class="text_column"><div class="center">

[Machine Learning the Gravity Equation for International Trade](https://papers.ssrn.com/abstract=4053795)<br>
Sergiy Verstyuk<sup>1</sup>, Michael R. Douglas.<sup>1</sup><br><sup>1</sup>Harvard University

</div>

**Abstract:** Machine learning (ML) is becoming more and more important throughout the mathematical and theoretical sciences. In this work we apply modern ML methods to gravity models of pairwise interactions in international economics. We explain the formulation of graphical neural networks (GNNs), models for graph-structured data that respect the properties of exchangeability and locality. GNNs are a natural and theoretically appealing class of models for international trade, which we demonstrate empirically by fitting them to a large panel of annual-frequency country-level data. We then use a symbolic regression algorithm to turn our fits into interpretable models with performance comparable to state of the art hand-crafted models motivated by economic theory. The resulting symbolic models contain objects resembling market access functions, which were developed in modern structural literature, but in our analysis arise ab initio without being explicitly postulated. Along the way, we also produce several model-consistent and model-agnostic ML-based measures of bilateral trade accessibility.

</div></div>
"""

import yaml
from textwrap import dedent

data_file = "papers.yml"
papers_header = "papers_header.md"
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
        <div class="row">


        <!-- Text column: -->
        <div class="text_column"><div class="center">
        <a href="{link}">{title}</a><br>{authors}<br><small>{affiliations}</small><br>\n\n
        **Abstract:** {abstract}\n\n
        </div></div>
        

        <!-- Image column: -->
        <div class="image_column">\n
        [![](images/{image_file})]({link})\n\n
        </div>


        </div>"""
        clean_paper_snippet = dedent(paper_snippet)
        f.write(clean_paper_snippet)
