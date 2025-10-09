#!/usr/bin/env python3
"""
Generate VitePress-compatible index.md by combining _index.md template with README.md content.
"""

from pathlib import Path


def process_readme_content(readme_content):
    """Process README.md content for VitePress."""
    lines = readme_content.split("\n")
    content_lines = []
    skip_until = False
    in_div = False

    for line in lines:
        # Skip the header logo and title section
        if line.startswith("[//]: # (Logo:)"):
            skip_until = True
            continue

        # Start capturing after the main title
        if skip_until and line.startswith("# PySR:"):
            skip_until = False
            # Don't include the title line itself as we have it in hero
            continue

        if skip_until:
            continue

        # Skip div tags but keep content inside
        if "<div" in line:
            in_div = True
            continue
        if "</div>" in line:
            in_div = False
            continue

        # Skip user-images URLs (GitHub-specific)
        if "user-images" in line:
            continue

        # Keep the video reference
        if "https://github.com/MilesCranmer/PySR/assets" in line and "c8511a49" in line:
            content_lines.append("")
            content_lines.append('<div align="center">')
            content_lines.append('<video width="800" height="600" controls>')
            content_lines.append(
                '<source src="https://github.com/MilesCranmer/PySR/assets/7593028/c8511a49-b408-488f-8f18-b1749078268f" type="video/mp4">'
            )
            content_lines.append("</video>")
            content_lines.append("</div>")
            content_lines.append("")
            continue

        # Transform headers - change # to ## for main sections
        if line.startswith("# ") and not line.startswith("# PySR"):
            line = "#" + line

        # Transform "### Test status" to "**Test status**"
        if line == "### Test status":
            line = "**Test status**"

        content_lines.append(line)

    # Find where the main content starts (after the badges/tables)
    main_content_start = 0
    for i, line in enumerate(content_lines):
        if line.startswith("## Why PySR?"):
            main_content_start = i
            break

    # Build the final content
    final_lines = []

    # Add opening text
    final_lines.extend(
        [
            "",
            "PySR searches for symbolic expressions which optimize a particular objective.",
            "",
        ]
    )

    # Add video if not already added
    final_lines.extend(
        [
            '<div align="center">',
            '<video width="800" height="600" controls>',
            '<source src="https://github.com/MilesCranmer/PySR/assets/7593028/c8511a49-b408-488f-8f18-b1749078268f" type="video/mp4">',
            "</video>",
            "</div>",
            "",
        ]
    )

    # Add simplified badges section
    badge_section = """<table>
<thead>
<tr>
<th align="center">Docs</th>
<th align="center">Forums</th>
<th align="center">Paper</th>
<th align="center">Colab Demo</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><a href="https://ai.damtp.cam.ac.uk/pysr/"><img src="https://github.com/MilesCranmer/PySR/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a></td>
<td align="center"><a href="https://github.com/MilesCranmer/PySR/discussions"><img src="https://img.shields.io/badge/discussions-github-informational" alt="Discussions"></a></td>
<td align="center"><a href="https://arxiv.org/abs/2305.01582"><img src="https://img.shields.io/badge/arXiv-2305.01582-b31b1b" alt="Paper"></a></td>
<td align="center"><a href="https://colab.research.google.com/github/MilesCranmer/PySR/blob/master/examples/pysr_demo.ipynb"><img src="https://img.shields.io/badge/colab-notebook-yellow" alt="Colab"></a></td>
</tr>
<tr>
<td align="center"><strong>Linux</strong></td>
<td align="center"><strong>Windows</strong></td>
<td align="center"><strong>macOS</strong></td>
<td align="center"><strong>Coverage</strong></td>
</tr>
<tr>
<td align="center"><a href="https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml"><img src="https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml/badge.svg" alt="Linux"></a></td>
<td align="center"><a href="https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml"><img src="https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml/badge.svg" alt="Windows"></a></td>
<td align="center"><a href="https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml"><img src="https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml/badge.svg" alt="macOS"></a></td>
<td align="center"><a href="https://codecov.io/gh/MilesCranmer/PySR"><img src="https://codecov.io/gh/MilesCranmer/PySR/branch/master/graph/badge.svg" alt="codecov"></a></td>
</tr>
</tbody>
</table>

If you find PySR useful, please cite the paper [arXiv:2305.01582](https://arxiv.org/abs/2305.01582).
If you've finished a project with PySR, please submit a PR to showcase your work on the [research showcase page](/papers)!
"""
    final_lines.append(badge_section)

    # Add the rest of the content from "Why PySR?" onwards
    for i in range(main_content_start, len(content_lines)):
        line = content_lines[i]
        # Fix internal links
        line = line.replace("(https://ai.damtp.cam.ac.uk/pysr/papers)", "(/papers)")
        # Remove Contents section as VitePress has its own navigation
        if line.startswith("**Contents**:") or line.startswith("- ["):
            if (
                "Why PySR?" in line
                or "Installation" in line
                or "Quickstart" in line
                or "Documentation" in line
                or "Contributors" in line
            ):
                continue
        final_lines.append(line)

    return "\n".join(final_lines)


def main():
    # Get paths
    script_dir = Path(__file__).parent
    template_path = script_dir / "src" / "_index.md"
    readme_path = script_dir.parent.parent.parent / "README.md"
    output_path = script_dir / "src" / "index.md"

    # Read template
    if not template_path.exists():
        print(f"Error: Template file {template_path} not found!")
        return 1

    template_content = template_path.read_text()

    # Read README
    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}!")
        return 1

    readme_content = readme_path.read_text()

    # Process README content
    processed_content = process_readme_content(readme_content)

    # Replace marker in template
    if "<!-- README_CONTENT_MARKER -->" in template_content:
        final_content = template_content.replace(
            "<!-- README_CONTENT_MARKER -->", processed_content
        )
    else:
        # If no marker, append content
        final_content = template_content + "\n" + processed_content

    # Write output
    output_path.write_text(final_content)
    print(f"Generated {output_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
