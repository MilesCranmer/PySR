#!/usr/bin/env python3
"""Validate docs/papers.yml.

Goal: a fast, deterministic PR check that catches the common formatting issues
maintainers would otherwise check manually.

This is intentionally "simple yet robust": it validates structure and a few
key invariants, without trying to enforce stylistic preferences.

Rules:
- YAML must parse.
- Top-level must be a mapping with key `papers` as a list.
- Each paper must be a mapping with required keys:
    - title: non-empty string
    - authors: non-empty list of strings
    - link: non-empty string
    - date: string in YYYY-MM-DD format (basic ISO date)
    - image: string that is either:
        - absolute http(s) URL, OR
        - a basename filename that exists under docs/src/public/images/

We also reject a common footgun:
- image URLs pointing at the *temporary* bot branch `paper-images/pr-...`.
"""

from __future__ import annotations

import os
import re
import sys
import datetime as _dt
from pathlib import Path

import yaml

PAPERS_PATH = Path("docs/papers.yml")
IMAGES_DIR = Path("docs/src/public/images")

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|webp)$", re.IGNORECASE)
TEMP_BOT_BRANCH_RE = re.compile(r"/paper-images/pr-\d+(/|$)")
ALLOWED_DOCS_URL_RE = re.compile(
    r"^https?://raw\.githubusercontent\.com/MilesCranmer/PySR_Docs/(master|refs/heads/master)/images/[^\s]+$",
    re.IGNORECASE,
)


def fail(msg: str) -> "NoReturn":
    print(f"ERROR: {msg}")
    raise SystemExit(1)


def is_nonempty_str(x: object) -> bool:
    return isinstance(x, str) and x.strip() != ""


def main() -> None:
    if not PAPERS_PATH.exists():
        fail(f"Missing {PAPERS_PATH}")

    raw = PAPERS_PATH.read_text(encoding="utf-8")

    try:
        obj = yaml.safe_load(raw)
    except Exception as e:
        fail(f"YAML parse error in {PAPERS_PATH}: {e}")

    if not isinstance(obj, dict):
        fail(f"{PAPERS_PATH} must be a mapping at top level")

    papers = obj.get("papers")
    if not isinstance(papers, list):
        fail(f"{PAPERS_PATH} must contain `papers:` as a list")

    errors: list[str] = []

    for i, paper in enumerate(papers):
        prefix = f"papers[{i}]"

        if not isinstance(paper, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue

        # Required fields
        title = paper.get("title")
        if not is_nonempty_str(title):
            errors.append(f"{prefix}.title: required non-empty string")

        authors = paper.get("authors")
        if not isinstance(authors, list) or len(authors) == 0 or not all(
            is_nonempty_str(a) for a in authors
        ):
            errors.append(f"{prefix}.authors: required non-empty list of strings")

        link = paper.get("link")
        if not is_nonempty_str(link):
            errors.append(f"{prefix}.link: required non-empty string")

        date = paper.get("date")
        # PyYAML parses unquoted ISO dates as datetime.date
        if isinstance(date, _dt.date):
            date_s = date.isoformat()
        elif isinstance(date, str):
            date_s = date.strip()
        else:
            date_s = ""

        if not date_s or not DATE_RE.match(date_s):
            errors.append(f"{prefix}.date: required YYYY-MM-DD")

        image = paper.get("image")
        if not is_nonempty_str(image):
            errors.append(f"{prefix}.image: required string")
        else:
            image_s = image.strip()

            if image_s.startswith("http://") or image_s.startswith("https://"):
                # We expect authors to upload an image with the PR. URLs are typically
                # only used after a maintainer moves images to PySR_Docs.
                # Allow only the stable raw.githubusercontent.com location in PySR_Docs.
                if TEMP_BOT_BRANCH_RE.search(image_s):
                    errors.append(
                        f"{prefix}.image: points to temporary bot branch URL; use a local uploaded image path (preferred)"
                    )
                elif not ALLOWED_DOCS_URL_RE.match(image_s):
                    errors.append(
                        f"{prefix}.image: URL must be the stable PySR_Docs raw URL (or upload an image in this PR)"
                    )
            else:
                # Must be a basename only (no paths)
                if os.path.basename(image_s) != image_s or "/" in image_s or "\\" in image_s:
                    errors.append(
                        f"{prefix}.image: local image must be a basename (e.g. myfig.jpg), not a path"
                    )
                else:
                    if not IMAGE_EXT_RE.search(image_s):
                        errors.append(
                            f"{prefix}.image: local image must end in .png/.jpg/.jpeg/.webp"
                        )
                    img_path = IMAGES_DIR / image_s
                    if not img_path.exists():
                        errors.append(
                            f"{prefix}.image: local image {image_s!r} not found at {img_path}"
                        )

    if errors:
        print("Papers.yml validation failed:\n")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("OK: docs/papers.yml looks valid")


if __name__ == "__main__":
    main()
