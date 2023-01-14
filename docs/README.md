# PySR Documentation

## Building locally

1. In the base directory, run `pip install -r docs/requirements.txt`.
2. Install PySR in editable mode: `pip install -e .`.
3. Build doc source with `cd docs && ./gen_docs.sh && cd ..`.
4. Create and serve docs with mkdocs: `mkdocs serve -w pysr`.
