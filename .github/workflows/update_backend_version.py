import json
import sys
from pathlib import Path

import toml

new_backend_version = sys.argv[1]

pyproject_toml = Path(__file__).parent / ".." / ".." / "pyproject.toml"
juliapkg_json = Path(__file__).parent / ".." / ".." / "pysr" / "juliapkg.json"

with open(pyproject_toml) as toml_file:
    pyproject_data = toml.load(toml_file)

with open(juliapkg_json) as f:
    juliapkg_data = json.load(f)

major, minor, patch, *dev = pyproject_data["project"]["version"].split(".")
pyproject_data["project"]["version"] = f"{major}.{minor}.{int(patch)+1}"

juliapkg_data["packages"]["SymbolicRegression"]["version"] = f"={new_backend_version}"

with open(pyproject_toml, "w") as toml_file:
    toml.dump(pyproject_data, toml_file)

with open(juliapkg_json, "w") as f:
    json.dump(juliapkg_data, f, indent=4)
