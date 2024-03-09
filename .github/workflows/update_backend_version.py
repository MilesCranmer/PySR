import json
import sys
from pathlib import Path

import tomlkit

new_backend_version = sys.argv[1]

assert not new_backend_version.startswith("v"), "Version should not start with 'v'"

pyproject_toml = Path(__file__).parent / ".." / ".." / "pyproject.toml"
juliapkg_json = Path(__file__).parent / ".." / ".." / "pysr" / "juliapkg.json"

with open(pyproject_toml) as toml_file:
    pyproject_data = tomlkit.parse(toml_file.read())

with open(juliapkg_json) as f:
    juliapkg_data = json.load(f)

major, minor, patch, *dev = pyproject_data["project"]["version"].split(".")
pyproject_data["project"]["version"] = f"{major}.{minor}.{int(patch)+1}"

juliapkg_data["packages"]["SymbolicRegression"]["version"] = f"={new_backend_version}"

with open(pyproject_toml, "w") as toml_file:
    toml_file.write(tomlkit.dumps(pyproject_data))

with open(juliapkg_json, "w") as f:
    json.dump(juliapkg_data, f, indent=4)
    # Ensure ends with newline
    f.write("\n")
