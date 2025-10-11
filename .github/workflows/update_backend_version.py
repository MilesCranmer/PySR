import json
import re
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

current_version = pyproject_data["project"]["version"]
parts = current_version.split(".")

if len(parts) < 3:
    raise ValueError(
        f"Invalid version format: {current_version}. Expected at least 3 components (major.minor.patch)"
    )

major, minor = parts[0], parts[1]

patch_match = re.match(r"^(\d+)(.*)$", parts[2])
if not patch_match:
    raise ValueError(
        f"Could not parse patch version from '{parts[2]}' in version {current_version}. "
        f"Expected patch to start with a number (e.g., '0', '1a1', '2rc3')"
    )

patch_num_str, patch_suffix = patch_match.groups()
patch_num = int(patch_num_str)

pre_release_match = re.fullmatch(r"(a|b|rc)(\d+)", patch_suffix)
if pre_release_match:
    pre_tag, pre_num = pre_release_match.groups()
    new_patch = patch_num
    new_suffix = f"{pre_tag}{int(pre_num) + 1}"
else:
    new_patch = patch_num + 1
    new_suffix = patch_suffix

# Add back any additional version components (e.g., "2.0.0.dev1" -> ".dev1")
extra_parts = "." + ".".join(parts[3:]) if len(parts) > 3 else ""
new_version = f"{major}.{minor}.{new_patch}{new_suffix}{extra_parts}"

pyproject_data["project"]["version"] = new_version

# Update backend - maintain current format (either "rev" or "version")
backend_pkg = juliapkg_data["packages"]["SymbolicRegression"]
if "rev" in backend_pkg:
    backend_pkg["rev"] = f"v{new_backend_version}"
elif "version" in backend_pkg:
    backend_pkg["version"] = f"~{new_backend_version}"
else:
    raise ValueError(
        "SymbolicRegression package must have either 'rev' or 'version' field"
    )

with open(pyproject_toml, "w") as toml_file:
    toml_file.write(tomlkit.dumps(pyproject_data))

with open(juliapkg_json, "w") as f:
    json.dump(juliapkg_data, f, indent=4)
    # Ensure ends with newline
    f.write("\n")
