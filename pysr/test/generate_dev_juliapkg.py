# Example call:
## python3 generate_dev_juliapkg.py /pysr/pysr/juliapkg.json /srjl
import json
import sys

juliapkg_json = sys.argv[1]
path_to_srjl = sys.argv[2]

with open(juliapkg_json, "r") as f:
    juliapkg = json.load(f)

del juliapkg["packages"]["SymbolicRegression"]["version"]
juliapkg["packages"]["SymbolicRegression"]["path"] = path_to_srjl
juliapkg["packages"]["SymbolicRegression"]["dev"] = True

with open(juliapkg_json, "w") as f:
    json.dump(juliapkg, f, indent=4)
