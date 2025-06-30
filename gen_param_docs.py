# Load YAML file param_groupings.yml:
import re
import sys

from docstring_parser import parse
from yaml import safe_load

sys.path.append("..")


from pysr import PySRRegressor

found_params = []


def str_param_groups(param_groupings, params, cur_heading=2):
    global found_params
    # Recursively print the parameter descriptions, defaults,
    # with headings from the param groupings dict.
    if isinstance(param_groupings, list):
        return "\n\n".join(
            str_param_groups(param, params, cur_heading) for param in param_groupings
        )
    elif isinstance(param_groupings, dict):
        for heading, param_grouping in param_groupings.items():
            return (
                f"{'#' * cur_heading} {heading}"
                + "\n\n"
                + str_param_groups(param_grouping, params, cur_heading + 1)
            )
    elif isinstance(param_groupings, str):
        found_params.append(param_groupings)

        default_value = re.search(
            r"Default is `(.*)`", params[param_groupings].description
        )
        clean_desc = re.sub(r"Default is .*", "", params[param_groupings].description)
        # Prepend every line with 4 spaces:
        clean_desc = "\n".join("    " + line for line in clean_desc.splitlines())
        return (
            f"  - **`{param_groupings}`**"
            + "\n\n"
            + clean_desc
            + (
                "\n\n    " + f"*Default:* `{default_value.group(1)}`"
                if default_value
                else ""
            )
        )
    else:
        raise TypeError(f"Unexpected type {type(param_groupings)}")


if __name__ == "__main__":
    # This is the path to the param_groupings.yml file
    # relative to the current file.
    path = "../pysr/param_groupings.yml"
    with open(path, "r") as f:
        param_groupings = safe_load(f)

    # This is basically a dict of lists and dicts.

    # Let's load in the parameter descriptions from the docstring of PySRRegressor:
    raw_params = parse(PySRRegressor.__doc__).params
    params = {
        param.arg_name: param
        for param in raw_params
        if param.arg_name[-1] != "_" and param.arg_name != "**kwargs"
    }

    output = str_param_groups(param_groupings, params, cur_heading=3)
    assert len(set(found_params) ^ set(params.keys())) == 0
    print("## PySRRegressor Parameters")
    print(output)
