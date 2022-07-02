"""Functions to help export PySR equations to LaTeX."""
import re


def set_precision_of_constants_in_string(s, precision=3):
    """Set precision of constants in string."""
    constants = re.findall(r"\b[-+]?\d*\.\d+|\b[-+]?\d+\.?\d*", s)
    for c in constants:
        reduced_c = "{:.{precision}g}".format(float(c), precision=precision)
        s = s.replace(c, reduced_c)
    return s


def generate_top_of_latex_table():
    latex_table_pieces = [
        r"\begin{table}[h]",
        r"\begin{center}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Equation & Complexity & Loss \\",
        r"\midrule",
    ]
    return "\n".join(latex_table_pieces)


def generate_bottom_of_latex_table():
    latex_table_pieces = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    return "\n".join(latex_table_pieces)
