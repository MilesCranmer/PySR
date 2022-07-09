"""Functions to help export PySR equations to LaTeX."""
import sympy
from sympy.printing.latex import LatexPrinter
import pandas as pd
from typing import List


class PreciseLatexPrinter(LatexPrinter):
    """Modified SymPy printer with custom float precision."""

    def __init__(self, settings=None, prec=3):
        super().__init__(settings)
        self.prec = prec

    def _print_Float(self, expr):
        # Reduce precision of float:
        reduced_float = sympy.Float(expr, self.prec)
        return super()._print_Float(reduced_float)


def to_latex(expr, prec=3, full_prec=True, **settings):
    """Convert sympy expression to LaTeX with custom precision."""
    settings["full_prec"] = full_prec
    printer = PreciseLatexPrinter(settings=settings, prec=prec)
    return printer.doprint(expr)


def generate_table_environment(columns=["equation", "complexity", "loss"]):
    margins = "".join([("l" if col == "equation" else "c") for col in columns])
    column_map = {
        "complexity": "Complexity",
        "loss": "Loss",
        "equation": "Equation",
        "score": "Score",
    }
    columns = [column_map[col] for col in columns]
    top_pieces = [
        r"\begin{table}[h]",
        r"\begin{center}",
        r"\begin{tabular}{@{}" + margins + r"@{}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    bottom_pieces = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    top_latex_table = "\n".join(top_pieces)
    bottom_latex_table = "\n".join(bottom_pieces)

    return top_latex_table, bottom_latex_table


def generate_table(
    equations: List[pd.DataFrame],
    indices: List[List[int]],
    precision=3,
    columns=["equation", "complexity", "loss", "score"],
):
    latex_top, latex_bottom = generate_table_environment(columns)

    latex_equations = [
        [to_latex(eq, prec=precision) for eq in equation_set["sympy_format"]]
        for equation_set in equations
    ]

    all_latex_table_str = []

    for output_feature, index_set in enumerate(indices):
        latex_table_content = []
        for i in index_set:
            latex_equation = latex_equations[output_feature][i]
            complexity = str(equations[output_feature].iloc[i]["complexity"])
            loss = to_latex(
                sympy.Float(equations[output_feature].iloc[i]["loss"]),
                prec=precision,
            )
            score = to_latex(
                sympy.Float(equations[output_feature].iloc[i]["score"]),
                prec=precision,
            )

            row_pieces = []
            for col in columns:
                if col == "equation":
                    row_pieces.append(latex_equation)
                elif col == "complexity":
                    row_pieces.append(complexity)
                elif col == "loss":
                    row_pieces.append(loss)
                elif col == "score":
                    row_pieces.append(score)
                else:
                    raise ValueError(f"Unknown column: {col}")

            row_pieces = ["$" + piece + "$" for piece in row_pieces]

            latex_table_content.append(
                " & ".join(row_pieces) + r" \\",
            )

        this_latex_table = "\n".join(
            [
                latex_top,
                *latex_table_content,
                latex_bottom,
            ]
        )
        all_latex_table_str.append(this_latex_table)

    return "\n\n".join(all_latex_table_str)
