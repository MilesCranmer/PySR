"""Functions to help export PySR equations to LaTeX."""
import sympy
from sympy.printing.latex import LatexPrinter
import pandas as pd
from typing import List
import warnings


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
    margins = "c" * len(columns)
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


def generate_single_table(
    equations: pd.DataFrame,
    indices: List[int] = None,
    precision: int = 3,
    columns=["equation", "complexity", "loss", "score"],
    max_equation_length: int = 50,
    output_variable_name: str = "y",
):
    """Generate a booktabs-style LaTeX table for a single set of equations."""
    assert isinstance(equations, pd.DataFrame)

    latex_top, latex_bottom = generate_table_environment(columns)
    latex_table_content = []

    if indices is None:
        indices = range(len(equations))

    for i in indices:
        latex_equation = to_latex(
            equations.iloc[i]["sympy_format"],
            prec=precision,
        )
        complexity = str(equations.iloc[i]["complexity"])
        loss = to_latex(
            sympy.Float(equations.iloc[i]["loss"]),
            prec=precision,
        )
        score = to_latex(
            sympy.Float(equations.iloc[i]["score"]),
            prec=precision,
        )

        row_pieces = []
        for col in columns:
            if col == "equation":
                if len(latex_equation) < max_equation_length:
                    row_pieces.append(
                        "$" + output_variable_name + " = " + latex_equation + "$"
                    )
                else:

                    broken_latex_equation = " ".join(
                        [
                            r"\begin{minipage}{0.8\linewidth}",
                            r"\vspace{-1em}",
                            r"\begin{dmath*}",
                            output_variable_name + " = " + latex_equation,
                            r"\end{dmath*}",
                            r"\end{minipage}",
                        ]
                    )
                    row_pieces.append(broken_latex_equation)

            elif col == "complexity":
                row_pieces.append("$" + complexity + "$")
            elif col == "loss":
                row_pieces.append("$" + loss + "$")
            elif col == "score":
                row_pieces.append("$" + score + "$")
            else:
                raise ValueError(f"Unknown column: {col}")

        latex_table_content.append(
            " & ".join(row_pieces) + r" \\",
        )

    return "\n".join([latex_top, *latex_table_content, latex_bottom])


def generate_multiple_tables(
    equations: List[pd.DataFrame],
    indices: List[List[int]] = None,
    precision: int = 3,
    columns=["equation", "complexity", "loss", "score"],
    output_variable_names: str = None,
):
    """Generate multiple latex tables for a list of equation sets."""
    # TODO: Let user specify custom output variable

    latex_tables = [
        generate_single_table(
            equations[i],
            (None if not indices else indices[i]),
            precision=precision,
            columns=columns,
            output_variable_name=(
                "y_{" + str(i) + "}"
                if output_variable_names is None
                else output_variable_names[i]
            ),
        )
        for i in range(len(equations))
    ]

    return "\n\n".join(latex_tables)
