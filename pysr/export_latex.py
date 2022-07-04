"""Functions to help export PySR equations to LaTeX."""
import sympy
from sympy.printing.latex import LatexPrinter


class PreciseLatexPrinter(LatexPrinter):
    """Modified SymPy printer with custom float precision."""

    def __init__(self, settings=None, prec=3, full_prec=True):
        super().__init__(settings)
        self.prec = prec
        self.full_prec = full_prec

    def _print_Float(self, expr):
        # Reduce precision of float:
        reduced_float = sympy.Float(expr, self.prec, full_prec=self.full_prec)
        return super()._print_Float(reduced_float)


def to_latex(expr, prec=3, full_prec=True, **settings):
    """Convert sympy expression to LaTeX with custom precision."""
    if len(settings) == 0:
        settings = None
    printer = PreciseLatexPrinter(
        settings=settings, prec=prec, full_prec=full_prec
    )
    return printer.doprint(expr)


def generate_top_of_latex_table(columns=["Equation", "Complexity", "Loss"]):
    margins = "".join([("l" if col == "Equation" else "c") for col in columns])
    latex_table_pieces = [
        r"\begin{table}[h]",
        r"\begin{center}",
        r"\begin{tabular}{@{}" + margins + r"@{}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
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
