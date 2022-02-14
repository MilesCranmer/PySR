import os
import sys
import numpy as np
import pandas as pd
import sympy
from sympy import sympify, lambdify
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import warnings
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator, RegressorMixin
from collections import OrderedDict
from hashlib import sha256

from .version import __version__, __symbolic_regression_jl_version__

is_julia_warning_silenced = False


def install(julia_project=None, quiet=False):  # pragma: no cover
    """Install PyCall.jl and all required dependencies for SymbolicRegression.jl.

    Also updates the local Julia registry."""
    import julia

    julia.install(quiet=quiet)

    julia_project, is_shared = _get_julia_project(julia_project)

    Main = init_julia()
    Main.eval("using Pkg")

    io = "devnull" if quiet else "stderr"
    io_arg = f"io={io}" if is_julia_version_greater_eq(Main, "1.6") else ""

    # Can't pass IO to Julia call as it evaluates to PyObject, so just directly
    # use Main.eval:
    Main.eval(
        f'Pkg.activate("{_escape_filename(julia_project)}", shared = Bool({int(is_shared)}), {io_arg})'
    )
    if is_shared:
        # Install SymbolicRegression.jl:
        _add_sr_to_julia_project(Main, io_arg)

    Main.eval(f"Pkg.instantiate({io_arg})")
    Main.eval(f"Pkg.precompile({io_arg})")
    if not quiet:
        warnings.warn(
            "It is recommended to restart Python after installing PySR's dependencies,"
            " so that the Julia environment is properly initialized."
        )


def import_error_string(julia_project=None):
    s = f"""
    Required dependencies are not installed or built.  Run the following code in the Python REPL:

        >>> import pysr
        >>> pysr.install()
    """

    if julia_project is not None:
        s += f"""
        Tried to activate project {julia_project} but failed."""

    return s


Main = None

already_ran = False

sympy_mappings = {
    "div": lambda x, y: x / y,
    "mult": lambda x, y: x * y,
    "sqrt_abs": lambda x: sympy.sqrt(abs(x)),
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "plus": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "neg": lambda x: -x,
    "pow": lambda x, y: abs(x) ** y,
    "cos": sympy.cos,
    "sin": sympy.sin,
    "tan": sympy.tan,
    "cosh": sympy.cosh,
    "sinh": sympy.sinh,
    "tanh": sympy.tanh,
    "exp": sympy.exp,
    "acos": sympy.acos,
    "asin": sympy.asin,
    "atan": sympy.atan,
    "acosh": lambda x: sympy.acosh(abs(x) + 1),
    "acosh_abs": lambda x: sympy.acosh(abs(x) + 1),
    "asinh": sympy.asinh,
    "atanh": lambda x: sympy.atanh(sympy.Mod(x + 1, 2) - 1),
    "atanh_clip": lambda x: sympy.atanh(sympy.Mod(x + 1, 2) - 1),
    "abs": abs,
    "mod": sympy.Mod,
    "erf": sympy.erf,
    "erfc": sympy.erfc,
    "log_abs": lambda x: sympy.log(abs(x)),
    "log10_abs": lambda x: sympy.log(abs(x), 10),
    "log2_abs": lambda x: sympy.log(abs(x), 2),
    "log1p_abs": lambda x: sympy.log(abs(x) + 1),
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "sign": sympy.sign,
    "gamma": sympy.gamma,
}


def pysr(X, y, weights=None, **kwargs):
    warnings.warn(
        "Calling `pysr` is deprecated. Please use `model = PySRRegressor(**params); model.fit(X, y)` going forward.",
        DeprecationWarning,
    )
    model = PySRRegressor(**kwargs)
    model.fit(X, y, weights=weights)
    return model.equations


def _handle_constraints(binary_operators, unary_operators, constraints):
    for op in unary_operators:
        if op not in constraints:
            constraints[op] = -1
    for op in binary_operators:
        if op not in constraints:
            constraints[op] = (-1, -1)
        if op in ["plus", "sub"]:
            if constraints[op][0] != constraints[op][1]:
                raise NotImplementedError(
                    "You need equal constraints on both sides for - and *, due to simplification strategies."
                )
        elif op == "mult":
            # Make sure the complex expression is in the left side.
            if constraints[op][0] == -1:
                continue
            if constraints[op][1] == -1 or constraints[op][0] < constraints[op][1]:
                constraints[op][0], constraints[op][1] = (
                    constraints[op][1],
                    constraints[op][0],
                )


def _create_inline_operators(binary_operators, unary_operators):
    global Main
    for op_list in [binary_operators, unary_operators]:
        for i, op in enumerate(op_list):
            is_user_defined_operator = "(" in op

            if is_user_defined_operator:
                Main.eval(op)
                # Cut off from the first non-alphanumeric char:
                first_non_char = [
                    j
                    for j, char in enumerate(op)
                    if not (char.isalpha() or char.isdigit())
                ][0]
                function_name = op[:first_non_char]
                op_list[i] = function_name


def _handle_feature_selection(X, select_k_features, y, variable_names):
    if select_k_features is not None:
        selection = run_feature_selection(X, y, select_k_features)
        print(f"Using features {[variable_names[i] for i in selection]}")
        X = X[:, selection]

    else:
        selection = None
    return X, selection


def _check_assertions(
    X,
    binary_operators,
    unary_operators,
    use_custom_variable_names,
    variable_names,
    weights,
    y,
):
    # Check for potential errors before they happen
    assert len(unary_operators) + len(binary_operators) > 0
    assert len(X.shape) == 2
    assert len(y.shape) in [1, 2]
    assert X.shape[0] == y.shape[0]
    if weights is not None:
        assert weights.shape == y.shape
        assert X.shape[0] == weights.shape[0]
    if use_custom_variable_names:
        assert len(variable_names) == X.shape[1]


def run_feature_selection(X, y, select_k_features):
    """Use a gradient boosting tree regressor as a proxy for finding
    the k most important features in X, returning indices for those
    features as output."""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    clf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(X, y)
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    return selector.get_support(indices=True)


def _escape_filename(filename):
    """Turns a file into a string representation with correctly escaped backslashes"""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def best(*args, **kwargs):
    raise NotImplementedError(
        "`best` has been deprecated. Please use the `PySRRegressor` interface. After fitting, you can return `.sympy()` to get the sympy representation of the best equation."
    )


def best_row(*args, **kwargs):
    raise NotImplementedError(
        "`best_row` has been deprecated. Please use the `PySRRegressor` interface. After fitting, you can run `print(model)` to view the best equation, or `model.get_best()` to return the best equation's row in `model.equations`."
    )


def best_tex(*args, **kwargs):
    raise NotImplementedError(
        "`best_tex` has been deprecated. Please use the `PySRRegressor` interface. After fitting, you can return `.latex()` to get the sympy representation of the best equation."
    )


def best_callable(*args, **kwargs):
    raise NotImplementedError(
        "`best_callable` has been deprecated. Please use the `PySRRegressor` interface. After fitting, you can use `.predict(X)` to use the best callable."
    )


def _denoise(X, y, Xresampled=None):
    """Denoise the dataset using a Gaussian process"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    gp_kernel = RBF(np.ones(X.shape[1])) + WhiteKernel(1e-1) + ConstantKernel()
    gpr = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=50)
    gpr.fit(X, y)
    if Xresampled is not None:
        return Xresampled, gpr.predict(Xresampled)

    return X, gpr.predict(X)


class CallableEquation:
    """Simple wrapper for numpy lambda functions built with sympy"""

    def __init__(self, sympy_symbols, eqn, selection=None, variable_names=None):
        self._sympy = eqn
        self._sympy_symbols = sympy_symbols
        self._selection = selection
        self._variable_names = variable_names
        self._lambda = lambdify(sympy_symbols, eqn)

    def __repr__(self):
        return f"PySRFunction(X=>{self._sympy})"

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            # Lambda function takes as argument:
            return self._lambda(**{k: X[k].values for k in X.columns})
        elif self._selection is not None:
            return self._lambda(*X[:, self._selection].T)
        return self._lambda(*X.T)


def _get_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        julia_project = f"pysr-{__version__}"
    else:
        is_shared = False
        julia_project = Path(julia_project)
    return julia_project, is_shared


def silence_julia_warning():
    global is_julia_warning_silenced
    is_julia_warning_silenced = True


def is_julia_version_greater_eq(Main, version="1.6"):
    """Check if Julia version is greater than specified version."""
    return Main.eval(f'VERSION >= v"{version}"')


def init_julia():
    """Initialize julia binary, turning off compiled modules if needed."""
    global is_julia_warning_silenced
    from julia.core import JuliaInfo, UnsupportedPythonError

    info = JuliaInfo.load(julia="julia")
    if not info.is_pycall_built():
        raise ImportError(import_error_string())

    Main = None
    try:
        from julia import Main as _Main

        Main = _Main
    except UnsupportedPythonError:
        if not is_julia_warning_silenced:
            warnings.warn(
                """
Your Python version is statically linked to libpython. For example, this could be the python included with conda, or maybe your system's built-in python.
This will still work, but the precompilation cache for Julia will be turned off, which may result in slower startup times on the initial pysr() call.

To install a Python version that is dynamically linked to libpython, pyenv is recommended (https://github.com/pyenv/pyenv). With pyenv, you can run: `PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.10` to install Python 3.9.10 with dynamic linking.

To silence this warning, you can run pysr.silence_julia_warning() after importing pysr."""
            )
        from julia.core import Julia

        jl = Julia(compiled_modules=False)
        from julia import Main as _Main

        Main = _Main

    return Main


def _add_sr_to_julia_project(Main, io_arg):
    Main.spec = Main.PackageSpec(
        name="SymbolicRegression",
        url="https://github.com/MilesCranmer/SymbolicRegression.jl",
        rev="v" + __symbolic_regression_jl_version__,
    )
    Main.eval(f"Pkg.add(spec, {io_arg})")


class PySRRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model_selection="best",
        *,
        weights=None,
        binary_operators=None,
        unary_operators=None,
        procs=cpu_count(),
        loss="L2DistLoss()",
        populations=100,
        niterations=4,
        ncyclesperiteration=100,
        timeout_in_seconds=None,
        alpha=0.1,
        annealing=False,
        fractionReplaced=0.01,
        fractionReplacedHof=0.005,
        npop=100,
        parsimony=1e-4,
        migration=True,
        hofMigration=True,
        shouldOptimizeConstants=True,
        topn=10,
        weightAddNode=1,
        weightInsertNode=3,
        weightDeleteNode=3,
        weightDoNothing=1,
        weightMutateConstant=10,
        weightMutateOperator=1,
        weightRandomize=1,
        weightSimplify=0.002,
        crossoverProbability=0.01,
        perturbationFactor=1.0,
        extra_sympy_mappings=None,
        extra_torch_mappings=None,
        extra_jax_mappings=None,
        equation_file=None,
        verbosity=1e9,
        update_verbosity=None,
        progress=None,
        maxsize=20,
        fast_cycle=False,
        maxdepth=None,
        variable_names=None,
        batching=False,
        batchSize=50,
        select_k_features=None,
        warmupMaxsizeBy=0.0,
        constraints=None,
        useFrequency=True,
        tempdir=None,
        delete_tempfiles=True,
        julia_project=None,
        update=True,
        temp_equation_file=False,
        output_jax_format=False,
        output_torch_format=False,
        optimizer_algorithm="BFGS",
        optimizer_nrestarts=3,
        optimize_probability=1.0,
        optimizer_iterations=10,
        tournament_selection_n=10,
        tournament_selection_p=1.0,
        denoise=False,
        Xresampled=None,
        precision=32,
        multithreading=None,
        use_symbolic_utils=False,
    ):
        """Initialize settings for an equation search in PySR.

        Note: most default parameters have been tuned over several example
        equations, but you should adjust `niterations`,
        `binary_operators`, `unary_operators` to your requirements.
        You can view more detailed explanations of the options on the
        [options page](https://pysr.readthedocs.io/en/latest/docs/options/) of the documentation.

        :param model_selection: How to select a model. Can be 'accuracy' or 'best'. The default, 'best', will optimize a combination of complexity and accuracy.
        :type model_selection: str
        :param binary_operators: List of strings giving the binary operators in Julia's Base. Default is ["+", "-", "*", "/",].
        :type binary_operators: list
        :param unary_operators: Same but for operators taking a single scalar. Default is [].
        :type unary_operators: list
        :param niterations: Number of iterations of the algorithm to run. The best equations are printed, and migrate between populations, at the end of each.
        :type niterations: int
        :param populations: Number of populations running.
        :type populations: int
        :param loss: String of Julia code specifying the loss function.  Can either be a loss from LossFunctions.jl, or your own loss written as a function. Examples of custom written losses include: `myloss(x, y) = abs(x-y)` for non-weighted, or `myloss(x, y, w) = w*abs(x-y)` for weighted.  Among the included losses, these are as follows. Regression: `LPDistLoss{P}()`, `L1DistLoss()`, `L2DistLoss()` (mean square), `LogitDistLoss()`, `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`, `PeriodicLoss(c)`, `QuantileLoss(τ)`.  Classification: `ZeroOneLoss()`, `PerceptronLoss()`, `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`, `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`, `SigmoidLoss()`, `DWDMarginLoss(q)`.
        :type loss: str
        :param denoise: Whether to use a Gaussian Process to denoise the data before inputting to PySR. Can help PySR fit noisy data.
        :type denoise: bool
        :param select_k_features: whether to run feature selection in Python using random forests, before passing to the symbolic regression code. None means no feature selection; an int means select that many features.
        :type select_k_features: None/int
        :param procs: Number of processes (=number of populations running).
        :type procs: int
        :param multithreading: Use multithreading instead of distributed backend. Default is yes. Using procs=0 will turn off both.
        :type multithreading: bool
        :param batching: whether to compare population members on small batches during evolution. Still uses full dataset for comparing against hall of fame.
        :type batching: bool
        :param batchSize: the amount of data to use if doing batching.
        :type batchSize: int
        :param maxsize: Max size of an equation.
        :type maxsize: int
        :param ncyclesperiteration: Number of total mutations to run, per 10 samples of the population, per iteration.
        :type ncyclesperiteration: int
        :param timeout_in_seconds: Make the search return early once this many seconds have passed.
        :type timeout_in_seconds: float/int
        :param alpha: Initial temperature.
        :type alpha: float
        :param annealing: Whether to use annealing. You should (and it is default).
        :type annealing: bool
        :param fractionReplaced: How much of population to replace with migrating equations from other populations.
        :type fractionReplaced: float
        :param fractionReplacedHof: How much of population to replace with migrating equations from hall of fame.
        :type fractionReplacedHof: float
        :param npop: Number of individuals in each population
        :type npop: int
        :param parsimony: Multiplicative factor for how much to punish complexity.
        :type parsimony: float
        :param migration: Whether to migrate.
        :type migration: bool
        :param hofMigration: Whether to have the hall of fame migrate.
        :type hofMigration: bool
        :param shouldOptimizeConstants: Whether to numerically optimize constants (Nelder-Mead/Newton) at the end of each iteration.
        :type shouldOptimizeConstants: bool
        :param topn: How many top individuals migrate from each population.
        :type topn: int
        :param perturbationFactor: Constants are perturbed by a max factor of (perturbationFactor*T + 1). Either multiplied by this or divided by this.
        :type perturbationFactor: float
        :param weightAddNode: Relative likelihood for mutation to add a node
        :type weightAddNode: float
        :param weightInsertNode: Relative likelihood for mutation to insert a node
        :type weightInsertNode: float
        :param weightDeleteNode: Relative likelihood for mutation to delete a node
        :type weightDeleteNode: float
        :param weightDoNothing: Relative likelihood for mutation to leave the individual
        :type weightDoNothing: float
        :param weightMutateConstant: Relative likelihood for mutation to change the constant slightly in a random direction.
        :type weightMutateConstant: float
        :param weightMutateOperator: Relative likelihood for mutation to swap an operator.
        :type weightMutateOperator: float
        :param weightRandomize: Relative likelihood for mutation to completely delete and then randomly generate the equation
        :type weightRandomize: float
        :param weightSimplify: Relative likelihood for mutation to simplify constant parts by evaluation
        :type weightSimplify: float
        :param crossoverProbability: Absolute probability of crossover-type genetic operation, instead of a mutation.
        :type crossoverProbability: float
        :param equation_file: Where to save the files (.csv separated by |)
        :type equation_file: str
        :param verbosity: What verbosity level to use. 0 means minimal print statements.
        :type verbosity: int
        :param update_verbosity: What verbosity level to use for package updates. Will take value of `verbosity` if not given.
        :type update_verbosity: int
        :param progress: Whether to use a progress bar instead of printing to stdout.
        :type progress: bool
        :param maxdepth: Max depth of an equation. You can use both maxsize and maxdepth.  maxdepth is by default set to = maxsize, which means that it is redundant.
        :type maxdepth: int
        :param fast_cycle: (experimental) - batch over population subsamples. This is a slightly different algorithm than regularized evolution, but does cycles 15% faster. May be algorithmically less efficient.
        :type fast_cycle: bool
        :param variable_names: a list of names for the variables, other than "x0", "x1", etc.
        :type variable_names: list
        :param warmupMaxsizeBy: whether to slowly increase max size from a small number up to the maxsize (if greater than 0).  If greater than 0, says the fraction of training time at which the current maxsize will reach the user-passed maxsize.
        :type warmupMaxsizeBy: float
        :param constraints: dictionary of int (unary) or 2-tuples (binary), this enforces maxsize constraints on the individual arguments of operators. E.g., `'pow': (-1, 1)` says that power laws can have any complexity left argument, but only 1 complexity exponent. Use this to force more interpretable solutions.
        :type constraints: dict
        :param useFrequency: whether to measure the frequency of complexities, and use that instead of parsimony to explore equation space. Will naturally find equations of all complexities.
        :type useFrequency: bool
        :param tempdir: directory for the temporary files
        :type tempdir: str/None
        :param delete_tempfiles: whether to delete the temporary files after finishing
        :type delete_tempfiles: bool
        :param julia_project: a Julia environment location containing a Project.toml (and potentially the source code for SymbolicRegression.jl).  Default gives the Python package directory, where a Project.toml file should be present from the install.
        :type julia_project: str/None
        :param update: Whether to automatically update Julia packages.
        :type update: bool
        :param temp_equation_file: Whether to put the hall of fame file in the temp directory. Deletion is then controlled with the delete_tempfiles argument.
        :type temp_equation_file: bool
        :param output_jax_format: Whether to create a 'jax_format' column in the output, containing jax-callable functions and the default parameters in a jax array.
        :type output_jax_format: bool
        :param output_torch_format: Whether to create a 'torch_format' column in the output, containing a torch module with trainable parameters.
        :type output_torch_format: bool
        :param tournament_selection_n: Number of expressions to consider in each tournament.
        :type tournament_selection_n: int
        :param tournament_selection_p: Probability of selecting the best expression in each tournament. The probability will decay as p*(1-p)^n for other expressions, sorted by loss.
        :type tournament_selection_p: float
        :param precision: What precision to use for the data. By default this is 32 (float32), but you can select 64 or 16 as well.
        :type precision: int
        :param use_symbolic_utils: Whether to use SymbolicUtils during simplification.
        :type use_symbolic_utils: bool
        :returns: Initialized model. Call `.fit(X, y)` to fit your data!
        :type: PySRRegressor
        """
        super().__init__()
        # TODO: Order args in docstring by order of declaration.
        self.model_selection = model_selection

        if binary_operators is None:
            binary_operators = "+ * - /".split(" ")
        if unary_operators is None:
            unary_operators = []
        if extra_sympy_mappings is None:
            extra_sympy_mappings = {}
        if variable_names is None:
            variable_names = []
        if constraints is None:
            constraints = {}
        if multithreading is None:
            # Default is multithreading=True, unless explicitly set,
            # or procs is set to 0 (serial mode).
            multithreading = procs != 0
        if update_verbosity is None:
            update_verbosity = verbosity

        buffer_available = "buffer" in sys.stdout.__dir__()

        if progress is not None:
            if progress and not buffer_available:
                warnings.warn(
                    "Note: it looks like you are running in Jupyter. The progress bar will be turned off."
                )
                progress = False
        else:
            progress = buffer_available

        assert optimizer_algorithm in ["NelderMead", "BFGS"]
        assert tournament_selection_n < npop

        if extra_jax_mappings is not None:
            for value in extra_jax_mappings.values():
                if not isinstance(value, str):
                    raise NotImplementedError(
                        "extra_jax_mappings must have keys that are strings! e.g., {sympy.sqrt: 'jnp.sqrt'}."
                    )
        else:
            extra_jax_mappings = {}

        if extra_torch_mappings is not None:
            for value in extra_jax_mappings.values():
                if not callable(value):
                    raise NotImplementedError(
                        "extra_torch_mappings must be callable functions! e.g., {sympy.sqrt: torch.sqrt}."
                    )
        else:
            extra_torch_mappings = {}

        if maxsize > 40:
            warnings.warn(
                "Note: Using a large maxsize for the equation search will be exponentially slower and use significant memory. You should consider turning `useFrequency` to False, and perhaps use `warmupMaxsizeBy`."
            )
        elif maxsize < 7:
            raise NotImplementedError("PySR requires a maxsize of at least 7")

        if maxdepth is None:
            maxdepth = maxsize

        if isinstance(binary_operators, str):
            binary_operators = [binary_operators]
        if isinstance(unary_operators, str):
            unary_operators = [unary_operators]

        self.params = {
            **dict(
                weights=weights,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                procs=procs,
                loss=loss,
                populations=populations,
                niterations=niterations,
                ncyclesperiteration=ncyclesperiteration,
                timeout_in_seconds=timeout_in_seconds,
                alpha=alpha,
                annealing=annealing,
                fractionReplaced=fractionReplaced,
                fractionReplacedHof=fractionReplacedHof,
                npop=npop,
                parsimony=float(parsimony),
                migration=migration,
                hofMigration=hofMigration,
                shouldOptimizeConstants=shouldOptimizeConstants,
                topn=topn,
                weightAddNode=weightAddNode,
                weightInsertNode=weightInsertNode,
                weightDeleteNode=weightDeleteNode,
                weightDoNothing=weightDoNothing,
                weightMutateConstant=weightMutateConstant,
                weightMutateOperator=weightMutateOperator,
                weightRandomize=weightRandomize,
                weightSimplify=weightSimplify,
                crossoverProbability=crossoverProbability,
                perturbationFactor=perturbationFactor,
                verbosity=verbosity,
                update_verbosity=update_verbosity,
                progress=progress,
                maxsize=maxsize,
                fast_cycle=fast_cycle,
                maxdepth=maxdepth,
                batching=batching,
                batchSize=batchSize,
                select_k_features=select_k_features,
                warmupMaxsizeBy=warmupMaxsizeBy,
                constraints=constraints,
                useFrequency=useFrequency,
                tempdir=tempdir,
                delete_tempfiles=delete_tempfiles,
                update=update,
                temp_equation_file=temp_equation_file,
                optimizer_algorithm=optimizer_algorithm,
                optimizer_nrestarts=optimizer_nrestarts,
                optimize_probability=optimize_probability,
                optimizer_iterations=optimizer_iterations,
                tournament_selection_n=tournament_selection_n,
                tournament_selection_p=tournament_selection_p,
                denoise=denoise,
                Xresampled=Xresampled,
                precision=precision,
                multithreading=multithreading,
                use_symbolic_utils=use_symbolic_utils,
            ),
        }

        # Stored equations:
        self.equations = None
        self.params_hash = None
        self.raw_julia_state = None

        self.multioutput = None
        self.equation_file = equation_file
        self.n_features = None
        self.extra_sympy_mappings = extra_sympy_mappings
        self.extra_torch_mappings = extra_torch_mappings
        self.extra_jax_mappings = extra_jax_mappings
        self.output_jax_format = output_jax_format
        self.output_torch_format = output_torch_format
        self.nout = 1
        self.selection = None
        self.variable_names = variable_names
        self.julia_project = julia_project

        self.surface_parameters = [
            "model_selection",
            "multioutput",
            "equation_file",
            "n_features",
            "extra_sympy_mappings",
            "extra_torch_mappings",
            "extra_jax_mappings",
            "output_jax_format",
            "output_torch_format",
            "nout",
            "selection",
            "variable_names",
            "julia_project",
        ]

    def __repr__(self):
        """Prints all current equations fitted by the model.

        The string `>>>>` denotes which equation is selected by the
        `model_selection`.
        """
        if self.equations is None:
            return "PySRRegressor.equations = None"

        output = "PySRRegressor.equations = [\n"

        equations = self.equations
        if not isinstance(equations, list):
            all_equations = [equations]
        else:
            all_equations = equations

        for i, equations in enumerate(all_equations):
            selected = ["" for _ in range(len(equations))]
            if self.model_selection == "accuracy":
                chosen_row = -1
            elif self.model_selection == "best":
                chosen_row = equations["score"].idxmax()
            else:
                raise NotImplementedError
            selected[chosen_row] = ">>>>"
            repr_equations = pd.DataFrame(
                dict(
                    pick=selected,
                    score=equations["score"],
                    equation=equations["equation"],
                    loss=equations["loss"],
                    complexity=equations["complexity"],
                )
            )

            if len(all_equations) > 1:
                output += "[\n"

            for line in repr_equations.__repr__().split("\n"):
                output += "\t" + line + "\n"

            if len(all_equations) > 1:
                output += "]"

            if i < len(all_equations) - 1:
                output += ", "

        output += "]"
        return output

    def set_params(self, **params):
        """Set parameters for equation search."""
        for key, value in params.items():
            if key in self.surface_parameters:
                self.__setattr__(key, value)
            else:
                self.params[key] = value

        return self

    def get_params(self, deep=True):
        """Get parameters for equation search."""
        del deep
        return {
            **self.params,
            **{key: self.__getattribute__(key) for key in self.surface_parameters},
        }

    def get_best(self):
        """Get best equation using `model_selection`."""
        if self.equations is None:
            raise ValueError("No equations have been generated yet.")
        if self.model_selection == "accuracy":
            if isinstance(self.equations, list):
                return [eq.iloc[-1] for eq in self.equations]
            return self.equations.iloc[-1]
        elif self.model_selection == "best":
            if isinstance(self.equations, list):
                return [eq.iloc[eq["score"].idxmax()] for eq in self.equations]
            return self.equations.iloc[self.equations["score"].idxmax()]
        else:
            raise NotImplementedError(
                f"{self.model_selection} is not a valid model selection strategy."
            )

    def fit(self, X, y, weights=None, variable_names=None):
        """Search for equations to fit the dataset and store them in `self.equations`.

        :param X: 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).
        :type X: np.ndarray/pandas.DataFrame
        :param y: 1D array (rows are examples) or 2D array (rows are examples, columns are outputs). Putting in a 2D array will trigger a search for equations for each feature of y.
        :type y: np.ndarray
        :param weights: Optional. Same shape as y. Each element is how to weight the mean-square-error loss for that particular element of y.
        :type weights: np.ndarray
        :param variable_names: a list of names for the variables, other than "x0", "x1", etc.
            You can also pass a pandas DataFrame for X.
        :type variable_names: list
        """
        if variable_names is None:
            variable_names = self.variable_names

        self._run(
            X=X,
            y=y,
            weights=weights,
            variable_names=variable_names,
        )

        return self

    def refresh(self):
        # Updates self.equations with any new options passed,
        # such as extra_sympy_mappings.
        self.equations = self.get_hof()

    def predict(self, X):
        """Predict y from input X using the equation chosen by `model_selection`.

        You may see what equation is used by printing this object. X should have the same
        columns as the training data.

        :param X: 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).
        :type X: np.ndarray/pandas.DataFrame
        :return: 1D array (rows are examples) or 2D array (rows are examples, columns are outputs).
        """
        self.refresh()
        best = self.get_best()
        if self.multioutput:
            return np.stack([eq["lambda_format"](X) for eq in best], axis=1)
        return best["lambda_format"](X)

    def sympy(self):
        """Return sympy representation of the equation(s) chosen by `model_selection`."""
        self.refresh()
        best = self.get_best()
        if self.multioutput:
            return [eq["sympy_format"] for eq in best]
        return best["sympy_format"]

    def latex(self):
        """Return latex representation of the equation(s) chosen by `model_selection`."""
        self.refresh()
        sympy_representation = self.sympy()
        if self.multioutput:
            return [sympy.latex(s) for s in sympy_representation]
        return sympy.latex(sympy_representation)

    def jax(self):
        """Return jax representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a dictionary
        containing {"callable": func, "parameters": params}. To call `func`, pass
        func(X, params). This function is differentiable using `jax.grad`.
        """
        if self.using_pandas:
            warnings.warn(
                "PySR's JAX modules are not set up to work with a "
                "model that was trained on pandas dataframes. "
                "Train on an array instead to ensure everything works as planned."
            )
        self.set_params(output_jax_format=True)
        self.refresh()
        best = self.get_best()
        if self.multioutput:
            return [eq["jax_format"] for eq in best]
        return best["jax_format"]

    def pytorch(self):
        """Return pytorch representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a PyTorch module
        containing the parameters as trainable attributes. You can use the module like
        any other PyTorch module: `module(X)`, where `X` is a tensor with the same
        column ordering as trained with.
        """
        if self.using_pandas:
            warnings.warn(
                "PySR's PyTorch modules are not set up to work with a "
                "model that was trained on pandas dataframes. "
                "Train on an array instead to ensure everything works as planned."
            )
        self.set_params(output_torch_format=True)
        self.refresh()
        best = self.get_best()
        if self.multioutput:
            return [eq["torch_format"] for eq in best]
        return best["torch_format"]

    def reset(self):
        """Reset the search state."""
        self.equations = None
        self.params_hash = None
        self.raw_julia_state = None
        self.variable_names = None
        self.selection = None

    def _run(self, X, y, weights, variable_names):
        global already_ran
        global Main

        for key in self.surface_parameters:
            if key in self.params:
                raise ValueError(
                    f"{key} is a surface parameter, and cannot be in self.params"
                )

        multithreading = self.params["multithreading"]
        procs = self.params["procs"]
        binary_operators = self.params["binary_operators"]
        unary_operators = self.params["unary_operators"]
        batching = self.params["batching"]
        maxsize = self.params["maxsize"]
        select_k_features = self.params["select_k_features"]
        Xresampled = self.params["Xresampled"]
        denoise = self.params["denoise"]
        constraints = self.params["constraints"]
        update = self.params["update"]
        loss = self.params["loss"]
        weightMutateConstant = self.params["weightMutateConstant"]
        weightMutateOperator = self.params["weightMutateOperator"]
        weightAddNode = self.params["weightAddNode"]
        weightInsertNode = self.params["weightInsertNode"]
        weightDeleteNode = self.params["weightDeleteNode"]
        weightSimplify = self.params["weightSimplify"]
        weightRandomize = self.params["weightRandomize"]
        weightDoNothing = self.params["weightDoNothing"]

        if Main is None:
            if multithreading:
                os.environ["JULIA_NUM_THREADS"] = str(procs)

            Main = init_julia()

        if isinstance(X, pd.DataFrame):
            if variable_names is not None:
                warnings.warn("Resetting variable_names from X.columns")

            variable_names = list(X.columns)
            X = np.array(X)
            self.using_pandas = True
        else:
            self.using_pandas = False

        if len(X.shape) == 1:
            X = X[:, None]

        assert not isinstance(y, pd.DataFrame)

        if variable_names is None or len(variable_names) == 0:
            variable_names = [f"x{i}" for i in range(X.shape[1])]

        use_custom_variable_names = len(variable_names) != 0
        # TODO: this is always true.

        _check_assertions(
            X,
            binary_operators,
            unary_operators,
            use_custom_variable_names,
            variable_names,
            weights,
            y,
        )

        self.n_features = X.shape[1]

        if len(X) > 10000 and not batching:
            warnings.warn(
                "Note: you are running with more than 10,000 datapoints. You should consider turning on batching (https://pysr.readthedocs.io/en/latest/docs/options/#batching). You should also reconsider if you need that many datapoints. Unless you have a large amount of noise (in which case you should smooth your dataset first), generally < 10,000 datapoints is enough to find a functional form with symbolic regression. More datapoints will lower the search speed."
            )

        X, selection = _handle_feature_selection(
            X, select_k_features, y, variable_names
        )

        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            self.multioutput = False
            self.nout = 1
            y = y.reshape(-1)
        elif len(y.shape) == 2:
            self.multioutput = True
            self.nout = y.shape[1]
        else:
            raise NotImplementedError("y shape not supported!")

        if denoise:
            if weights is not None:
                raise NotImplementedError(
                    "No weights for denoising - the weights are learned."
                )
            if Xresampled is not None:
                # Select among only the selected features:
                if isinstance(Xresampled, pd.DataFrame):
                    # Handle Xresampled is pandas dataframe
                    if selection is not None:
                        Xresampled = Xresampled[[variable_names[i] for i in selection]]
                    else:
                        Xresampled = Xresampled[variable_names]
                    Xresampled = np.array(Xresampled)
                else:
                    if selection is not None:
                        Xresampled = Xresampled[:, selection]
            if self.multioutput:
                y = np.stack(
                    [
                        _denoise(X, y[:, i], Xresampled=Xresampled)[1]
                        for i in range(self.nout)
                    ],
                    axis=1,
                )
                if Xresampled is not None:
                    X = Xresampled
            else:
                X, y = _denoise(X, y, Xresampled=Xresampled)

        self.julia_project, is_shared = _get_julia_project(self.julia_project)

        tmpdir = Path(tempfile.mkdtemp(dir=self.params["tempdir"]))

        if self.params["temp_equation_file"]:
            self.equation_file = tmpdir / "hall_of_fame.csv"
        elif self.equation_file is None:
            date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3]
            self.equation_file = "hall_of_fame_" + date_time + ".csv"

        _create_inline_operators(
            binary_operators=binary_operators, unary_operators=unary_operators
        )
        _handle_constraints(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            constraints=constraints,
        )

        una_constraints = [constraints[op] for op in unary_operators]
        bin_constraints = [constraints[op] for op in binary_operators]

        try:
            # TODO: is this needed since Julia now prints directly to stdout?
            term_width = shutil.get_terminal_size().columns
        except:
            _, term_width = subprocess.check_output(["stty", "size"]).split()

        if not already_ran:
            Main.eval("using Pkg")
            io = "devnull" if self.params["update_verbosity"] == 0 else "stderr"
            io_arg = f"io={io}" if is_julia_version_greater_eq(Main, "1.6") else ""

            Main.eval(
                f'Pkg.activate("{_escape_filename(self.julia_project)}", shared = Bool({int(is_shared)}), {io_arg})'
            )
            from julia.api import JuliaError

            if is_shared:
                # Install SymbolicRegression.jl:
                _add_sr_to_julia_project(Main, io_arg)

            try:
                if update:
                    Main.eval(f"Pkg.resolve({io_arg})")
                    Main.eval(f"Pkg.instantiate({io_arg})")
                else:
                    Main.eval(f"Pkg.instantiate({io_arg})")
            except (JuliaError, RuntimeError) as e:
                raise ImportError(import_error_string(self.julia_project)) from e
            Main.eval("using SymbolicRegression")

            Main.plus = Main.eval("(+)")
            Main.sub = Main.eval("(-)")
            Main.mult = Main.eval("(*)")
            Main.pow = Main.eval("(^)")
            Main.div = Main.eval("(/)")

        Main.custom_loss = Main.eval(loss)

        mutationWeights = [
            float(weightMutateConstant),
            float(weightMutateOperator),
            float(weightAddNode),
            float(weightInsertNode),
            float(weightDeleteNode),
            float(weightSimplify),
            float(weightRandomize),
            float(weightDoNothing),
        ]

        params_to_hash = {
            **{k: self.__getattribute__(k) for k in self.surface_parameters},
            **self.params,
        }
        params_excluded_from_hash = [
            "niterations",
        ]
        # Delete these^ from params_to_hash:
        params_to_hash = {
            k: v
            for k, v in params_to_hash.items()
            if k not in params_excluded_from_hash
        }

        # Sort params_to_hash by key:
        params_to_hash = OrderedDict(sorted(params_to_hash.items()))
        # Hash all parameters:
        cur_hash = sha256(str(params_to_hash).encode()).hexdigest()

        if self.params_hash is not None:
            if cur_hash != self.params_hash:
                warnings.warn(
                    "Warning: PySR options have changed since the last run. "
                    "This is experimental and may not work. "
                    "For example, if the operators change, or even their order,"
                    " the saved equations will be in the wrong format."
                    "\n\n"
                    "To reset the search state, run `.reset()`. "
                )

        self.params_hash = cur_hash

        options = Main.Options(
            binary_operators=Main.eval(str(tuple(binary_operators)).replace("'", "")),
            unary_operators=Main.eval(str(tuple(unary_operators)).replace("'", "")),
            bin_constraints=bin_constraints,
            una_constraints=una_constraints,
            loss=Main.custom_loss,
            maxsize=int(maxsize),
            hofFile=_escape_filename(self.equation_file),
            npopulations=int(self.params["populations"]),
            batching=batching,
            batchSize=int(
                min([self.params["batchSize"], len(X)]) if batching else len(X)
            ),
            mutationWeights=mutationWeights,
            terminal_width=int(term_width),
            probPickFirst=self.params["tournament_selection_p"],
            ns=self.params["tournament_selection_n"],
            # These have the same name:
            parsimony=self.params["parsimony"],
            alpha=self.params["alpha"],
            maxdepth=self.params["maxdepth"],
            fast_cycle=self.params["fast_cycle"],
            migration=self.params["migration"],
            hofMigration=self.params["hofMigration"],
            fractionReplacedHof=self.params["fractionReplacedHof"],
            shouldOptimizeConstants=self.params["shouldOptimizeConstants"],
            warmupMaxsizeBy=self.params["warmupMaxsizeBy"],
            useFrequency=self.params["useFrequency"],
            npop=self.params["npop"],
            ncyclesperiteration=self.params["ncyclesperiteration"],
            fractionReplaced=self.params["fractionReplaced"],
            topn=self.params["topn"],
            verbosity=self.params["verbosity"],
            optimizer_algorithm=self.params["optimizer_algorithm"],
            optimizer_nrestarts=self.params["optimizer_nrestarts"],
            optimize_probability=self.params["optimize_probability"],
            optimizer_iterations=self.params["optimizer_iterations"],
            perturbationFactor=self.params["perturbationFactor"],
            annealing=self.params["annealing"],
            stateReturn=True,  # Required for state saving.
            use_symbolic_utils=self.params["use_symbolic_utils"],
            progress=self.params["progress"],
            timeout_in_seconds=self.params["timeout_in_seconds"],
            crossoverProbability=self.params["crossoverProbability"],
        )

        np_dtype = {16: np.float16, 32: np.float32, 64: np.float64}[
            self.params["precision"]
        ]

        Main.X = np.array(X, dtype=np_dtype).T
        if len(y.shape) == 1:
            Main.y = np.array(y, dtype=np_dtype)
        else:
            Main.y = np.array(y, dtype=np_dtype).T
        if weights is not None:
            if len(weights.shape) == 1:
                Main.weights = np.array(weights, dtype=np_dtype)
            else:
                Main.weights = np.array(weights, dtype=np_dtype).T
        else:
            Main.weights = None

        cprocs = 0 if multithreading else procs

        self.raw_julia_state = Main.EquationSearch(
            Main.X,
            Main.y,
            weights=Main.weights,
            niterations=int(self.params["niterations"]),
            varMap=(
                variable_names
                if selection is None
                else [variable_names[i] for i in selection]
            ),
            options=options,
            numprocs=int(cprocs),
            multithreading=bool(multithreading),
            saved_state=self.raw_julia_state,
        )

        self.variable_names = variable_names
        self.selection = selection

        # Not in params:
        # selection, variable_names, multioutput

        self.equations = self.get_hof()

        if self.params["delete_tempfiles"]:
            shutil.rmtree(tmpdir)

        already_ran = True

    def get_hof(self):
        """Get the equations from a hall of fame file. If no arguments
        entered, the ones used previously from a call to PySR will be used."""

        try:
            if self.multioutput:
                all_outputs = []
                for i in range(1, self.nout + 1):
                    df = pd.read_csv(
                        str(self.equation_file) + f".out{i}" + ".bkup",
                        sep="|",
                    )
                    # Rename Complexity column to complexity:
                    df.rename(
                        columns={
                            "Complexity": "complexity",
                            "MSE": "loss",
                            "Equation": "equation",
                        },
                        inplace=True,
                    )

                    all_outputs.append(df)
            else:
                all_outputs = [pd.read_csv(str(self.equation_file) + ".bkup", sep="|")]
                all_outputs[-1].rename(
                    columns={
                        "Complexity": "complexity",
                        "MSE": "loss",
                        "Equation": "equation",
                    },
                    inplace=True,
                )
        except FileNotFoundError:
            raise RuntimeError(
                "Couldn't find equation file! The equation search likely exited before a single iteration completed."
            )

        ret_outputs = []

        for output in all_outputs:

            scores = []
            lastMSE = None
            lastComplexity = 0
            sympy_format = []
            lambda_format = []
            if self.output_jax_format:
                jax_format = []
            if self.output_torch_format:
                torch_format = []
            use_custom_variable_names = len(self.variable_names) != 0
            local_sympy_mappings = {
                **self.extra_sympy_mappings,
                **sympy_mappings,
            }

            if use_custom_variable_names:
                sympy_symbols = [
                    sympy.Symbol(self.variable_names[i]) for i in range(self.n_features)
                ]
            else:
                sympy_symbols = [
                    sympy.Symbol("x%d" % i) for i in range(self.n_features)
                ]

            for _, eqn_row in output.iterrows():
                eqn = sympify(eqn_row["equation"], locals=local_sympy_mappings)
                sympy_format.append(eqn)

                # Numpy:
                lambda_format.append(
                    CallableEquation(
                        sympy_symbols, eqn, self.selection, self.variable_names
                    )
                )

                # JAX:
                if self.output_jax_format:
                    from .export_jax import sympy2jax

                    func, params = sympy2jax(
                        eqn,
                        sympy_symbols,
                        selection=self.selection,
                        extra_jax_mappings=self.extra_jax_mappings,
                    )
                    jax_format.append({"callable": func, "parameters": params})

                # Torch:
                if self.output_torch_format:
                    from .export_torch import sympy2torch

                    module = sympy2torch(
                        eqn,
                        sympy_symbols,
                        selection=self.selection,
                        extra_torch_mappings=self.extra_torch_mappings,
                    )
                    torch_format.append(module)

                curMSE = eqn_row["loss"]
                curComplexity = eqn_row["complexity"]

                if lastMSE is None:
                    cur_score = 0.0
                else:
                    if curMSE > 0.0:
                        cur_score = -np.log(curMSE / lastMSE) / (
                            curComplexity - lastComplexity
                        )
                    else:
                        cur_score = np.inf

                scores.append(cur_score)
                lastMSE = curMSE
                lastComplexity = curComplexity

            output["score"] = np.array(scores)
            output["sympy_format"] = sympy_format
            output["lambda_format"] = lambda_format
            output_cols = [
                "complexity",
                "loss",
                "score",
                "equation",
                "sympy_format",
                "lambda_format",
            ]
            if self.output_jax_format:
                output_cols += ["jax_format"]
                output["jax_format"] = jax_format
            if self.output_torch_format:
                output_cols += ["torch_format"]
                output["torch_format"] = torch_format

            ret_outputs.append(output[output_cols])

        if self.multioutput:
            return ret_outputs
        return ret_outputs[0]

    def score(self, X, y):
        del X
        del y
        raise NotImplementedError
