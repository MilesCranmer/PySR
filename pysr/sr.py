import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple
import pathlib
import numpy as np
import pandas as pd
import sympy
from sympy import sympify, Symbol, lambdify
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import warnings

global_state = dict(
    equation_file="hall_of_fame.csv",
    n_features=None,
    variable_names=[],
    extra_sympy_mappings={},
    extra_torch_mappings={},
    extra_jax_mappings={},
    output_jax_format=False,
    output_torch_format=False,
    multioutput=False,
    nout=1,
    selection=None,
)

sympy_mappings = {
    "div": lambda x, y: x / y,
    "mult": lambda x, y: x * y,
    "sqrt_abs": lambda x: sympy.sqrt(abs(x)),
    "square": lambda x: x ** 2,
    "cube": lambda x: x ** 3,
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


def pysr(
    X,
    y,
    weights=None,
    binary_operators=None,
    unary_operators=None,
    procs=4,
    loss="L2DistLoss()",
    populations=20,
    niterations=100,
    ncyclesperiteration=300,
    alpha=0.1,
    annealing=False,
    fractionReplaced=0.10,
    fractionReplacedHof=0.10,
    npop=1000,
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
    weightSimplify=0.01,
    perturbationFactor=1.0,
    timeout=None,
    extra_sympy_mappings=None,
    extra_torch_mappings=None,
    extra_jax_mappings=None,
    equation_file=None,
    verbosity=1e9,
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
    julia_optimization=3,
    julia_project=None,
    user_input=True,
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
):
    """Run symbolic regression to fit f(X[i, :]) ~ y[i] for all i.
    Note: most default parameters have been tuned over several example
    equations, but you should adjust `niterations`,
    `binary_operators`, `unary_operators` to your requirements.
    You can view more detailed explanations of the options on the
    [options page](https://pysr.readthedocs.io/en/latest/docs/options/) of the documentation.

    :param X: 2D array. Rows are examples, columns are features. If pandas DataFrame, the columns are used for variable names (so make sure they don't contain spaces).
    :type X: np.ndarray/pandas.DataFrame
    :param y: 1D array (rows are examples) or 2D array (rows are examples, columns are outputs). Putting in a 2D array will trigger a search for equations for each feature of y.
    :type y: np.ndarray
    :param weights: same shape as y. Each element is how to weight the mean-square-error loss for that particular element of y.
    :type weights: np.ndarray
    :param binary_operators: List of strings giving the binary operators in Julia's Base. Default is ["+", "-", "*", "/",].
    :type binary_operators: list
    :param unary_operators: Same but for operators taking a single scalar. Default is [].
    :type unary_operators: list
    :param procs: Number of processes (=number of populations running).
    :type procs: int
    :param loss: String of Julia code specifying the loss function.  Can either be a loss from LossFunctions.jl, or your own loss written as a function. Examples of custom written losses include: `myloss(x, y) = abs(x-y)` for non-weighted, or `myloss(x, y, w) = w*abs(x-y)` for weighted.  Among the included losses, these are as follows. Regression: `LPDistLoss{P}()`, `L1DistLoss()`, `L2DistLoss()` (mean square), `LogitDistLoss()`, `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`, `PeriodicLoss(c)`, `QuantileLoss(τ)`.  Classification: `ZeroOneLoss()`, `PerceptronLoss()`, `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`, `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`, `SigmoidLoss()`, `DWDMarginLoss(q)`.
    :type loss: str
    :param populations: Number of populations running.
    :type populations: int
    :param niterations: Number of iterations of the algorithm to run. The best equations are printed, and migrate between populations, at the end of each.
    :type niterations: int
    :param ncyclesperiteration: Number of total mutations to run, per 10 samples of the population, per iteration.
    :type ncyclesperiteration: int
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
    :param timeout: Time in seconds to timeout search
    :type timeout: float
    :param equation_file: Where to save the files (.csv separated by |)
    :type equation_file: str
    :param verbosity: What verbosity level to use. 0 means minimal print statements.
    :type verbosity: int
    :param progress: Whether to use a progress bar instead of printing to stdout.
    :type progress: bool
    :param maxsize: Max size of an equation.
    :type maxsize: int
    :param maxdepth: Max depth of an equation. You can use both maxsize and maxdepth.  maxdepth is by default set to = maxsize, which means that it is redundant.
    :type maxdepth: int
    :param fast_cycle: (experimental) - batch over population subsamples. This is a slightly different algorithm than regularized evolution, but does cycles 15% faster. May be algorithmically less efficient.
    :type fast_cycle: bool
    :param variable_names: a list of names for the variables, other than "x0", "x1", etc.
    :type variable_names: list
    :param batching: whether to compare population members on small batches during evolution. Still uses full dataset for comparing against hall of fame.
    :type batching: bool
    :param batchSize: the amount of data to use if doing batching.
    :type batchSize: int
    :param select_k_features: whether to run feature selection in Python using random forests, before passing to the symbolic regression code. None means no feature selection; an int means select that many features.
    :type select_k_features: None/int
    :param warmupMaxsizeBy: whether to slowly increase max size from a small number up to the maxsize (if greater than 0).  If greater than 0, says the fraction of training time at which the current maxsize will reach the user-passed maxsize.
    :type warmupMaxsizeBy: float
    :param constraints: dictionary of int (unary) or 2-tuples (binary), this enforces maxsize constraints on the individual arguments of operators. E.g., `'pow': (-1, 1)` says that power laws can have any complexity left argument, but only 1 complexity exponent. Use this to force more interpretable solutions.
    :type constraints: dict
    :param useFrequency: whether to measure the frequency of complexities, and use that instead of parsimony to explore equation space. Will naturally find equations of all complexities.
    :type useFrequency: bool
    :param julia_optimization: Optimization level (0, 1, 2, 3)
    :type julia_optimization: int
    :param tempdir: directory for the temporary files
    :type tempdir: str/None
    :param delete_tempfiles: whether to delete the temporary files after finishing
    :type delete_tempfiles: bool
    :param julia_project: a Julia environment location containing a Project.toml (and potentially the source code for SymbolicRegression.jl).  Default gives the Python package directory, where a Project.toml file should be present from the install.
    :type julia_project: str/None
    :param user_input: Whether to ask for user input or not for installing (to be used for automated scripts). Will choose to install when asked.
    :type user_input: bool
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
    :param denoise: Whether to use a Gaussian Process to denoise the data before inputting to PySR. Can help PySR fit noisy data.
    :type denoise: bool
    :returns: Results dataframe, giving complexity, MSE, and equations (as strings), as well as functional forms. If list, each element corresponds to a dataframe of equations for each output.
    :type: pd.DataFrame/list
    """
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

    if progress is not None:
        if progress and ("buffer" not in sys.stdout.__dir__()):
            warnings.warn(
                "Note: it looks like you are running in Jupyter. The progress bar will be turned off."
            )
            progress = False
    else:
        if "buffer" in sys.stdout.__dir__():
            progress = True
        else:
            progress = False

    assert optimizer_algorithm in ["NelderMead", "BFGS"]
    assert tournament_selection_n < npop

    if isinstance(X, pd.DataFrame):
        variable_names = list(X.columns)
        X = np.array(X)

    if len(X.shape) == 1:
        X = X[:, None]

    if len(variable_names) == 0:
        variable_names = [f"x{i}" for i in range(X.shape[1])]

    if extra_jax_mappings is not None:
        for key, value in extra_jax_mappings:
            if not isinstance(value, str):
                raise NotImplementedError(
                    "extra_jax_mappings must have keys that are strings! e.g., {sympy.sqrt: 'jnp.sqrt'}."
                )

    if extra_torch_mappings is not None:
        for key, value in extra_jax_mappings:
            if not callable(value):
                raise NotImplementedError(
                    "extra_torch_mappings must be callable functions! e.g., {sympy.sqrt: torch.sqrt}."
                )

    use_custom_variable_names = len(variable_names) != 0

    _check_assertions(
        X,
        binary_operators,
        unary_operators,
        use_custom_variable_names,
        variable_names,
        weights,
        y,
    )
    _check_for_julia_installation()

    if len(X) > 10000 and not batching:
        warnings.warn(
            "Note: you are running with more than 10,000 datapoints. You should consider turning on batching (https://pysr.readthedocs.io/en/latest/docs/options/#batching). You should also reconsider if you need that many datapoints. Unless you have a large amount of noise (in which case you should smooth your dataset first), generally < 10,000 datapoints is enough to find a functional form with symbolic regression. More datapoints will lower the search speed."
        )

    if maxsize > 40:
        warnings.warn(
            "Note: Using a large maxsize for the equation search will be exponentially slower and use significant memory. You should consider turning `useFrequency` to False, and perhaps use `warmupMaxsizeBy`."
        )
    if maxsize < 7:
        raise NotImplementedError("PySR requires a maxsize of at least 7")

    X, variable_names, selection = _handle_feature_selection(
        X, select_k_features, use_custom_variable_names, variable_names, y
    )

    if maxdepth is None:
        maxdepth = maxsize
    if isinstance(binary_operators, str):
        binary_operators = [binary_operators]
    if isinstance(unary_operators, str):
        unary_operators = [unary_operators]

    if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
        multioutput = False
        nout = 1
        y = y.reshape(-1)
    elif len(y.shape) == 2:
        multioutput = True
        nout = y.shape[1]
    else:
        raise NotImplementedError("y shape not supported!")

    if denoise:
        if weights is not None:
            raise NotImplementedError(
                "No weights for denoising - the weights are learned."
            )
        if Xresampled is not None and selection is not None:
            # Select among only the selected features:
            Xresampled = Xresampled[:, selection]
        if multioutput:
            y = np.stack(
                [_denoise(X, y[:, i], Xresampled=Xresampled)[1] for i in range(nout)],
                axis=1,
            )
            if Xresampled is not None:
                X = Xresampled
        else:
            X, y = _denoise(X, y, Xresampled=Xresampled)

    kwargs = dict(
        X=X,
        y=y,
        weights=weights,
        alpha=alpha,
        annealing=annealing,
        batchSize=batchSize,
        batching=batching,
        binary_operators=binary_operators,
        fast_cycle=fast_cycle,
        fractionReplaced=fractionReplaced,
        ncyclesperiteration=ncyclesperiteration,
        niterations=niterations,
        npop=npop,
        topn=topn,
        verbosity=verbosity,
        progress=progress,
        update=update,
        julia_optimization=julia_optimization,
        timeout=timeout,
        fractionReplacedHof=fractionReplacedHof,
        hofMigration=hofMigration,
        maxdepth=maxdepth,
        maxsize=maxsize,
        migration=migration,
        optimizer_algorithm=optimizer_algorithm,
        optimizer_nrestarts=optimizer_nrestarts,
        optimize_probability=optimize_probability,
        optimizer_iterations=optimizer_iterations,
        parsimony=parsimony,
        perturbationFactor=perturbationFactor,
        populations=populations,
        procs=procs,
        shouldOptimizeConstants=shouldOptimizeConstants,
        unary_operators=unary_operators,
        useFrequency=useFrequency,
        use_custom_variable_names=use_custom_variable_names,
        variable_names=variable_names,
        warmupMaxsizeBy=warmupMaxsizeBy,
        weightAddNode=weightAddNode,
        weightDeleteNode=weightDeleteNode,
        weightDoNothing=weightDoNothing,
        weightInsertNode=weightInsertNode,
        weightMutateConstant=weightMutateConstant,
        weightMutateOperator=weightMutateOperator,
        weightRandomize=weightRandomize,
        weightSimplify=weightSimplify,
        constraints=constraints,
        extra_sympy_mappings=extra_sympy_mappings,
        extra_jax_mappings=extra_jax_mappings,
        extra_torch_mappings=extra_torch_mappings,
        julia_project=julia_project,
        loss=loss,
        output_jax_format=output_jax_format,
        output_torch_format=output_torch_format,
        selection=selection,
        multioutput=multioutput,
        nout=nout,
        tournament_selection_n=tournament_selection_n,
        tournament_selection_p=tournament_selection_p,
        denoise=denoise,
    )

    kwargs = {**_set_paths(tempdir), **kwargs}

    if temp_equation_file:
        equation_file = kwargs["tmpdir"] / "hall_of_fame.csv"
    elif equation_file is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3]
        equation_file = "hall_of_fame_" + date_time + ".csv"

    kwargs = {**dict(equation_file=equation_file), **kwargs}

    pkg_directory = kwargs["pkg_directory"]
    if kwargs["julia_project"] is not None:
        manifest_filepath = Path(kwargs["julia_project"]) / "Manifest.toml"
    else:
        manifest_filepath = pkg_directory / "Manifest.toml"

    kwargs["need_install"] = False

    if not (manifest_filepath).is_file():
        kwargs["need_install"] = (not user_input) or _yesno(
            "I will install Julia packages using PySR's Project.toml file. OK?"
        )
        if kwargs["need_install"]:
            print("OK. I will install at launch.")
            assert update

    kwargs["def_hyperparams"] = _create_inline_operators(**kwargs)

    _handle_constraints(**kwargs)

    kwargs["constraints_str"] = _make_constraints_str(**kwargs)
    kwargs["def_hyperparams"] = _make_hyperparams_julia_str(**kwargs)
    kwargs["def_datasets"] = _make_datasets_julia_str(**kwargs)

    _create_julia_files(**kwargs)
    _final_pysr_process(**kwargs)
    _set_globals(**kwargs)

    equations = get_hof(**kwargs)

    if delete_tempfiles:
        shutil.rmtree(kwargs["tmpdir"])

    return equations


def _set_globals(X, **kwargs):
    global global_state

    global_state["n_features"] = X.shape[1]
    for key, value in kwargs.items():
        if key in global_state:
            global_state[key] = value


def _final_pysr_process(julia_optimization, runfile_filename, timeout, **kwargs):
    command = [
        "julia",
        f"-O{julia_optimization:d}",
        str(runfile_filename),
    ]
    if timeout is not None:
        command = ["timeout", f"{timeout}"] + command
    _cmd_runner(command, **kwargs)


def _cmd_runner(command, progress, **kwargs):
    if kwargs["verbosity"] > 0:
        print("Running on", " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=-1)
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode("utf-8")
            if progress:
                decoded_line = (
                    decoded_line.replace("\\033[K", "\033[K")
                    .replace("\\033[1A", "\033[1A")
                    .replace("\\033[1B", "\033[1B")
                    .replace("\\r", "\r")
                    .encode(sys.stdout.encoding, errors="replace")
                )
                sys.stdout.buffer.write(decoded_line)
                sys.stdout.flush()
            else:
                print(decoded_line, end="")

        process.stdout.close()
        process.wait()
    except KeyboardInterrupt:
        print("Killing process... will return when done.")
        process.kill()


def _create_julia_files(
    dataset_filename,
    def_datasets,
    hyperparam_filename,
    def_hyperparams,
    niterations,
    runfile_filename,
    julia_project,
    procs,
    weights,
    X,
    variable_names,
    pkg_directory,
    need_install,
    update,
    **kwargs,
):
    with open(hyperparam_filename, "w") as f:
        print(def_hyperparams, file=f)
    with open(dataset_filename, "w") as f:
        print(def_datasets, file=f)
    with open(runfile_filename, "w") as f:
        if julia_project is None:
            julia_project = pkg_directory
        else:
            julia_project = Path(julia_project)
        print(f"import Pkg", file=f)
        print(f'Pkg.activate("{_escape_filename(julia_project)}")', file=f)
        if need_install:
            print(f"Pkg.instantiate()", file=f)
            print("Pkg.update()", file=f)
            print("Pkg.precompile()", file=f)
        elif update:
            print(f"Pkg.update()", file=f)
        print(f"using SymbolicRegression", file=f)
        print(f'include("{_escape_filename(hyperparam_filename)}")', file=f)
        print(f'include("{_escape_filename(dataset_filename)}")', file=f)
        if len(variable_names) == 0:
            varMap = "[" + ",".join([f'"x{i}"' for i in range(X.shape[1])]) + "]"
        else:
            varMap = (
                "[" + ",".join(['"' + vname + '"' for vname in variable_names]) + "]"
            )

        if weights is not None:
            print(
                f"EquationSearch(X, y, weights=weights, niterations={niterations:d}, varMap={varMap}, options=options, numprocs={procs})",
                file=f,
            )
        else:
            print(
                f"EquationSearch(X, y, niterations={niterations:d}, varMap={varMap}, options=options, numprocs={procs})",
                file=f,
            )


def _make_datasets_julia_str(
    X, X_filename, weights, weights_filename, y, y_filename, multioutput, **kwargs
):
    def_datasets = """using DelimitedFiles"""
    np.savetxt(X_filename, X.astype(np.float32), delimiter=",")
    if multioutput:
        np.savetxt(y_filename, y.astype(np.float32), delimiter=",")
    else:
        np.savetxt(y_filename, y.reshape(-1, 1).astype(np.float32), delimiter=",")
    if weights is not None:
        if multioutput:
            np.savetxt(weights_filename, weights.astype(np.float32), delimiter=",")
        else:
            np.savetxt(
                weights_filename,
                weights.reshape(-1, 1).astype(np.float32),
                delimiter=",",
            )
    def_datasets += f"""
X = copy(transpose(readdlm("{_escape_filename(X_filename)}", ',', Float32, '\\n')))"""

    if multioutput:
        def_datasets += f"""
y = copy(transpose(readdlm("{_escape_filename(y_filename)}", ',', Float32, '\\n')))"""
    else:
        def_datasets += f"""
y = readdlm("{_escape_filename(y_filename)}", ',', Float32, '\\n')[:, 1]"""

    if weights is not None:
        if multioutput:
            def_datasets += f"""
weights = copy(transpose(readdlm("{_escape_filename(weights_filename)}", ',', Float32, '\\n')))"""
        else:
            def_datasets += f"""
weights = readdlm("{_escape_filename(weights_filename)}", ',', Float32, '\\n')[:, 1]"""
    return def_datasets


def _make_hyperparams_julia_str(
    X,
    alpha,
    annealing,
    batchSize,
    batching,
    binary_operators,
    constraints_str,
    def_hyperparams,
    equation_file,
    fast_cycle,
    fractionReplacedHof,
    hofMigration,
    maxdepth,
    maxsize,
    migration,
    optimizer_algorithm,
    optimizer_nrestarts,
    optimize_probability,
    optimizer_iterations,
    npop,
    parsimony,
    perturbationFactor,
    populations,
    shouldOptimizeConstants,
    unary_operators,
    useFrequency,
    warmupMaxsizeBy,
    weightAddNode,
    ncyclesperiteration,
    fractionReplaced,
    topn,
    verbosity,
    progress,
    loss,
    weightDeleteNode,
    weightDoNothing,
    weightInsertNode,
    weightMutateConstant,
    weightMutateOperator,
    weightRandomize,
    weightSimplify,
    tournament_selection_n,
    tournament_selection_p,
    **kwargs,
):
    try:
        term_width = shutil.get_terminal_size().columns
    except:
        _, term_width = subprocess.check_output(["stty", "size"]).split()

    def tuple_fix(ops):
        if len(ops) > 1:
            return ", ".join(ops)
        if len(ops) == 0:
            return ""
        return ops[0] + ","

    def_hyperparams += f"""\n
plus=(+)
sub=(-)
mult=(*)
square=SymbolicRegression.square
cube=SymbolicRegression.cube
pow=(^)
div=(/)
log_abs=SymbolicRegression.log_abs
log2_abs=SymbolicRegression.log2_abs
log10_abs=SymbolicRegression.log10_abs
log1p_abs=SymbolicRegression.log1p_abs
acosh_abs=SymbolicRegression.acosh_abs
atanh_clip=SymbolicRegression.atanh_clip
sqrt_abs=SymbolicRegression.sqrt_abs
neg=SymbolicRegression.neg
greater=SymbolicRegression.greater
relu=SymbolicRegression.relu
logical_or=SymbolicRegression.logical_or
logical_and=SymbolicRegression.logical_and
_custom_loss = {loss}

options = SymbolicRegression.Options(binary_operators={'(' + tuple_fix(binary_operators) + ')'},
unary_operators={'(' + tuple_fix(unary_operators) + ')'},
{constraints_str}
parsimony={parsimony:f}f0,
loss=_custom_loss,
alpha={alpha:f}f0,
maxsize={maxsize:d},
maxdepth={maxdepth:d},
fast_cycle={'true' if fast_cycle else 'false'},
migration={'true' if migration else 'false'},
hofMigration={'true' if hofMigration else 'false'},
fractionReplacedHof={fractionReplacedHof}f0,
shouldOptimizeConstants={'true' if shouldOptimizeConstants else 'false'},
hofFile="{_escape_filename(equation_file)}",
npopulations={populations:d},
optimizer_algorithm="{optimizer_algorithm}",
optimizer_nrestarts={optimizer_nrestarts:d},
optimize_probability={optimize_probability:f}f0,
optimizer_iterations={optimizer_iterations:d},
perturbationFactor={perturbationFactor:f}f0,
annealing={"true" if annealing else "false"},
batching={"true" if batching else "false"},
batchSize={min([batchSize, len(X)]) if batching else len(X):d},
mutationWeights=[
    {weightMutateConstant:f},
    {weightMutateOperator:f},
    {weightAddNode:f},
    {weightInsertNode:f},
    {weightDeleteNode:f},
    {weightSimplify:f},
    {weightRandomize:f},
    {weightDoNothing:f}
],
warmupMaxsizeBy={warmupMaxsizeBy:f}f0,
useFrequency={"true" if useFrequency else "false"},
npop={npop:d},
ns={tournament_selection_n:d},
probPickFirst={tournament_selection_p:f}f0,
ncyclesperiteration={ncyclesperiteration:d},
fractionReplaced={fractionReplaced:f}f0,
topn={topn:d},
verbosity=round(Int32, {verbosity:f}),
progress={'true' if progress else 'false'},
terminal_width={term_width:d}
"""

    def_hyperparams += "\n)"
    return def_hyperparams


def _make_constraints_str(binary_operators, constraints, unary_operators, **kwargs):
    constraints_str = "una_constraints = ["
    first = True
    for op in unary_operators:
        val = constraints[op]
        if not first:
            constraints_str += ", "
        constraints_str += f"{val:d}"
        first = False
    constraints_str += """],
bin_constraints = ["""
    first = True
    for op in binary_operators:
        tup = constraints[op]
        if not first:
            constraints_str += ", "
        constraints_str += f"({tup[0]:d}, {tup[1]:d})"
        first = False
    constraints_str += "],"
    return constraints_str


def _handle_constraints(binary_operators, constraints, unary_operators, **kwargs):
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


def _create_inline_operators(binary_operators, unary_operators, **kwargs):
    def_hyperparams = ""
    for op_list in [binary_operators, unary_operators]:
        for i in range(len(op_list)):
            op = op_list[i]
            is_user_defined_operator = "(" in op

            if is_user_defined_operator:
                def_hyperparams += op + "\n"
                # Cut off from the first non-alphanumeric char:
                first_non_char = [
                    j
                    for j in range(len(op))
                    if not (op[j].isalpha() or op[j].isdigit())
                ][0]
                function_name = op[:first_non_char]
                op_list[i] = function_name
    return def_hyperparams


def _handle_feature_selection(
    X, select_k_features, use_custom_variable_names, variable_names, y
):
    if select_k_features is not None:
        selection = run_feature_selection(X, y, select_k_features)
        print(f"Using features {selection}")
        X = X[:, selection]

        if use_custom_variable_names:
            variable_names = [
                variable_names[selection[i]] for i in range(len(selection))
            ]
    else:
        selection = None
    return X, variable_names, selection


def _set_paths(tempdir):
    # System-independent paths
    pkg_directory = Path(__file__).parents[1]
    default_project_file = pkg_directory / "Project.toml"
    tmpdir = Path(tempfile.mkdtemp(dir=tempdir))
    hyperparam_filename = tmpdir / f"hyperparams.jl"
    dataset_filename = tmpdir / f"dataset.jl"
    runfile_filename = tmpdir / "runfile.jl"
    X_filename = tmpdir / "X.csv"
    y_filename = tmpdir / "y.csv"
    weights_filename = tmpdir / "weights.csv"
    return dict(
        pkg_directory=pkg_directory,
        default_project_file=default_project_file,
        X_filename=X_filename,
        dataset_filename=dataset_filename,
        hyperparam_filename=hyperparam_filename,
        runfile_filename=runfile_filename,
        tmpdir=tmpdir,
        weights_filename=weights_filename,
        y_filename=y_filename,
    )


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


def _check_for_julia_installation():
    try:
        process = subprocess.Popen(["julia", "-v"], stdout=subprocess.PIPE, bufsize=-1)
        while True:
            line = process.stdout.readline()
            if not line:
                break
        process.stdout.close()
        process.wait()
    except FileNotFoundError:

        raise RuntimeError(
            f"Your current $PATH is: {os.environ['PATH']}\nPySR could not start julia. Make sure julia is installed and on your $PATH."
        )
    process.kill()


def run_feature_selection(X, y, select_k_features):
    """Use a gradient boosting tree regressor as a proxy for finding
    the k most important features in X, returning indices for those
    features as output."""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel, SelectKBest

    clf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(X, y)
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    return selector.get_support(indices=True)


def get_hof(
    equation_file=None,
    n_features=None,
    variable_names=None,
    output_jax_format=None,
    output_torch_format=None,
    selection=None,
    extra_sympy_mappings=None,
    extra_jax_mappings=None,
    extra_torch_mappings=None,
    multioutput=None,
    nout=None,
    **kwargs,
):
    """Get the equations from a hall of fame file. If no arguments
    entered, the ones used previously from a call to PySR will be used."""

    global global_state

    if equation_file is None:
        equation_file = global_state["equation_file"]
    if n_features is None:
        n_features = global_state["n_features"]
    if variable_names is None:
        variable_names = global_state["variable_names"]
    if extra_sympy_mappings is None:
        extra_sympy_mappings = global_state["extra_sympy_mappings"]
    if extra_jax_mappings is None:
        extra_jax_mappings = global_state["extra_jax_mappings"]
    if extra_torch_mappings is None:
        extra_torch_mappings = global_state["extra_torch_mappings"]
    if output_torch_format is None:
        output_torch_format = global_state["output_torch_format"]
    if output_jax_format is None:
        output_jax_format = global_state["output_jax_format"]
    if multioutput is None:
        multioutput = global_state["multioutput"]
    if nout is None:
        nout = global_state["nout"]
    if selection is None:
        selection = global_state["selection"]

    global_state["selection"] = selection
    global_state["equation_file"] = equation_file
    global_state["n_features"] = n_features
    global_state["variable_names"] = variable_names
    global_state["extra_sympy_mappings"] = extra_sympy_mappings
    global_state["extra_jax_mappings"] = extra_jax_mappings
    global_state["extra_torch_mappings"] = extra_torch_mappings
    global_state["output_torch_format"] = output_torch_format
    global_state["output_jax_format"] = output_jax_format
    global_state["multioutput"] = multioutput
    global_state["nout"] = nout
    global_state["selection"] = selection

    try:
        if multioutput:
            all_outputs = [
                pd.read_csv(str(equation_file) + f".out{i}" + ".bkup", sep="|")
                for i in range(1, nout + 1)
            ]
        else:
            all_outputs = [pd.read_csv(str(equation_file) + ".bkup", sep="|")]
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
        if output_jax_format:
            jax_format = []
        if output_torch_format:
            torch_format = []
        use_custom_variable_names = len(variable_names) != 0
        local_sympy_mappings = {**extra_sympy_mappings, **sympy_mappings}

        if use_custom_variable_names:
            sympy_symbols = [sympy.Symbol(variable_names[i]) for i in range(n_features)]
        else:
            sympy_symbols = [sympy.Symbol("x%d" % i) for i in range(n_features)]

        for i in range(len(output)):
            eqn = sympify(output.loc[i, "Equation"], locals=local_sympy_mappings)
            sympy_format.append(eqn)

            # Numpy:
            lambda_format.append(CallableEquation(sympy_symbols, eqn, selection))

            # JAX:
            if output_jax_format:
                from .export_jax import sympy2jax

                func, params = sympy2jax(
                    eqn,
                    sympy_symbols,
                    selection=selection,
                    extra_jax_mappings=extra_jax_mappings,
                )
                jax_format.append({"callable": func, "parameters": params})

            # Torch:
            if output_torch_format:
                from .export_torch import sympy2torch

                module = sympy2torch(
                    eqn,
                    sympy_symbols,
                    selection=selection,
                    extra_torch_mappings=extra_torch_mappings,
                )
                torch_format.append(module)

            curMSE = output.loc[i, "MSE"]
            curComplexity = output.loc[i, "Complexity"]

            if lastMSE is None:
                cur_score = 0.0
            else:
                cur_score = -np.log(curMSE / lastMSE) / (curComplexity - lastComplexity)

            scores.append(cur_score)
            lastMSE = curMSE
            lastComplexity = curComplexity

        output["score"] = np.array(scores)
        output["sympy_format"] = sympy_format
        output["lambda_format"] = lambda_format
        output_cols = [
            "Complexity",
            "MSE",
            "score",
            "Equation",
            "sympy_format",
            "lambda_format",
        ]
        if output_jax_format:
            output_cols += ["jax_format"]
            output["jax_format"] = jax_format
        if output_torch_format:
            output_cols += ["torch_format"]
            output["torch_format"] = torch_format

        ret_outputs.append(output[output_cols])

    if multioutput:
        return ret_outputs
    return ret_outputs[0]


def best_row(equations=None):
    """Return the best row of a hall of fame file using the score column.
    By default this uses the last equation file.
    """
    if equations is None:
        equations = get_hof()
    if isinstance(equations, list):
        return [eq.iloc[np.argmax(eq["score"])] for eq in equations]
    return equations.iloc[np.argmax(equations["score"])]


def best_tex(equations=None):
    """Return the equation with the best score, in latex format
    By default this uses the last equation file.
    """
    if equations is None:
        equations = get_hof()
    if isinstance(equations, list):
        return [
            sympy.latex(best_row(eq)["sympy_format"].simplify()) for eq in equations
        ]
    return sympy.latex(best_row(equations)["sympy_format"].simplify())


def best(equations=None):
    """Return the equation with the best score, in sympy format.
    By default this uses the last equation file.
    """
    if equations is None:
        equations = get_hof()
    if isinstance(equations, list):
        return [best_row(eq)["sympy_format"].simplify() for eq in equations]
    return best_row(equations)["sympy_format"].simplify()


def best_callable(equations=None):
    """Return the equation with the best score, in callable format.
    By default this uses the last equation file.
    """
    if equations is None:
        equations = get_hof()
    if isinstance(equations, list):
        return [best_row(eq)["lambda_format"] for eq in equations]
    return best_row(equations)["lambda_format"]


def _escape_filename(filename):
    """Turns a file into a string representation with correctly escaped backslashes"""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


# https://gist.github.com/garrettdreyfus/8153571
def _yesno(question):
    """Simple Yes/No Function."""
    prompt = f"{question} (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return _yesno(question)
    if ans == "y":
        return True
    return False


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


class CallableEquation(object):
    """Simple wrapper for numpy lambda functions built with sympy"""

    def __init__(self, sympy_symbols, eqn, selection=None):
        self._sympy = eqn
        self._sympy_symbols = sympy_symbols
        self._selection = selection
        self._lambda = lambdify(sympy_symbols, eqn)

    def __repr__(self):
        return f"PySRFunction(X=>{self._sympy})"

    def __call__(self, X):
        if self._selection is not None:
            return self._lambda(*X[:, self._selection].T)
        return self._lambda(*X.T)
