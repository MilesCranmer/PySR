import os
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
from .export import sympy2jax

global_equation_file = 'hall_of_fame.csv'
global_n_features = None
global_variable_names = []
global_extra_sympy_mappings = {}

sympy_mappings = {
    'div':  lambda x, y : x/y,
    'mult': lambda x, y : x*y,
    'sqrtm':lambda x    : sympy.sqrt(abs(x)),
    'square':lambda x   : x**2,
    'cube': lambda x    : x**3,
    'plus': lambda x, y : x + y,
    'sub':  lambda x, y : x - y,
    'neg':  lambda x    : -x,
    'pow':  lambda x, y : sympy.sign(x)*abs(x)**y,
    'cos':  lambda x    : sympy.cos(x),
    'sin':  lambda x    : sympy.sin(x),
    'tan':  lambda x    : sympy.tan(x),
    'cosh': lambda x    : sympy.cosh(x),
    'sinh': lambda x    : sympy.sinh(x),
    'tanh': lambda x    : sympy.tanh(x),
    'exp':  lambda x    : sympy.exp(x),
    'acos': lambda x    : sympy.acos(x),
    'asin': lambda x    : sympy.asin(x),
    'atan': lambda x    : sympy.atan(x),
    'acosh':lambda x    : sympy.acosh(x),
    'asinh':lambda x    : sympy.asinh(x),
    'atanh':lambda x    : sympy.atanh(x),
    'abs':  lambda x    : abs(x),
    'mod':  lambda x, y : sympy.Mod(x, y),
    'erf':  lambda x    : sympy.erf(x),
    'erfc': lambda x    : sympy.erfc(x),
    'logm': lambda x    : sympy.log(abs(x)),
    'logm10':lambda x    : sympy.log(abs(x), 10),
    'logm2': lambda x    : sympy.log(abs(x), 2),
    'log1p': lambda x    : sympy.log(x + 1),
    'floor': lambda x    : sympy.floor(x),
    'ceil': lambda x    : sympy.ceil(x),
    'sign': lambda x    : sympy.sign(x),
}

def pysr(X=None, y=None, weights=None,
            binary_operators=["plus", "mult"],
            unary_operators=["cos", "exp", "sin"],
            procs=4,
            loss='L2DistLoss()',
            populations=None,
            niterations=100,
            ncyclesperiteration=300,
            alpha=0.1,
            annealing=True,
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
            extra_sympy_mappings={},
            equation_file=None,
            test='simple1',
            verbosity=1e9,
            progress=False,
            maxsize=20,
            fast_cycle=False,
            maxdepth=None,
            variable_names=[],
            batching=False,
            batchSize=50,
            select_k_features=None,
            warmupMaxsizeBy=0.0,
            constraints={},
            useFrequency=False,
            tempdir=None,
            delete_tempfiles=True,
            julia_optimization=3,
            julia_project=None,
            user_input=True,
            update=True,
            temp_equation_file=False,
            output_jax_format=False,
            warmupMaxsize=None, #Deprecated
            nrestarts=None,
            optimizer_algorithm="NelderMead",
            optimizer_nrestarts=3,
            optimize_probability=0.1,
            optimizer_iterations=100,
        ):
    """Run symbolic regression to fit f(X[i, :]) ~ y[i] for all i.
    Note: most default parameters have been tuned over several example
    equations, but you should adjust `niterations`,
    `binary_operators`, `unary_operators` to your requirements.

    :param X: np.ndarray or pandas.DataFrame, 2D array. Rows are examples,
        columns are features. If pandas DataFrame, the columns are used
        for variable names (so make sure they don't contain spaces).
    :param y: np.ndarray, 1D array. Rows are examples.
    :param weights: np.ndarray, 1D array. Each row is how to weight the
        mean-square-error loss on weights.
    :param binary_operators: list, List of strings giving the binary operators
        in Julia's Base.
    :param unary_operators: list, Same but for operators taking a single scalar.
    :param procs: int, Number of processes (=number of populations running).
    :param loss: str, String of Julia code specifying the loss function.
        Can either be a loss from LossFunctions.jl, or your own
        loss written as a function. Examples of custom written losses
        include: `myloss(x, y) = abs(x-y)` for non-weighted, or 
        `myloss(x, y, w) = w*abs(x-y)` for weighted.
        Among the included losses, these are as follows. Regression:
        `LPDistLoss{P}()`, `L1DistLoss()`, `L2DistLoss()` (mean square),
        `LogitDistLoss()`, `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`,
        `L2EpsilonInsLoss(ϵ)`, `PeriodicLoss(c)`, `QuantileLoss(τ)`.
        Classification: `ZeroOneLoss()`, `PerceptronLoss()`, `L1HingeLoss()`,
        `SmoothedL1HingeLoss(γ)`, `ModifiedHuberLoss()`, `L2MarginLoss()`,
        `ExpLoss()`, `SigmoidLoss()`, `DWDMarginLoss(q)`.
    :param populations: int, Number of populations running; by default=procs.
    :param niterations: int, Number of iterations of the algorithm to run. The best
        equations are printed, and migrate between populations, at the
        end of each.
    :param ncyclesperiteration: int, Number of total mutations to run, per 10
        samples of the population, per iteration.
    :param alpha: float, Initial temperature.
    :param annealing: bool, Whether to use annealing. You should (and it is default).
    :param fractionReplaced: float, How much of population to replace with migrating
        equations from other populations.
    :param fractionReplacedHof: float, How much of population to replace with migrating
        equations from hall of fame.
    :param npop: int, Number of individuals in each population
    :param parsimony: float, Multiplicative factor for how much to punish complexity.
    :param migration: bool, Whether to migrate.
    :param hofMigration: bool, Whether to have the hall of fame migrate.
    :param shouldOptimizeConstants: bool, Whether to numerically optimize
        constants (Nelder-Mead/Newton) at the end of each iteration.
    :param topn: int, How many top individuals migrate from each population.
    :param nrestarts: int, Number of times to restart the constant optimizer
    :param perturbationFactor: float, Constants are perturbed by a max
        factor of (perturbationFactor*T + 1). Either multiplied by this
        or divided by this.
    :param weightAddNode: float, Relative likelihood for mutation to add a node
    :param weightInsertNode: float, Relative likelihood for mutation to insert a node
    :param weightDeleteNode: float, Relative likelihood for mutation to delete a node
    :param weightDoNothing: float, Relative likelihood for mutation to leave the individual
    :param weightMutateConstant: float, Relative likelihood for mutation to change
        the constant slightly in a random direction.
    :param weightMutateOperator: float, Relative likelihood for mutation to swap
        an operator.
    :param weightRandomize: float, Relative likelihood for mutation to completely
        delete and then randomly generate the equation
    :param weightSimplify: float, Relative likelihood for mutation to simplify
        constant parts by evaluation
    :param timeout: float, Time in seconds to timeout search
    :param equation_file: str, Where to save the files (.csv separated by |)
    :param test: str, What test to run, if X,y not passed.
    :param verbosity: int, What verbosity level to use. 0 means minimal print statements.
    :param progress: bool, Whether to use a progress bar instead of printing to stdout.
    :param maxsize: int, Max size of an equation.
    :param maxdepth: int, Max depth of an equation. You can use both maxsize and maxdepth.
        maxdepth is by default set to = maxsize, which means that it is redundant.
    :param fast_cycle: bool, (experimental) - batch over population subsamples. This
        is a slightly different algorithm than regularized evolution, but does cycles
        15% faster. May be algorithmically less efficient.
    :param variable_names: list, a list of names for the variables, other
        than "x0", "x1", etc.
    :param batching: bool, whether to compare population members on small batches
        during evolution. Still uses full dataset for comparing against
        hall of fame.
    :param batchSize: int, the amount of data to use if doing batching.
    :param select_k_features: (None, int), whether to run feature selection in
        Python using random forests, before passing to the symbolic regression
        code. None means no feature selection; an int means select that many
        features.
    :param warmupMaxsizeBy: float, whether to slowly increase max size from
        a small number up to the maxsize (if greater than 0).
        If greater than 0, says the fraction of training time at which
        the current maxsize will reach the user-passed maxsize.
    :param constraints: dict of int (unary) or 2-tuples (binary),
        this enforces maxsize constraints on the individual
        arguments of operators. E.g., `'pow': (-1, 1)`
        says that power laws can have any complexity left argument, but only
        1 complexity exponent. Use this to force more interpretable solutions.
    :param useFrequency: bool, whether to measure the frequency of complexities,
        and use that instead of parsimony to explore equation space. Will
        naturally find equations of all complexities.
    :param julia_optimization: int, Optimization level (0, 1, 2, 3)
    :param tempdir: str or None, directory for the temporary files
    :param delete_tempfiles: bool, whether to delete the temporary files after finishing
    :param julia_project: str or None, a Julia environment location containing
        a Project.toml (and potentially the source code for SymbolicRegression.jl).
        Default gives the Python package directory, where a Project.toml file
        should be present from the install.
    :param user_input: Whether to ask for user input or not for installing (to
        be used for automated scripts). Will choose to install when asked.
    :param update: Whether to automatically update Julia packages.
    :param temp_equation_file: Whether to put the hall of fame file in
        the temp directory. Deletion is then controlled with the
        delete_tempfiles argument.
    :param output_jax_format: Whether to create a 'jax_format' column in the output,
        containing jax-callable functions and the default parameters in a jax array.
    :returns: pd.DataFrame, Results dataframe, giving complexity, MSE, and equations
        (as strings).

    """
    assert warmupMaxsize == None, "warmupMaxsize is deprecated. Use warmupMaxsizeBy and give a fraction of time."
    if nrestarts != None:
        optimizer_nrestarts = nrestarts

    assert optimizer_algorithm in ['NelderMead', 'BFGS']

    if isinstance(X, pd.DataFrame):
        variable_names = list(X.columns)
        X = np.array(X)

    use_custom_variable_names = (len(variable_names) != 0)

    if len(X.shape) == 1:
        X = X[:, None]

    _check_assertions(X, binary_operators, unary_operators,
                     use_custom_variable_names, variable_names, weights, y)


    if len(X) > 10000 and not batching:
        warnings.warn("Note: you are running with more than 10,000 datapoints. You should consider turning on batching (https://pysr.readthedocs.io/en/latest/docs/options/#batching). You should also reconsider if you need that many datapoints. Unless you have a large amount of noise (in which case you should smooth your dataset first), generally < 10,000 datapoints is enough to find a functional form with symbolic regression. More datapoints will lower the search speed.")

    X, variable_names = _handle_feature_selection(
            X, select_k_features,
            use_custom_variable_names, variable_names, y
        )

    if maxdepth is None:
        maxdepth = maxsize
    if populations is None:
        populations = procs
    if isinstance(binary_operators, str):
        binary_operators = [binary_operators]
    if isinstance(unary_operators, str):
        unary_operators = [unary_operators]
    if X is None:
        X, y = _using_test_input(X, test, y)

    kwargs = dict(X=X, y=y, weights=weights,
                 alpha=alpha, annealing=annealing, batchSize=batchSize,
                 batching=batching, binary_operators=binary_operators,
                 fast_cycle=fast_cycle,
                 fractionReplaced=fractionReplaced,
                 ncyclesperiteration=ncyclesperiteration,
                 niterations=niterations, npop=npop, topn=topn,
                 verbosity=verbosity, progress=progress, update=update,
                 julia_optimization=julia_optimization, timeout=timeout,
                 fractionReplacedHof=fractionReplacedHof,
                 hofMigration=hofMigration, maxdepth=maxdepth,
                 maxsize=maxsize, migration=migration,
                 optimizer_algorithm=optimizer_algorithm,
                 optimizer_nrestarts=optimizer_nrestarts,
                 optimize_probability=optimize_probability,
                 optimizer_iterations=optimizer_iterations,
                 parsimony=parsimony, perturbationFactor=perturbationFactor,
                 populations=populations, procs=procs,
                 shouldOptimizeConstants=shouldOptimizeConstants,
                 unary_operators=unary_operators, useFrequency=useFrequency,
                 use_custom_variable_names=use_custom_variable_names,
                 variable_names=variable_names, warmupMaxsizeBy=warmupMaxsizeBy,
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
                 julia_project=julia_project, loss=loss,
                 output_jax_format=output_jax_format)

    kwargs = {**_set_paths(tempdir), **kwargs}

    if temp_equation_file:
        equation_file = kwargs['tmpdir'] / f'hall_of_fame.csv'
    elif equation_file is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3]
        equation_file = 'hall_of_fame_' + date_time + '.csv'

    kwargs = {**dict(equation_file=equation_file), **kwargs}


    pkg_directory = kwargs['pkg_directory']
    kwargs['need_install'] = False
    if not (pkg_directory / 'Manifest.toml').is_file():
        kwargs['need_install'] = (not user_input) or _yesno("I will install Julia packages using PySR's Project.toml file. OK?")
        if kwargs['need_install']:
            print("OK. I will install at launch.")
            assert update

    kwargs['def_hyperparams'] = _create_inline_operators(**kwargs)

    _handle_constraints(**kwargs)

    kwargs['constraints_str'] = _make_constraints_str(**kwargs)
    kwargs['def_hyperparams'] = _make_hyperparams_julia_str(**kwargs)
    kwargs['def_datasets'] = _make_datasets_julia_str(**kwargs)

    _create_julia_files(**kwargs)
    _final_pysr_process(**kwargs)
    _set_globals(**kwargs)

    equations = get_hof(**kwargs)

    if delete_tempfiles:
        shutil.rmtree(kwargs['tmpdir'])

    return equations



def _set_globals(X, equation_file, extra_sympy_mappings, variable_names, **kwargs):
    global global_n_features
    global global_equation_file
    global global_variable_names
    global global_extra_sympy_mappings
    global_n_features = X.shape[1]
    global_equation_file = equation_file
    global_variable_names = variable_names
    global_extra_sympy_mappings = extra_sympy_mappings


def _final_pysr_process(julia_optimization, runfile_filename, timeout, **kwargs):
    command = [
        f'julia', f'-O{julia_optimization:d}',
        str(runfile_filename),
    ]
    if timeout is not None:
        command = [f'timeout', f'{timeout}'] + command
    _cmd_runner(command, **kwargs)

def _cmd_runner(command, **kwargs):
    if kwargs['verbosity'] > 0:
        print("Running on", ' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=-1)
    try:
        while True:
            line = process.stdout.readline()
            if not line: break
            decoded_line = (line.decode('utf-8')
                                .replace('\\033[K',  '\033[K')
                                .replace('\\033[1A', '\033[1A')
                                .replace('\\033[1B', '\033[1B')
                                .replace('\\r',      '\r'))
            print(decoded_line, end='')


        process.stdout.close()
        process.wait()
    except KeyboardInterrupt:
        print("Killing process... will return when done.")
        process.kill()

def _create_julia_files(dataset_filename, def_datasets,  hyperparam_filename, def_hyperparams,
                        fractionReplaced, ncyclesperiteration, niterations, npop,
                        runfile_filename, topn, verbosity, julia_project, procs, weights,
                        X, variable_names, pkg_directory, need_install, update, **kwargs):
    with open(hyperparam_filename, 'w') as f:
        print(def_hyperparams, file=f)
    with open(dataset_filename, 'w') as f:
        print(def_datasets, file=f)
    with open(runfile_filename, 'w') as f:
        if julia_project is None:
            julia_project = pkg_directory
        else:
            julia_project = Path(julia_project)
        print(f'import Pkg', file=f)
        print(f'Pkg.activate("{_escape_filename(julia_project)}")', file=f)
        if need_install:
            print(f'Pkg.instantiate()', file=f)
            print(f'Pkg.update()', file=f)
            print(f'Pkg.precompile()', file=f)
        elif update:
            print(f'Pkg.update()', file=f)
        print(f'using SymbolicRegression', file=f)
        print(f'include("{_escape_filename(hyperparam_filename)}")', file=f)
        print(f'include("{_escape_filename(dataset_filename)}")', file=f)
        if len(variable_names) == 0:
            varMap = "[" + ",".join([f'"x{i}"' for i in range(X.shape[1])]) + "]"
        else:
            varMap = "[" + ",".join(['"' + vname + '"' for vname in variable_names]) + "]"

        if weights is not None:
            print(f'EquationSearch(X, y, weights=weights, niterations={niterations:d}, varMap={varMap}, options=options, numprocs={procs})', file=f)
        else:
            print(f'EquationSearch(X, y, niterations={niterations:d}, varMap={varMap}, options=options, numprocs={procs})', file=f)


def _make_datasets_julia_str(X, X_filename, weights, weights_filename, y, y_filename, **kwargs):
    def_datasets = """using DelimitedFiles"""
    np.savetxt(X_filename, X.astype(np.float32), delimiter=',')
    np.savetxt(y_filename, y.reshape(-1, 1).astype(np.float32), delimiter=',')
    if weights is not None:
        np.savetxt(weights_filename, weights.reshape(-1, 1), delimiter=',')
    def_datasets += f"""
X = copy(transpose(readdlm("{_escape_filename(X_filename)}", ',', Float32, '\\n')))
y = readdlm("{_escape_filename(y_filename)}", ',', Float32, '\\n')[:, 1]"""
    if weights is not None:
        def_datasets += f"""
weights = readdlm("{_escape_filename(weights_filename)}", ',', Float32, '\\n')[:, 1]"""
    return def_datasets

def _make_hyperparams_julia_str(X, alpha, annealing, batchSize, batching, binary_operators, constraints_str,
                               def_hyperparams, equation_file, fast_cycle, fractionReplacedHof, hofMigration,
                               maxdepth, maxsize, migration,
                               optimizer_algorithm, optimizer_nrestarts,
                               optimize_probability, optimizer_iterations, npop,
                               parsimony, perturbationFactor, populations, procs, shouldOptimizeConstants,
                               unary_operators, useFrequency, use_custom_variable_names,
                               variable_names, warmupMaxsizeBy, weightAddNode,
                               ncyclesperiteration, fractionReplaced, topn, verbosity, progress, loss,
                               weightDeleteNode, weightDoNothing, weightInsertNode, weightMutateConstant,
                               weightMutateOperator, weightRandomize, weightSimplify, weights, **kwargs):
    try:
        term_width = shutil.get_terminal_size().columns
    except:
        _, term_width = subprocess.check_output(['stty', 'size']).split()
    def tuple_fix(ops):
        if len(ops) > 1:
            return ', '.join(ops)
        elif len(ops) == 0:
            return ''
        else:
            return ops[0] + ','

    def_hyperparams += f"""\n
plus=(+)
sub=(-)
mult=(*)
square=SymbolicRegression.square
cube=SymbolicRegression.cube
pow=(^)
div=(/)
logm=SymbolicRegression.logm
logm2=SymbolicRegression.logm2
logm10=SymbolicRegression.logm10
sqrtm=SymbolicRegression.sqrtm
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
ncyclesperiteration={ncyclesperiteration:d},
fractionReplaced={fractionReplaced:f}f0,
topn={topn:d},
verbosity=round(Int32, {verbosity:f}),
progress={'true' if progress else 'false'},
terminal_width={term_width:d}
"""

    def_hyperparams += '\n)'
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
        if op in ['plus', 'sub']:
            if constraints[op][0] != constraints[op][1]:
                raise NotImplementedError(
                    "You need equal constraints on both sides for - and *, due to simplification strategies.")
        elif op == 'mult':
            # Make sure the complex expression is in the left side.
            if constraints[op][0] == -1:
                continue
            elif constraints[op][1] == -1 or constraints[op][0] < constraints[op][1]:
                constraints[op][0], constraints[op][1] = constraints[op][1], constraints[op][0]


def _create_inline_operators(binary_operators, unary_operators, **kwargs):
    def_hyperparams = ""
    for op_list in [binary_operators, unary_operators]:
        for i in range(len(op_list)):
            op = op_list[i]
            is_user_defined_operator = '(' in op

            if is_user_defined_operator:
                def_hyperparams += op + "\n"
                # Cut off from the first non-alphanumeric char:
                first_non_char = [
                    j for j in range(len(op))
                    if not (op[j].isalpha() or op[j].isdigit())][0]
                function_name = op[:first_non_char]
                op_list[i] = function_name
    return def_hyperparams


def _using_test_input(X, test, y):
    if test == 'simple1':
        eval_str = "np.sign(X[:, 2])*np.abs(X[:, 2])**2.5 + 5*np.cos(X[:, 3]) - 5"
    elif test == 'simple2':
        eval_str = "np.sign(X[:, 2])*np.abs(X[:, 2])**3.5 + 1/(np.abs(X[:, 0])+1)"
    elif test == 'simple3':
        eval_str = "np.exp(X[:, 0]/2) + 12.0 + np.log(np.abs(X[:, 0])*10 + 1)"
    elif test == 'simple4':
        eval_str = "1.0 + 3*X[:, 0]**2 - 0.5*X[:, 0]**3 + 0.1*X[:, 0]**4"
    elif test == 'simple5':
        eval_str = "(np.exp(X[:, 3]) + 3)/(np.abs(X[:, 1]) + np.cos(X[:, 0]) + 1.1)"
    X = np.random.randn(100, 5) * 3
    y = eval(eval_str)
    print("Running on", eval_str)
    return X, y


def _handle_feature_selection(X, select_k_features, use_custom_variable_names, variable_names, y):
    if select_k_features is not None:
        selection = run_feature_selection(X, y, select_k_features)
        print(f"Using features {selection}")
        X = X[:, selection]

        if use_custom_variable_names:
            variable_names = [variable_names[selection[i]] for i in range(len(selection))]
    return X, variable_names


def _set_paths(tempdir):
    # System-independent paths
    pkg_directory = Path(__file__).parents[1]
    default_project_file = pkg_directory / "Project.toml"
    tmpdir = Path(tempfile.mkdtemp(dir=tempdir))
    hyperparam_filename = tmpdir / f'hyperparams.jl'
    dataset_filename = tmpdir / f'dataset.jl'
    runfile_filename = tmpdir / f'runfile.jl'
    X_filename = tmpdir / "X.csv"
    y_filename = tmpdir / "y.csv"
    weights_filename = tmpdir / "weights.csv"
    return dict(pkg_directory=pkg_directory,
	    default_project_file=default_project_file,
	    X_filename=X_filename,
            dataset_filename=dataset_filename,
            hyperparam_filename=hyperparam_filename,
            runfile_filename=runfile_filename, tmpdir=tmpdir,
            weights_filename=weights_filename, y_filename=y_filename)


def _check_assertions(X, binary_operators, unary_operators, use_custom_variable_names, variable_names, weights, y):
    # Check for potential errors before they happen
    assert len(unary_operators) + len(binary_operators) > 0
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert X.shape[0] == y.shape[0]
    if weights is not None:
        assert len(weights.shape) == 1
        assert X.shape[0] == weights.shape[0]
    if use_custom_variable_names:
        assert len(variable_names) == X.shape[1]


def run_feature_selection(X, y, select_k_features):
    """Use a gradient boosting tree regressor as a proxy for finding
        the k most important features in X, returning indices for those
        features as output."""

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.feature_selection import SelectFromModel, SelectKBest

    clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls') #RandomForestRegressor()
    clf.fit(X, y)
    selector = SelectFromModel(clf, threshold=-np.inf,
            max_features=select_k_features, prefit=True)
    return selector.get_support(indices=True)

def get_hof(equation_file=None, n_features=None, variable_names=None,
            extra_sympy_mappings=None, output_jax_format=False, **kwargs):
    """Get the equations from a hall of fame file. If no arguments
    entered, the ones used previously from a call to PySR will be used."""

    global global_n_features
    global global_equation_file
    global global_variable_names
    global global_extra_sympy_mappings

    if equation_file is None: equation_file = global_equation_file
    if n_features is None: n_features = global_n_features
    if variable_names is None: variable_names = global_variable_names
    if extra_sympy_mappings is None: extra_sympy_mappings = global_extra_sympy_mappings

    global_equation_file = equation_file
    global_n_features = n_features
    global_variable_names = variable_names
    global_extra_sympy_mappings = extra_sympy_mappings

    try:
        output = pd.read_csv(str(equation_file) + '.bkup', sep="|")
    except FileNotFoundError:
        print("Couldn't find equation file!")
        return pd.DataFrame()

    scores = []
    lastMSE = None
    lastComplexity = 0
    sympy_format = []
    lambda_format = []
    if output_jax_format:
        jax_format = []
    use_custom_variable_names = (len(variable_names) != 0)
    local_sympy_mappings = {
            **extra_sympy_mappings,
            **sympy_mappings
    }

    if use_custom_variable_names:
        sympy_symbols = [sympy.Symbol(variable_names[i]) for i in range(n_features)]
    else:
        sympy_symbols = [sympy.Symbol('x%d'%i) for i in range(n_features)]

    for i in range(len(output)):
        eqn = sympify(output.loc[i, 'Equation'], locals=local_sympy_mappings)
        sympy_format.append(eqn)
        if output_jax_format:
            func, params = sympy2jax(eqn, sympy_symbols)
            jax_format.append({'callable': func, 'parameters': params})
        lambda_format.append(lambdify(sympy_symbols, eqn))
        curMSE = output.loc[i, 'MSE']
        curComplexity = output.loc[i, 'Complexity']

        if lastMSE is None:
            cur_score = 0.0
        else:
            cur_score = - np.log(curMSE/lastMSE)/(curComplexity - lastComplexity)

        scores.append(cur_score)
        lastMSE = curMSE
        lastComplexity = curComplexity

    output['score'] = np.array(scores)
    output['sympy_format'] = sympy_format
    output['lambda_format'] = lambda_format
    output_cols = ['Complexity', 'MSE', 'score', 'Equation', 'sympy_format', 'lambda_format']
    if output_jax_format:
        output_cols += ['jax_format']
        output['jax_format'] = jax_format

    return output[output_cols]

def best_row(equations=None):
    """Return the best row of a hall of fame file using the score column.
    By default this uses the last equation file.
    """
    if equations is None: equations = get_hof()
    best_idx = np.argmax(equations['score'])
    return equations.iloc[best_idx]

def best_tex(equations=None):
    """Return the equation with the best score, in latex format
    By default this uses the last equation file.
    """
    if equations is None: equations = get_hof()
    best_sympy = best_row(equations)['sympy_format']
    return sympy.latex(best_sympy.simplify())

def best(equations=None):
    """Return the equation with the best score, in sympy format.
    By default this uses the last equation file.
    """
    if equations is None: equations = get_hof()
    best_sympy = best_row(equations)['sympy_format']
    return best_sympy.simplify()

def best_callable(equations=None):
    """Return the equation with the best score, in callable format.
    By default this uses the last equation file.
    """
    if equations is None: equations = get_hof()
    return best_row(equations)['lambda_format']

def _escape_filename(filename):
    """Turns a file into a string representation with correctly escaped backslashes"""
    repr = str(filename)
    repr = repr.replace('\\', '\\\\')
    return repr

# https://gist.github.com/garrettdreyfus/8153571
def _yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return _yesno(question)
    if ans == 'y':
        return True
    return False
