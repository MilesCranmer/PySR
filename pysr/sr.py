"""Define the PySRRegressor scikit-learn interface."""

import copy
import os
import pickle as pkl
import re
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass, fields
from datetime import datetime
from io import StringIO
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils.validation import _check_feature_names_in  # type: ignore
from sklearn.utils.validation import check_is_fitted

from .denoising import denoise, multi_denoise
from .deprecated import DEPRECATED_KWARGS
from .export_jax import sympy2jax
from .export_latex import (
    sympy2latex,
    sympy2latextable,
    sympy2multilatextable,
    with_preamble,
)
from .export_numpy import sympy2numpy
from .export_sympy import assert_valid_sympy_symbol, create_sympy_symbols, pysr2sympy
from .export_torch import sympy2torch
from .feature_selection import run_feature_selection
from .julia_extensions import load_required_packages
from .julia_helpers import (
    PythonCall,
    _escape_filename,
    _load_cluster_manager,
    jl_array,
    jl_deserialize,
    jl_is_function,
    jl_serialize,
)
from .julia_import import SymbolicRegression, jl
from .utils import (
    ArrayLike,
    PathLike,
    _csv_filename_to_pkl_filename,
    _preprocess_julia_floats,
    _safe_check_feature_names_in,
    _subscriptify,
    _suggest_keywords,
)

ALREADY_RAN = False


def _process_constraints(binary_operators, unary_operators, constraints):
    constraints = constraints.copy()
    for op in unary_operators:
        if op not in constraints:
            constraints[op] = -1
    for op in binary_operators:
        if op not in constraints:
            if op in ["^", "pow"]:
                # Warn user that they should set up constraints
                warnings.warn(
                    "You are using the `^` operator, but have not set up `constraints` for it. "
                    "This may lead to overly complex expressions. "
                    "One typical constraint is to use `constraints={..., '^': (-1, 1)}`, which "
                    "will allow arbitrary-complexity base (-1) but only powers such as "
                    "a constant or variable (1). "
                    "For more tips, please see https://astroautomata.com/PySR/tuning/"
                )
            constraints[op] = (-1, -1)
        if op in ["plus", "sub", "+", "-"]:
            if constraints[op][0] != constraints[op][1]:
                raise NotImplementedError(
                    "You need equal constraints on both sides for - and +, "
                    "due to simplification strategies."
                )
        elif op in ["mult", "*"]:
            # Make sure the complex expression is in the left side.
            if constraints[op][0] == -1:
                continue
            if constraints[op][1] == -1 or constraints[op][0] < constraints[op][1]:
                constraints[op][0], constraints[op][1] = (
                    constraints[op][1],
                    constraints[op][0],
                )
    return constraints


def _maybe_create_inline_operators(
    binary_operators, unary_operators, extra_sympy_mappings
):
    binary_operators = binary_operators.copy()
    unary_operators = unary_operators.copy()
    for op_list in [binary_operators, unary_operators]:
        for i, op in enumerate(op_list):
            is_user_defined_operator = "(" in op

            if is_user_defined_operator:
                jl.seval(op)
                # Cut off from the first non-alphanumeric char:
                first_non_char = [j for j, char in enumerate(op) if char == "("][0]
                function_name = op[:first_non_char]
                # Assert that function_name only contains
                # alphabetical characters, numbers,
                # and underscores:
                if not re.match(r"^[a-zA-Z0-9_]+$", function_name):
                    raise ValueError(
                        f"Invalid function name {function_name}. "
                        "Only alphanumeric characters, numbers, "
                        "and underscores are allowed."
                    )
                if (extra_sympy_mappings is None) or (
                    function_name not in extra_sympy_mappings
                ):
                    raise ValueError(
                        f"Custom function {function_name} is not defined in `extra_sympy_mappings`. "
                        "You can define it with, "
                        "e.g., `model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})`, where "
                        "`lambda x: 1/x` is a valid SymPy function defining the operator. "
                        "You can also define these at initialization time."
                    )
                op_list[i] = function_name
    return binary_operators, unary_operators


def _check_assertions(
    X,
    use_custom_variable_names,
    variable_names,
    complexity_of_variables,
    weights,
    y,
    X_units,
    y_units,
):
    # Check for potential errors before they happen
    assert len(X.shape) == 2
    assert len(y.shape) in [1, 2]
    assert X.shape[0] == y.shape[0]
    if weights is not None:
        assert weights.shape == y.shape
        assert X.shape[0] == weights.shape[0]
    if use_custom_variable_names:
        assert len(variable_names) == X.shape[1]
        # Check none of the variable names are function names:
        for var_name in variable_names:
            # Check if alphanumeric only:
            if not re.match(r"^[₀₁₂₃₄₅₆₇₈₉a-zA-Z0-9_]+$", var_name):
                raise ValueError(
                    f"Invalid variable name {var_name}. "
                    "Only alphanumeric characters, numbers, "
                    "and underscores are allowed."
                )
            assert_valid_sympy_symbol(var_name)
    if (
        isinstance(complexity_of_variables, list)
        and len(complexity_of_variables) != X.shape[1]
    ):
        raise ValueError(
            "The number of elements in `complexity_of_variables` must equal the number of features in `X`."
        )
    if X_units is not None and len(X_units) != X.shape[1]:
        raise ValueError(
            "The number of units in `X_units` must equal the number of features in `X`."
        )
    if y_units is not None:
        good_y_units = False
        if isinstance(y_units, list):
            if len(y.shape) == 1:
                good_y_units = len(y_units) == 1
            else:
                good_y_units = len(y_units) == y.shape[1]
        else:
            good_y_units = len(y.shape) == 1 or y.shape[1] == 1

        if not good_y_units:
            raise ValueError(
                "The number of units in `y_units` must equal the number of output features in `y`."
            )


# Class validation constants
VALID_OPTIMIZER_ALGORITHMS = ["BFGS", "NelderMead"]


@dataclass
class _DynamicallySetParams:
    """Defines some parameters that are set at runtime."""

    binary_operators: List[str]
    unary_operators: List[str]
    maxdepth: int
    constraints: Dict[str, str]
    multithreading: bool
    batch_size: int
    update_verbosity: int
    progress: bool
    warmup_maxsize_by: float


class PySRRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """
    High-performance symbolic regression algorithm.

    This is the scikit-learn interface for SymbolicRegression.jl.
    This model will automatically search for equations which fit
    a given dataset subject to a particular loss and set of
    constraints.

    Most default parameters have been tuned over several example equations,
    but you should adjust `niterations`, `binary_operators`, `unary_operators`
    to your requirements. You can view more detailed explanations of the options
    on the [options page](https://astroautomata.com/PySR/options) of the
    documentation.

    Parameters
    ----------
    model_selection : str
        Model selection criterion when selecting a final expression from
        the list of best expression at each complexity.
        Can be `'accuracy'`, `'best'`, or `'score'`. Default is `'best'`.
        `'accuracy'` selects the candidate model with the lowest loss
        (highest accuracy).
        `'score'` selects the candidate model with the highest score.
        Score is defined as the negated derivative of the log-loss with
        respect to complexity - if an expression has a much better
        loss at a slightly higher complexity, it is preferred.
        `'best'` selects the candidate model with the highest score
        among expressions with a loss better than at least 1.5x the
        most accurate model.
    binary_operators : list[str]
        List of strings for binary operators used in the search.
        See the [operators page](https://astroautomata.com/PySR/operators/)
        for more details.
        Default is `["+", "-", "*", "/"]`.
    unary_operators : list[str]
        Operators which only take a single scalar as input.
        For example, `"cos"` or `"exp"`.
        Default is `None`.
    niterations : int
        Number of iterations of the algorithm to run. The best
        equations are printed and migrate between populations at the
        end of each iteration.
        Default is `40`.
    populations : int
        Number of populations running.
        Default is `15`.
    population_size : int
        Number of individuals in each population.
        Default is `33`.
    max_evals : int
        Limits the total number of evaluations of expressions to
        this number.  Default is `None`.
    maxsize : int
        Max complexity of an equation.  Default is `20`.
    maxdepth : int
        Max depth of an equation. You can use both `maxsize` and
        `maxdepth`. `maxdepth` is by default not used.
        Default is `None`.
    warmup_maxsize_by : float
        Whether to slowly increase max size from a small number up to
        the maxsize (if greater than 0).  If greater than 0, says the
        fraction of training time at which the current maxsize will
        reach the user-passed maxsize.
        Default is `0.0`.
    timeout_in_seconds : float
        Make the search return early once this many seconds have passed.
        Default is `None`.
    constraints : dict[str, int | tuple[int,int]]
        Dictionary of int (unary) or 2-tuples (binary), this enforces
        maxsize constraints on the individual arguments of operators.
        E.g., `'pow': (-1, 1)` says that power laws can have any
        complexity left argument, but only 1 complexity in the right
        argument. Use this to force more interpretable solutions.
        Default is `None`.
    nested_constraints : dict[str, dict]
        Specifies how many times a combination of operators can be
        nested. For example, `{"sin": {"cos": 0}}, "cos": {"cos": 2}}`
        specifies that `cos` may never appear within a `sin`, but `sin`
        can be nested with itself an unlimited number of times. The
        second term specifies that `cos` can be nested up to 2 times
        within a `cos`, so that `cos(cos(cos(x)))` is allowed
        (as well as any combination of `+` or `-` within it), but
        `cos(cos(cos(cos(x))))` is not allowed. When an operator is not
        specified, it is assumed that it can be nested an unlimited
        number of times. This requires that there is no operator which
        is used both in the unary operators and the binary operators
        (e.g., `-` could be both subtract, and negation). For binary
        operators, you only need to provide a single number: both
        arguments are treated the same way, and the max of each
        argument is constrained.
        Default is `None`.
    elementwise_loss : str
        String of Julia code specifying an elementwise loss function.
        Can either be a loss from LossFunctions.jl, or your own loss
        written as a function. Examples of custom written losses include:
        `myloss(x, y) = abs(x-y)` for non-weighted, or
        `myloss(x, y, w) = w*abs(x-y)` for weighted.
        The included losses include:
        Regression: `LPDistLoss{P}()`, `L1DistLoss()`,
        `L2DistLoss()` (mean square), `LogitDistLoss()`,
        `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`,
        `PeriodicLoss(c)`, `QuantileLoss(τ)`.
        Classification: `ZeroOneLoss()`, `PerceptronLoss()`,
        `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`,
        `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`,
        `SigmoidLoss()`, `DWDMarginLoss(q)`.
        Default is `"L2DistLoss()"`.
    loss_function : str
        Alternatively, you can specify the full objective function as
        a snippet of Julia code, including any sort of custom evaluation
        (including symbolic manipulations beforehand), and any sort
        of loss function or regularizations. The default `loss_function`
        used in SymbolicRegression.jl is roughly equal to:
        ```julia
        function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return L(Inf)
            end
            return sum((prediction .- dataset.y) .^ 2) / dataset.n
        end
        ```
        where the example elementwise loss is mean-squared error.
        You may pass a function with the same arguments as this (note
        that the name of the function doesn't matter). Here,
        both `prediction` and `dataset.y` are 1D arrays of length `dataset.n`.
        If using `batching`, then you should add an
        `idx` argument to the function, which is `nothing`
        for non-batched, and a 1D array of indices for batched.
        Default is `None`.
    complexity_of_operators : dict[str, Union[int, float]]
        If you would like to use a complexity other than 1 for an
        operator, specify the complexity here. For example,
        `{"sin": 2, "+": 1}` would give a complexity of 2 for each use
        of the `sin` operator, and a complexity of 1 for each use of
        the `+` operator (which is the default). You may specify real
        numbers for a complexity, and the total complexity of a tree
        will be rounded to the nearest integer after computing.
        Default is `None`.
    complexity_of_constants : int | float
        Complexity of constants. Default is `1`.
    complexity_of_variables : int | float
        Global complexity of variables. To set different complexities for
        different variables, pass a list of complexities to the `fit` method
        with keyword `complexity_of_variables`. You cannot use both.
        Default is `1`.
    parsimony : float
        Multiplicative factor for how much to punish complexity.
        Default is `0.0032`.
    dimensional_constraint_penalty : float
        Additive penalty for if dimensional analysis of an expression fails.
        By default, this is `1000.0`.
    dimensionless_constants_only : bool
        Whether to only search for dimensionless constants, if using units.
        Default is `False`.
    use_frequency : bool
        Whether to measure the frequency of complexities, and use that
        instead of parsimony to explore equation space. Will naturally
        find equations of all complexities.
        Default is `True`.
    use_frequency_in_tournament : bool
        Whether to use the frequency mentioned above in the tournament,
        rather than just the simulated annealing.
        Default is `True`.
    adaptive_parsimony_scaling : float
        If the adaptive parsimony strategy (`use_frequency` and
        `use_frequency_in_tournament`), this is how much to (exponentially)
        weight the contribution. If you find that the search is only optimizing
        the most complex expressions while the simpler expressions remain stagnant,
        you should increase this value.
        Default is `20.0`.
    alpha : float
        Initial temperature for simulated annealing
        (requires `annealing` to be `True`).
        Default is `0.1`.
    annealing : bool
        Whether to use annealing.  Default is `False`.
    early_stop_condition : float | str
        Stop the search early if this loss is reached. You may also
        pass a string containing a Julia function which
        takes a loss and complexity as input, for example:
        `"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`.
        Default is `None`.
    ncycles_per_iteration : int
        Number of total mutations to run, per 10 samples of the
        population, per iteration.
        Default is `550`.
    fraction_replaced : float
        How much of population to replace with migrating equations from
        other populations.
        Default is `0.000364`.
    fraction_replaced_hof : float
        How much of population to replace with migrating equations from
        hall of fame. Default is `0.035`.
    weight_add_node : float
        Relative likelihood for mutation to add a node.
        Default is `0.79`.
    weight_insert_node : float
        Relative likelihood for mutation to insert a node.
        Default is `5.1`.
    weight_delete_node : float
        Relative likelihood for mutation to delete a node.
        Default is `1.7`.
    weight_do_nothing : float
        Relative likelihood for mutation to leave the individual.
        Default is `0.21`.
    weight_mutate_constant : float
        Relative likelihood for mutation to change the constant slightly
        in a random direction.
        Default is `0.048`.
    weight_mutate_operator : float
        Relative likelihood for mutation to swap an operator.
        Default is `0.47`.
    weight_swap_operands : float
        Relative likehood for swapping operands in binary operators.
        Default is `0.1`.
    weight_randomize : float
        Relative likelihood for mutation to completely delete and then
        randomly generate the equation
        Default is `0.00023`.
    weight_simplify : float
        Relative likelihood for mutation to simplify constant parts by evaluation
        Default is `0.0020`.
    weight_optimize: float
        Constant optimization can also be performed as a mutation, in addition to
        the normal strategy controlled by `optimize_probability` which happens
        every iteration. Using it as a mutation is useful if you want to use
        a large `ncycles_periteration`, and may not optimize very often.
        Default is `0.0`.
    crossover_probability : float
        Absolute probability of crossover-type genetic operation, instead of a mutation.
        Default is `0.066`.
    skip_mutation_failures : bool
        Whether to skip mutation and crossover failures, rather than
        simply re-sampling the current member.
        Default is `True`.
    migration : bool
        Whether to migrate.  Default is `True`.
    hof_migration : bool
        Whether to have the hall of fame migrate.  Default is `True`.
    topn : int
        How many top individuals migrate from each population.
        Default is `12`.
    should_simplify : bool
        Whether to use algebraic simplification in the search. Note that only
        a few simple rules are implemented. Default is `True`.
    should_optimize_constants : bool
        Whether to numerically optimize constants (Nelder-Mead/Newton)
        at the end of each iteration. Default is `True`.
    optimizer_algorithm : str
        Optimization scheme to use for optimizing constants. Can currently
        be `NelderMead` or `BFGS`.
        Default is `"BFGS"`.
    optimizer_nrestarts : int
        Number of time to restart the constants optimization process with
        different initial conditions.
        Default is `2`.
    optimize_probability : float
        Probability of optimizing the constants during a single iteration of
        the evolutionary algorithm.
        Default is `0.14`.
    optimizer_iterations : int
        Number of iterations that the constants optimizer can take.
        Default is `8`.
    perturbation_factor : float
        Constants are perturbed by a max factor of
        (perturbation_factor*T + 1). Either multiplied by this or
        divided by this.
        Default is `0.076`.
    tournament_selection_n : int
        Number of expressions to consider in each tournament.
        Default is `10`.
    tournament_selection_p : float
        Probability of selecting the best expression in each
        tournament. The probability will decay as p*(1-p)^n for other
        expressions, sorted by loss.
        Default is `0.86`.
    procs : int
        Number of processes (=number of populations running).
        Default is `cpu_count()`.
    multithreading : bool
        Use multithreading instead of distributed backend.
        Using procs=0 will turn off both. Default is `True`.
    cluster_manager : str
        For distributed computing, this sets the job queue system. Set
        to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or
        "htc". If set to one of these, PySR will run in distributed
        mode, and use `procs` to figure out how many processes to launch.
        Default is `None`.
    heap_size_hint_in_bytes : int
        For multiprocessing, this sets the `--heap-size-hint` parameter
        for new Julia processes. This can be configured when using
        multi-node distributed compute, to give a hint to each process
        about how much memory they can use before aggressive garbage
        collection.
    batching : bool
        Whether to compare population members on small batches during
        evolution. Still uses full dataset for comparing against hall
        of fame. Default is `False`.
    batch_size : int
        The amount of data to use if doing batching. Default is `50`.
    fast_cycle : bool
        Batch over population subsamples. This is a slightly different
        algorithm than regularized evolution, but does cycles 15%
        faster. May be algorithmically less efficient.
        Default is `False`.
    turbo: bool
        (Experimental) Whether to use LoopVectorization.jl to speed up the
        search evaluation. Certain operators may not be supported.
        Does not support 16-bit precision floats.
        Default is `False`.
    bumper: bool
        (Experimental) Whether to use Bumper.jl to speed up the search
        evaluation. Does not support 16-bit precision floats.
        Default is `False`.
    precision : int
        What precision to use for the data. By default this is `32`
        (float32), but you can select `64` or `16` as well, giving
        you 64 or 16 bits of floating point precision, respectively.
        If you pass complex data, the corresponding complex precision
        will be used (i.e., `64` for complex128, `32` for complex64).
        Default is `32`.
    enable_autodiff : bool
        Whether to create derivative versions of operators for automatic
        differentiation. This is only necessary if you wish to compute
        the gradients of an expression within a custom loss function.
        Default is `False`.
    random_state : int, Numpy RandomState instance or None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
        Default is `None`.
    deterministic : bool
        Make a PySR search give the same result every run.
        To use this, you must turn off parallelism
        (with `procs`=0, `multithreading`=False),
        and set `random_state` to a fixed seed.
        Default is `False`.
    warm_start : bool
        Tells fit to continue from where the last call to fit finished.
        If false, each call to fit will be fresh, overwriting previous results.
        Default is `False`.
    verbosity : int
        What verbosity level to use. 0 means minimal print statements.
        Default is `1`.
    update_verbosity : int
        What verbosity level to use for package updates.
        Will take value of `verbosity` if not given.
        Default is `None`.
    print_precision : int
        How many significant digits to print for floats. Default is `5`.
    progress : bool
        Whether to use a progress bar instead of printing to stdout.
        Default is `True`.
    equation_file : str
        Where to save the files (.csv extension).
        Default is `None`.
    temp_equation_file : bool
        Whether to put the hall of fame file in the temp directory.
        Deletion is then controlled with the `delete_tempfiles`
        parameter.
        Default is `False`.
    tempdir : str
        directory for the temporary files. Default is `None`.
    delete_tempfiles : bool
        Whether to delete the temporary files after finishing.
        Default is `True`.
    update: bool
        Whether to automatically update Julia packages when `fit` is called.
        You should make sure that PySR is up-to-date itself first, as
        the packaged Julia packages may not necessarily include all
        updated dependencies.
        Default is `False`.
    output_jax_format : bool
        Whether to create a 'jax_format' column in the output,
        containing jax-callable functions and the default parameters in
        a jax array.
        Default is `False`.
    output_torch_format : bool
        Whether to create a 'torch_format' column in the output,
        containing a torch module with trainable parameters.
        Default is `False`.
    extra_sympy_mappings : dict[str, Callable]
        Provides mappings between custom `binary_operators` or
        `unary_operators` defined in julia strings, to those same
        operators defined in sympy.
        E.G if `unary_operators=["inv(x)=1/x"]`, then for the fitted
        model to be export to sympy, `extra_sympy_mappings`
        would be `{"inv": lambda x: 1/x}`.
        Default is `None`.
    extra_jax_mappings : dict[Callable, str]
        Similar to `extra_sympy_mappings` but for model export
        to jax. The dictionary maps sympy functions to jax functions.
        For example: `extra_jax_mappings={sympy.sin: "jnp.sin"}` maps
        the `sympy.sin` function to the equivalent jax expression `jnp.sin`.
        Default is `None`.
    extra_torch_mappings : dict[Callable, Callable]
        The same as `extra_jax_mappings` but for model export
        to pytorch. Note that the dictionary keys should be callable
        pytorch expressions.
        For example: `extra_torch_mappings={sympy.sin: torch.sin}`.
        Default is `None`.
    denoise : bool
        Whether to use a Gaussian Process to denoise the data before
        inputting to PySR. Can help PySR fit noisy data.
        Default is `False`.
    select_k_features : int
        Whether to run feature selection in Python using random forests,
        before passing to the symbolic regression code. None means no
        feature selection; an int means select that many features.
        Default is `None`.
    **kwargs : dict
        Supports deprecated keyword arguments. Other arguments will
        result in an error.
    Attributes
    ----------
    equations_ : pandas.DataFrame | list[pandas.DataFrame]
        Processed DataFrame containing the results of model fitting.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    display_feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Pretty names of features, used only during printing.
    X_units_ : list[str] of length n_features
        Units of each variable in the training dataset, `X`.
    y_units_ : str | list[str] of length n_out
        Units of each variable in the training dataset, `y`.
    nout_ : int
        Number of output dimensions.
    selection_mask_ : ndarray of shape (`n_features_in_`,)
        Mask of which features of `X` to use when `select_k_features` is set.
    tempdir_ : Path
        Path to the temporary equations directory.
    equation_file_ : Union[str, Path]
        Output equation file name produced by the julia backend.
    julia_state_stream_ : ndarray
        The serialized state for the julia SymbolicRegression.jl backend (after fitting),
        stored as an array of uint8, produced by Julia's Serialization.serialize function.
    julia_options_stream_ : ndarray
        The serialized julia options, stored as an array of uint8,
    equation_file_contents_ : list[pandas.DataFrame]
        Contents of the equation file output by the Julia backend.
    show_pickle_warnings_ : bool
        Whether to show warnings about what attributes can be pickled.

    Examples
    --------
    ```python
    >>> import numpy as np
    >>> from pysr import PySRRegressor
    >>> randstate = np.random.RandomState(0)
    >>> X = 2 * randstate.randn(100, 5)
    >>> # y = 2.5382 * cos(x_3) + x_0 - 0.5
    >>> y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
    >>> model = PySRRegressor(
    ...     niterations=40,
    ...     binary_operators=["+", "*"],
    ...     unary_operators=[
    ...         "cos",
    ...         "exp",
    ...         "sin",
    ...         "inv(x) = 1/x",  # Custom operator (julia syntax)
    ...     ],
    ...     model_selection="best",
    ...     elementwise_loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
    ... )
    >>> model.fit(X, y)
    >>> model
    PySRRegressor.equations_ = [
    0         0.000000                                          3.8552167  3.360272e+01           1
    1         1.189847                                          (x0 * x0)  3.110905e+00           3
    2         0.010626                          ((x0 * x0) + -0.25573406)  3.045491e+00           5
    3         0.896632                              (cos(x3) + (x0 * x0))  1.242382e+00           6
    4         0.811362                ((x0 * x0) + (cos(x3) * 2.4384754))  2.451971e-01           8
    5  >>>>  13.733371          (((cos(x3) * 2.5382) + (x0 * x0)) + -0.5)  2.889755e-13          10
    6         0.194695  ((x0 * x0) + (((cos(x3) + -0.063180044) * 2.53...  1.957723e-13          12
    7         0.006988  ((x0 * x0) + (((cos(x3) + -0.32505524) * 1.538...  1.944089e-13          13
    8         0.000955  (((((x0 * x0) + cos(x3)) + -0.8251649) + (cos(...  1.940381e-13          15
    ]
    >>> model.score(X, y)
    1.0
    >>> model.predict(np.array([1,2,3,4,5]))
    array([-1.15907818, -1.15907818, -1.15907818, -1.15907818, -1.15907818])
    ```
    """

    equations_: Union[pd.DataFrame, List[pd.DataFrame], None]
    n_features_in_: int
    feature_names_in_: ArrayLike[str]
    display_feature_names_in_: ArrayLike[str]
    complexity_of_variables_: Union[int, float, List[Union[int, float]], None]
    X_units_: Union[ArrayLike[str], None]
    y_units_: Union[str, ArrayLike[str], None]
    nout_: int
    selection_mask_: Union[NDArray[np.bool_], None]
    tempdir_: Path
    equation_file_: PathLike
    julia_state_stream_: Union[NDArray[np.uint8], None]
    julia_options_stream_: Union[NDArray[np.uint8], None]
    equation_file_contents_: Union[List[pd.DataFrame], None]
    show_pickle_warnings_: bool

    def __init__(
        self,
        model_selection: Literal["best", "accuracy", "score"] = "best",
        *,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        niterations: int = 40,
        populations: int = 15,
        population_size: int = 33,
        max_evals: Optional[int] = None,
        maxsize: int = 20,
        maxdepth: Optional[int] = None,
        warmup_maxsize_by: Optional[float] = None,
        timeout_in_seconds: Optional[float] = None,
        constraints: Optional[Dict[str, Union[int, Tuple[int, int]]]] = None,
        nested_constraints: Optional[Dict[str, Dict[str, int]]] = None,
        elementwise_loss: Optional[str] = None,
        loss_function: Optional[str] = None,
        complexity_of_operators: Optional[Dict[str, Union[int, float]]] = None,
        complexity_of_constants: Union[int, float] = 1,
        complexity_of_variables: Optional[Union[int, float]] = None,
        parsimony: float = 0.0032,
        dimensional_constraint_penalty: Optional[float] = None,
        dimensionless_constants_only: bool = False,
        use_frequency: bool = True,
        use_frequency_in_tournament: bool = True,
        adaptive_parsimony_scaling: float = 20.0,
        alpha: float = 0.1,
        annealing: bool = False,
        early_stop_condition: Optional[Union[float, str]] = None,
        ncycles_per_iteration: int = 550,
        fraction_replaced: float = 0.000364,
        fraction_replaced_hof: float = 0.035,
        weight_add_node: float = 0.79,
        weight_insert_node: float = 5.1,
        weight_delete_node: float = 1.7,
        weight_do_nothing: float = 0.21,
        weight_mutate_constant: float = 0.048,
        weight_mutate_operator: float = 0.47,
        weight_swap_operands: float = 0.1,
        weight_randomize: float = 0.00023,
        weight_simplify: float = 0.0020,
        weight_optimize: float = 0.0,
        crossover_probability: float = 0.066,
        skip_mutation_failures: bool = True,
        migration: bool = True,
        hof_migration: bool = True,
        topn: int = 12,
        should_simplify: Optional[bool] = None,
        should_optimize_constants: bool = True,
        optimizer_algorithm: Literal["BFGS", "NelderMead"] = "BFGS",
        optimizer_nrestarts: int = 2,
        optimize_probability: float = 0.14,
        optimizer_iterations: int = 8,
        perturbation_factor: float = 0.076,
        tournament_selection_n: int = 10,
        tournament_selection_p: float = 0.86,
        procs: int = cpu_count(),
        multithreading: Optional[bool] = None,
        cluster_manager: Optional[
            Literal["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"]
        ] = None,
        heap_size_hint_in_bytes: Optional[int] = None,
        batching: bool = False,
        batch_size: int = 50,
        fast_cycle: bool = False,
        turbo: bool = False,
        bumper: bool = False,
        precision: int = 32,
        enable_autodiff: bool = False,
        random_state=None,
        deterministic: bool = False,
        warm_start: bool = False,
        verbosity: int = 1,
        update_verbosity: Optional[int] = None,
        print_precision: int = 5,
        progress: bool = True,
        equation_file: Optional[str] = None,
        temp_equation_file: bool = False,
        tempdir: Optional[str] = None,
        delete_tempfiles: bool = True,
        update: bool = False,
        output_jax_format: bool = False,
        output_torch_format: bool = False,
        extra_sympy_mappings: Optional[Dict[str, Callable]] = None,
        extra_torch_mappings: Optional[Dict[Callable, Callable]] = None,
        extra_jax_mappings: Optional[Dict[Callable, str]] = None,
        denoise: bool = False,
        select_k_features: Optional[int] = None,
        **kwargs,
    ):
        # Hyperparameters
        # - Model search parameters
        self.model_selection = model_selection
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.ncycles_per_iteration = ncycles_per_iteration
        # - Equation Constraints
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.warmup_maxsize_by = warmup_maxsize_by
        self.should_simplify = should_simplify
        # - Early exit conditions:
        self.max_evals = max_evals
        self.timeout_in_seconds = timeout_in_seconds
        self.early_stop_condition = early_stop_condition
        # - Loss parameters
        self.elementwise_loss = elementwise_loss
        self.loss_function = loss_function
        self.complexity_of_operators = complexity_of_operators
        self.complexity_of_constants = complexity_of_constants
        self.complexity_of_variables = complexity_of_variables
        self.parsimony = parsimony
        self.dimensional_constraint_penalty = dimensional_constraint_penalty
        self.dimensionless_constants_only = dimensionless_constants_only
        self.use_frequency = use_frequency
        self.use_frequency_in_tournament = use_frequency_in_tournament
        self.adaptive_parsimony_scaling = adaptive_parsimony_scaling
        self.alpha = alpha
        self.annealing = annealing
        # - Evolutionary search parameters
        # -- Mutation parameters
        self.weight_add_node = weight_add_node
        self.weight_insert_node = weight_insert_node
        self.weight_delete_node = weight_delete_node
        self.weight_do_nothing = weight_do_nothing
        self.weight_mutate_constant = weight_mutate_constant
        self.weight_mutate_operator = weight_mutate_operator
        self.weight_swap_operands = weight_swap_operands
        self.weight_randomize = weight_randomize
        self.weight_simplify = weight_simplify
        self.weight_optimize = weight_optimize
        self.crossover_probability = crossover_probability
        self.skip_mutation_failures = skip_mutation_failures
        # -- Migration parameters
        self.migration = migration
        self.hof_migration = hof_migration
        self.fraction_replaced = fraction_replaced
        self.fraction_replaced_hof = fraction_replaced_hof
        self.topn = topn
        # -- Constants parameters
        self.should_optimize_constants = should_optimize_constants
        self.optimizer_algorithm = optimizer_algorithm
        self.optimizer_nrestarts = optimizer_nrestarts
        self.optimize_probability = optimize_probability
        self.optimizer_iterations = optimizer_iterations
        self.perturbation_factor = perturbation_factor
        # -- Selection parameters
        self.tournament_selection_n = tournament_selection_n
        self.tournament_selection_p = tournament_selection_p
        # -- Performance parameters
        self.procs = procs
        self.multithreading = multithreading
        self.cluster_manager = cluster_manager
        self.heap_size_hint_in_bytes = heap_size_hint_in_bytes
        self.batching = batching
        self.batch_size = batch_size
        self.fast_cycle = fast_cycle
        self.turbo = turbo
        self.bumper = bumper
        self.precision = precision
        self.enable_autodiff = enable_autodiff
        self.random_state = random_state
        self.deterministic = deterministic
        self.warm_start = warm_start
        # Additional runtime parameters
        # - Runtime user interface
        self.verbosity = verbosity
        self.update_verbosity = update_verbosity
        self.print_precision = print_precision
        self.progress = progress
        # - Project management
        self.equation_file = equation_file
        self.temp_equation_file = temp_equation_file
        self.tempdir = tempdir
        self.delete_tempfiles = delete_tempfiles
        self.update = update
        self.output_jax_format = output_jax_format
        self.output_torch_format = output_torch_format
        self.extra_sympy_mappings = extra_sympy_mappings
        self.extra_jax_mappings = extra_jax_mappings
        self.extra_torch_mappings = extra_torch_mappings
        # Pre-modelling transformation
        self.denoise = denoise
        self.select_k_features = select_k_features

        # Once all valid parameters have been assigned handle the
        # deprecated kwargs
        if len(kwargs) > 0:  # pragma: no cover
            for k, v in kwargs.items():
                # Handle renamed kwargs
                if k in DEPRECATED_KWARGS:
                    updated_kwarg_name = DEPRECATED_KWARGS[k]
                    setattr(self, updated_kwarg_name, v)
                    warnings.warn(
                        f"`{k}` has been renamed to `{updated_kwarg_name}` in PySRRegressor. "
                        "Please use that instead.",
                        FutureWarning,
                    )
                # Handle kwargs that have been moved to the fit method
                elif k in ["weights", "variable_names", "Xresampled"]:
                    warnings.warn(
                        f"`{k}` is a data-dependent parameter and should be passed when fit is called. "
                        f"Ignoring parameter; please pass `{k}` during the call to fit instead.",
                        FutureWarning,
                    )
                elif k == "julia_project":
                    warnings.warn(
                        "The `julia_project` parameter has been deprecated. To use a custom "
                        "julia project, please see `https://astroautomata.com/PySR/backend`.",
                        FutureWarning,
                    )
                elif k == "julia_kwargs":
                    warnings.warn(
                        "The `julia_kwargs` parameter has been deprecated. To pass custom "
                        "keyword arguments to the julia backend, you should use environment variables. "
                        "See the Julia documentation for more information.",
                        FutureWarning,
                    )
                else:
                    suggested_keywords = _suggest_keywords(PySRRegressor, k)
                    err_msg = (
                        f"`{k}` is not a valid keyword argument for PySRRegressor."
                    )
                    if len(suggested_keywords) > 0:
                        err_msg += f" Did you mean {', '.join(map(lambda s: f'`{s}`', suggested_keywords))}?"
                    raise TypeError(err_msg)

    @classmethod
    def from_file(
        cls,
        equation_file: PathLike,
        *,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        n_features_in: Optional[int] = None,
        feature_names_in: Optional[ArrayLike[str]] = None,
        selection_mask: Optional[NDArray[np.bool_]] = None,
        nout: int = 1,
        **pysr_kwargs,
    ):
        """
        Create a model from a saved model checkpoint or equation file.

        Parameters
        ----------
        equation_file : str or Path
            Path to a pickle file containing a saved model, or a csv file
            containing equations.
        binary_operators : list[str]
            The same binary operators used when creating the model.
            Not needed if loading from a pickle file.
        unary_operators : list[str]
            The same unary operators used when creating the model.
            Not needed if loading from a pickle file.
        n_features_in : int
            Number of features passed to the model.
            Not needed if loading from a pickle file.
        feature_names_in : list[str]
            Names of the features passed to the model.
            Not needed if loading from a pickle file.
        selection_mask : NDArray[np.bool_]
            If using `select_k_features`, you must pass `model.selection_mask_` here.
            Not needed if loading from a pickle file.
        nout : int
            Number of outputs of the model.
            Not needed if loading from a pickle file.
            Default is `1`.
        **pysr_kwargs : dict
            Any other keyword arguments to initialize the PySRRegressor object.
            These will overwrite those stored in the pickle file.
            Not needed if loading from a pickle file.

        Returns
        -------
        model : PySRRegressor
            The model with fitted equations.
        """

        pkl_filename = _csv_filename_to_pkl_filename(equation_file)

        # Try to load model from <equation_file>.pkl
        print(f"Checking if {pkl_filename} exists...")
        if os.path.exists(pkl_filename):
            print(f"Loading model from {pkl_filename}")
            assert binary_operators is None
            assert unary_operators is None
            assert n_features_in is None
            with open(pkl_filename, "rb") as f:
                model = pkl.load(f)
            # Change equation_file_ to be in the same dir as the pickle file
            base_dir = os.path.dirname(pkl_filename)
            base_equation_file = os.path.basename(model.equation_file_)
            model.equation_file_ = os.path.join(base_dir, base_equation_file)

            # Update any parameters if necessary, such as
            # extra_sympy_mappings:
            model.set_params(**pysr_kwargs)
            if "equations_" not in model.__dict__ or model.equations_ is None:
                model.refresh()

            return model

        # Else, we re-create it.
        print(
            f"{pkl_filename} does not exist, "
            "so we must create the model from scratch."
        )
        assert binary_operators is not None or unary_operators is not None
        assert n_features_in is not None

        # TODO: copy .bkup file if exists.
        model = cls(
            equation_file=str(equation_file),
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            **pysr_kwargs,
        )

        model.nout_ = nout
        model.n_features_in_ = n_features_in

        if feature_names_in is None:
            model.feature_names_in_ = np.array([f"x{i}" for i in range(n_features_in)])
            model.display_feature_names_in_ = np.array(
                [f"x{_subscriptify(i)}" for i in range(n_features_in)]
            )
        else:
            assert len(feature_names_in) == n_features_in
            model.feature_names_in_ = feature_names_in
            model.display_feature_names_in_ = feature_names_in

        if selection_mask is None:
            model.selection_mask_ = np.ones(n_features_in, dtype=np.bool_)
        else:
            model.selection_mask_ = selection_mask

        model.refresh(checkpoint_file=equation_file)

        return model

    def __repr__(self):
        """
        Print all current equations fitted by the model.

        The string `>>>>` denotes which equation is selected by the
        `model_selection`.
        """
        if not hasattr(self, "equations_") or self.equations_ is None:
            return "PySRRegressor.equations_ = None"

        output = "PySRRegressor.equations_ = [\n"

        equations = self.equations_
        if not isinstance(equations, list):
            all_equations = [equations]
        else:
            all_equations = equations

        for i, equations in enumerate(all_equations):
            selected = pd.Series([""] * len(equations), index=equations.index)
            chosen_row = idx_model_selection(equations, self.model_selection)
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

    def __getstate__(self):
        """
        Handle pickle serialization for PySRRegressor.

        The Scikit-learn standard requires estimators to be serializable via
        `pickle.dumps()`. However, some attributes do not support pickling
        and need to be hidden, such as the JAX and Torch representations.
        """
        state = self.__dict__
        show_pickle_warning = not (
            "show_pickle_warnings_" in state and not state["show_pickle_warnings_"]
        )
        state_keys_containing_lambdas = ["extra_sympy_mappings", "extra_torch_mappings"]
        for state_key in state_keys_containing_lambdas:
            if state[state_key] is not None and show_pickle_warning:
                warnings.warn(
                    f"`{state_key}` cannot be pickled and will be removed from the "
                    "serialized instance. When loading the model, please redefine "
                    f"`{state_key}` at runtime."
                )
        state_keys_to_clear = state_keys_containing_lambdas
        pickled_state = {
            key: (None if key in state_keys_to_clear else value)
            for key, value in state.items()
        }
        if ("equations_" in pickled_state) and (
            pickled_state["equations_"] is not None
        ):
            pickled_state["output_torch_format"] = False
            pickled_state["output_jax_format"] = False
            if self.nout_ == 1:
                pickled_columns = ~pickled_state["equations_"].columns.isin(
                    ["jax_format", "torch_format"]
                )
                pickled_state["equations_"] = (
                    pickled_state["equations_"].loc[:, pickled_columns].copy()
                )
            else:
                pickled_columns = [
                    ~dataframe.columns.isin(["jax_format", "torch_format"])
                    for dataframe in pickled_state["equations_"]
                ]
                pickled_state["equations_"] = [
                    dataframe.loc[:, signle_pickled_columns]
                    for dataframe, signle_pickled_columns in zip(
                        pickled_state["equations_"], pickled_columns
                    )
                ]
        return pickled_state

    def _checkpoint(self):
        """Save the model's current state to a checkpoint file.

        This should only be used internally by PySRRegressor.
        """
        # Save model state:
        self.show_pickle_warnings_ = False
        with open(_csv_filename_to_pkl_filename(self.equation_file_), "wb") as f:
            pkl.dump(self, f)
        self.show_pickle_warnings_ = True

    @property
    def equations(self):  # pragma: no cover
        warnings.warn(
            "PySRRegressor.equations is now deprecated. "
            "Please use PySRRegressor.equations_ instead.",
            FutureWarning,
        )
        return self.equations_

    @property
    def julia_options_(self):
        """The deserialized julia options."""
        return jl_deserialize(self.julia_options_stream_)

    @property
    def julia_state_(self):
        """The deserialized state."""
        return jl_deserialize(self.julia_state_stream_)

    @property
    def raw_julia_state_(self):
        warnings.warn(
            "PySRRegressor.raw_julia_state_ is now deprecated. "
            "Please use PySRRegressor.julia_state_ instead, or julia_state_stream_ "
            "for the raw stream of bytes.",
            FutureWarning,
        )
        return self.julia_state_

    def get_best(self, index=None) -> Union[pd.Series, List[pd.Series]]:
        """
        Get best equation using `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from `self.equations_`,
            give the row number here. This overrides the `model_selection`
            parameter. If there are multiple output features, then pass
            a list of indices with the order the same as the output feature.

        Returns
        -------
        best_equation : pandas.Series
            Dictionary representing the best expression found.

        Raises
        ------
        NotImplementedError
            Raised when an invalid model selection strategy is provided.
        """
        check_is_fitted(self, attributes=["equations_"])

        if index is not None:
            if isinstance(self.equations_, list):
                assert isinstance(
                    index, list
                ), "With multiple output features, index must be a list."
                return [eq.iloc[i] for eq, i in zip(self.equations_, index)]
            else:
                equations_ = cast(pd.DataFrame, self.equations_)
                return cast(pd.Series, equations_.iloc[index])

        if isinstance(self.equations_, list):
            return [
                cast(pd.Series, eq.loc[idx_model_selection(eq, self.model_selection)])
                for eq in self.equations_
            ]
        else:
            equations_ = cast(pd.DataFrame, self.equations_)
            return cast(
                pd.Series,
                equations_.loc[idx_model_selection(equations_, self.model_selection)],
            )

    def _setup_equation_file(self):
        """
        Set the full pathname of the equation file.

        This is performed using `tempdir` and
        `equation_file`.
        """
        # Cast tempdir string as a Path object
        self.tempdir_ = Path(tempfile.mkdtemp(dir=self.tempdir))
        if self.temp_equation_file:
            self.equation_file_ = self.tempdir_ / "hall_of_fame.csv"
        elif self.equation_file is None:
            if self.warm_start and (
                hasattr(self, "equation_file_") and self.equation_file_
            ):
                pass
            else:
                date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S.%f")[:-3]
                self.equation_file_ = "hall_of_fame_" + date_time + ".csv"
        else:
            self.equation_file_ = self.equation_file
        self.equation_file_contents_ = None

    def _validate_and_modify_params(self) -> _DynamicallySetParams:
        """
        Ensure parameters passed at initialization are valid.

        Also returns a dictionary of parameters to update from their
        values given at initialization.

        Returns
        -------
        packed_modified_params : dict
            Dictionary of parameters to modify from their initialized
            values. For example, default parameters are set here
            when a parameter is left set to `None`.
        """
        # Immutable parameter validation
        # Ensure instance parameters are allowable values:
        if self.tournament_selection_n > self.population_size:
            raise ValueError(
                "`tournament_selection_n` parameter must be smaller than `population_size`."
            )

        if self.maxsize > 40:
            warnings.warn(
                "Note: Using a large maxsize for the equation search will be "
                "exponentially slower and use significant memory."
            )
        elif self.maxsize < 7:
            raise ValueError("PySR requires a maxsize of at least 7")

        if self.deterministic and not (
            self.multithreading in [False, None]
            and self.procs == 0
            and self.random_state is not None
        ):
            raise ValueError(
                "To ensure deterministic searches, you must set `random_state` to a seed, "
                "`procs` to `0`, and `multithreading` to `False` or `None`."
            )

        if self.random_state is not None and (
            not self.deterministic or self.procs != 0
        ):
            warnings.warn(
                "Note: Setting `random_state` without also setting `deterministic` "
                "to True and `procs` to 0 will result in non-deterministic searches. "
            )

        if self.elementwise_loss is not None and self.loss_function is not None:
            raise ValueError(
                "You cannot set both `elementwise_loss` and `loss_function`."
            )

        # NotImplementedError - Values that could be supported at a later time
        if self.optimizer_algorithm not in VALID_OPTIMIZER_ALGORITHMS:
            raise NotImplementedError(
                f"PySR currently only supports the following optimizer algorithms: {VALID_OPTIMIZER_ALGORITHMS}"
            )

        param_container = _DynamicallySetParams(
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[],
            maxdepth=self.maxsize,
            constraints={},
            multithreading=self.procs != 0 and self.cluster_manager is None,
            batch_size=1,
            update_verbosity=int(self.verbosity),
            progress=self.progress,
            warmup_maxsize_by=0.0,
        )

        for param_name in map(lambda x: x.name, fields(_DynamicallySetParams)):
            user_param_value = getattr(self, param_name)
            if user_param_value is None:
                # Leave as the default in DynamicallySetParams
                ...
            else:
                # If user has specified it, we will override the default.
                # However, there are some special cases to mutate it:
                new_param_value = _mutate_parameter(param_name, user_param_value)
                setattr(param_container, param_name, new_param_value)
        # TODO: This should just be part of the __init__ of _DynamicallySetParams

        assert (
            len(param_container.binary_operators) > 0
            or len(param_container.unary_operators) > 0
        ), "At least one operator must be provided."

        return param_container

    def _validate_and_set_fit_params(
        self,
        X,
        y,
        Xresampled,
        weights,
        variable_names,
        complexity_of_variables,
        X_units,
        y_units,
    ) -> Tuple[
        ndarray,
        ndarray,
        Optional[ndarray],
        Optional[ndarray],
        ArrayLike[str],
        Union[int, float, List[Union[int, float]]],
        Optional[ArrayLike[str]],
        Optional[Union[str, ArrayLike[str]]],
    ]:
        """
        Validate the parameters passed to the :term`fit` method.

        This method also sets the `nout_` attribute.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape `(n_samples, n_features)`.
        y : ndarray | pandas.DataFrame}
            Target values of shape `(n_samples,)` or `(n_samples, n_targets)`.
            Will be cast to `X`'s dtype if necessary.
        Xresampled : ndarray | pandas.DataFrame
            Resampled training data used for denoising,
            of shape `(n_resampled, n_features)`.
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `y`.
            Each element is how to weight the mean-square-error loss
            for that particular element of y.
        variable_names : ndarray of length n_features
            Names of each variable in the training dataset, `X`.
        complexity_of_variables : int | float | list[int | float]
            Complexity of each variable in the training dataset, `X`.
        X_units : list[str] of length n_features
            Units of each variable in the training dataset, `X`.
        y_units : str | list[str] of length n_out
            Units of each variable in the training dataset, `y`.

        Returns
        -------
        X_validated : ndarray of shape (n_samples, n_features)
            Validated training data.
        y_validated : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Validated target data.
        Xresampled : ndarray of shape (n_resampled, n_features)
            Validated resampled training data used for denoising.
        variable_names_validated : list[str] of length n_features
            Validated list of variable names for each feature in `X`.
        X_units : list[str] of length n_features
            Validated units for `X`.
        y_units : str | list[str] of length n_out
            Validated units for `y`.

        """
        if isinstance(X, pd.DataFrame):
            if variable_names:
                variable_names = None
                warnings.warn(
                    "`variable_names` has been reset to `None` as `X` is a DataFrame. "
                    "Using DataFrame column names instead."
                )

            if (
                pd.api.types.is_object_dtype(X.columns)
                and X.columns.str.contains(" ").any()
            ):
                X.columns = X.columns.str.replace(" ", "_")
                warnings.warn(
                    "Spaces in DataFrame column names are not supported. "
                    "Spaces have been replaced with underscores. \n"
                    "Please rename the columns to valid names."
                )
        elif variable_names and any([" " in name for name in variable_names]):
            variable_names = [name.replace(" ", "_") for name in variable_names]
            warnings.warn(
                "Spaces in `variable_names` are not supported. "
                "Spaces have been replaced with underscores. \n"
                "Please use valid names instead."
            )

        if (
            complexity_of_variables is not None
            and self.complexity_of_variables is not None
        ):
            raise ValueError(
                "You cannot set `complexity_of_variables` at both `fit` and `__init__`. "
                "Pass it at `__init__` to set it to global default, OR use `fit` to set it for "
                "each variable individually."
            )
        elif complexity_of_variables is not None:
            complexity_of_variables = complexity_of_variables
        elif self.complexity_of_variables is not None:
            complexity_of_variables = self.complexity_of_variables
        else:
            complexity_of_variables = 1

        # Data validation and feature name fetching via sklearn
        # This method sets the n_features_in_ attribute
        if Xresampled is not None:
            Xresampled = check_array(Xresampled)
        if weights is not None:
            weights = check_array(weights, ensure_2d=False)
            check_consistent_length(weights, y)
        X, y = self._validate_data_X_y(X, y)
        self.feature_names_in_ = _safe_check_feature_names_in(
            self, variable_names, generate_names=False
        )

        if self.feature_names_in_ is None:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
            self.display_feature_names_in_ = np.array(
                [f"x{_subscriptify(i)}" for i in range(X.shape[1])]
            )
            variable_names = self.feature_names_in_
        else:
            self.display_feature_names_in_ = self.feature_names_in_
            variable_names = self.feature_names_in_

        # Handle multioutput data
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            y = y.reshape(-1)
        elif len(y.shape) == 2:
            self.nout_ = y.shape[1]
        else:
            raise NotImplementedError("y shape not supported!")

        self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)
        self.X_units_ = copy.deepcopy(X_units)
        self.y_units_ = copy.deepcopy(y_units)

        return (
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        )

    def _validate_data_X_y(self, X, y) -> Tuple[ndarray, ndarray]:
        raw_out = self._validate_data(X=X, y=y, reset=True, multi_output=True)  # type: ignore
        return cast(Tuple[ndarray, ndarray], raw_out)

    def _validate_data_X(self, X) -> Tuple[ndarray]:
        raw_out = self._validate_data(X=X, reset=False)  # type: ignore
        return cast(Tuple[ndarray], raw_out)

    def _pre_transform_training_data(
        self,
        X: ndarray,
        y: ndarray,
        Xresampled: Union[ndarray, None],
        variable_names: ArrayLike[str],
        complexity_of_variables: Union[int, float, List[Union[int, float]]],
        X_units: Union[ArrayLike[str], None],
        y_units: Union[ArrayLike[str], str, None],
        random_state: np.random.RandomState,
    ):
        """
        Transform the training data before fitting the symbolic regressor.

        This method also updates/sets the `selection_mask_` attribute.

        Parameters
        ----------
        X : ndarray
            Training data of shape (n_samples, n_features).
        y : ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).
            Will be cast to X's dtype if necessary.
        Xresampled : ndarray | None
            Resampled training data, of shape `(n_resampled, n_features)`,
            used for denoising.
        variable_names : list[str]
            Names of each variable in the training dataset, `X`.
            Of length `n_features`.
        complexity_of_variables : int | float | list[int | float]
            Complexity of each variable in the training dataset, `X`.
        X_units : list[str]
            Units of each variable in the training dataset, `X`.
        y_units : str | list[str]
            Units of each variable in the training dataset, `y`.
        random_state : int | np.RandomState
            Pass an int for reproducible results across multiple function calls.
            See :term:`Glossary <random_state>`. Default is `None`.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed training data. n_samples will be equal to
            `Xresampled.shape[0]` if `self.denoise` is `True`,
            and `Xresampled is not None`, otherwise it will be
            equal to `X.shape[0]`. n_features will be equal to
            `self.select_k_features` if `self.select_k_features is not None`,
            otherwise it will be equal to `X.shape[1]`
        y_transformed : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Transformed target data. n_samples will be equal to
            `Xresampled.shape[0]` if `self.denoise` is `True`,
            and `Xresampled is not None`, otherwise it will be
            equal to `X.shape[0]`.
        variable_names_transformed : list[str] of length n_features
            Names of each variable in the transformed dataset,
            `X_transformed`.
        X_units_transformed : list[str] of length n_features
            Units of each variable in the transformed dataset.
        y_units_transformed : str | list[str] of length n_out
            Units of each variable in the transformed dataset.
        """
        # Feature selection transformation
        if self.select_k_features:
            selection_mask = run_feature_selection(
                X, y, self.select_k_features, random_state=random_state
            )
            X = X[:, selection_mask]

            if Xresampled is not None:
                Xresampled = Xresampled[:, selection_mask]

            # Reduce variable_names to selection
            variable_names = cast(
                ArrayLike[str],
                [
                    variable_names[i]
                    for i in range(len(variable_names))
                    if selection_mask[i]
                ],
            )

            if isinstance(complexity_of_variables, list):
                complexity_of_variables = [
                    complexity_of_variables[i]
                    for i in range(len(complexity_of_variables))
                    if selection_mask[i]
                ]
                self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)

            if X_units is not None:
                X_units = cast(
                    ArrayLike[str],
                    [X_units[i] for i in range(len(X_units)) if selection_mask[i]],
                )
                self.X_units_ = copy.deepcopy(X_units)

            # Re-perform data validation and feature name updating
            X, y = self._validate_data_X_y(X, y)
            # Update feature names with selected variable names
            self.selection_mask_ = selection_mask
            self.feature_names_in_ = _check_feature_names_in(self, variable_names)
            self.display_feature_names_in_ = self.feature_names_in_
            print(f"Using features {self.feature_names_in_}")

        # Denoising transformation
        if self.denoise:
            if self.nout_ > 1:
                X, y = multi_denoise(
                    X, y, Xresampled=Xresampled, random_state=random_state
                )
            else:
                X, y = denoise(X, y, Xresampled=Xresampled, random_state=random_state)

        return X, y, variable_names, complexity_of_variables, X_units, y_units

    def _run(
        self,
        X: ndarray,
        y: ndarray,
        runtime_params: _DynamicallySetParams,
        weights: Optional[ndarray],
        seed: int,
    ):
        """
        Run the symbolic regression fitting process on the julia backend.

        Parameters
        ----------
        X : ndarray
            Training data of shape `(n_samples, n_features)`.
        y : ndarray
            Target values of shape `(n_samples,)` or `(n_samples, n_targets)`.
            Will be cast to `X`'s dtype if necessary.
        runtime_params : DynamicallySetParams
            Dynamically set versions of some parameters passed in __init__.
        weights : ndarray | None
            Weight array of the same shape as `y`.
            Each element is how to weight the mean-square-error loss
            for that particular element of y.
        seed : int
            Random seed for julia backend process.

        Returns
        -------
        self : object
            Reference to `self` with fitted attributes.

        Raises
        ------
        ImportError
            Raised when the julia backend fails to import a package.
        """
        # Need to be global as we don't want to recreate/reinstate julia for
        # every new instance of PySRRegressor
        global ALREADY_RAN

        # These are the parameters which may be modified from the ones
        # specified in init, so we define them here locally:
        binary_operators = runtime_params.binary_operators
        unary_operators = runtime_params.unary_operators
        maxdepth = runtime_params.maxdepth
        constraints = runtime_params.constraints
        multithreading = runtime_params.multithreading
        batch_size = runtime_params.batch_size
        update_verbosity = runtime_params.update_verbosity
        progress = runtime_params.progress
        warmup_maxsize_by = runtime_params.warmup_maxsize_by

        nested_constraints = self.nested_constraints
        complexity_of_operators = self.complexity_of_operators
        complexity_of_variables = self.complexity_of_variables_
        cluster_manager = self.cluster_manager

        # Start julia backend processes
        if not ALREADY_RAN and update_verbosity != 0:
            print("Compiling Julia backend...")

        if cluster_manager is not None:
            cluster_manager = _load_cluster_manager(cluster_manager)

        # TODO(mcranmer): These functions should be part of this class.
        binary_operators, unary_operators = _maybe_create_inline_operators(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            extra_sympy_mappings=self.extra_sympy_mappings,
        )
        constraints = _process_constraints(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            constraints=constraints,
        )

        una_constraints = [constraints[op] for op in unary_operators]
        bin_constraints = [constraints[op] for op in binary_operators]

        # Parse dict into Julia Dict for nested constraints::
        if nested_constraints is not None:
            nested_constraints_str = "Dict("
            for outer_k, outer_v in nested_constraints.items():
                nested_constraints_str += f"({outer_k}) => Dict("
                for inner_k, inner_v in outer_v.items():
                    nested_constraints_str += f"({inner_k}) => {inner_v}, "
                nested_constraints_str += "), "
            nested_constraints_str += ")"
            nested_constraints = jl.seval(nested_constraints_str)

        # Parse dict into Julia Dict for complexities:
        if complexity_of_operators is not None:
            complexity_of_operators_str = "Dict("
            for k, v in complexity_of_operators.items():
                complexity_of_operators_str += f"({k}) => {v}, "
            complexity_of_operators_str += ")"
            complexity_of_operators = jl.seval(complexity_of_operators_str)
        # TODO: Refactor this into helper function

        if isinstance(complexity_of_variables, list):
            complexity_of_variables = jl_array(complexity_of_variables)

        custom_loss = jl.seval(
            str(self.elementwise_loss)
            if self.elementwise_loss is not None
            else "nothing"
        )
        custom_full_objective = jl.seval(
            str(self.loss_function) if self.loss_function is not None else "nothing"
        )

        early_stop_condition = jl.seval(
            str(self.early_stop_condition)
            if self.early_stop_condition is not None
            else "nothing"
        )

        load_required_packages(
            turbo=self.turbo,
            bumper=self.bumper,
            enable_autodiff=self.enable_autodiff,
            cluster_manager=cluster_manager,
        )

        mutation_weights = SymbolicRegression.MutationWeights(
            mutate_constant=self.weight_mutate_constant,
            mutate_operator=self.weight_mutate_operator,
            swap_operands=self.weight_swap_operands,
            add_node=self.weight_add_node,
            insert_node=self.weight_insert_node,
            delete_node=self.weight_delete_node,
            simplify=self.weight_simplify,
            randomize=self.weight_randomize,
            do_nothing=self.weight_do_nothing,
            optimize=self.weight_optimize,
        )

        jl_binary_operators: List[Any] = []
        jl_unary_operators: List[Any] = []
        for input_list, output_list, name in [
            (binary_operators, jl_binary_operators, "binary"),
            (unary_operators, jl_unary_operators, "unary"),
        ]:
            for op in input_list:
                jl_op = jl.seval(op)
                if not jl_is_function(jl_op):
                    raise ValueError(
                        f"When building `{name}_operators`, `'{op}'` did not return a Julia function"
                    )
                output_list.append(jl_op)

        # Call to Julia backend.
        # See https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/OptionsStruct.jl
        options = SymbolicRegression.Options(
            binary_operators=jl_array(jl_binary_operators, dtype=jl.Function),
            unary_operators=jl_array(jl_unary_operators, dtype=jl.Function),
            bin_constraints=jl_array(bin_constraints),
            una_constraints=jl_array(una_constraints),
            complexity_of_operators=complexity_of_operators,
            complexity_of_constants=self.complexity_of_constants,
            complexity_of_variables=complexity_of_variables,
            nested_constraints=nested_constraints,
            elementwise_loss=custom_loss,
            loss_function=custom_full_objective,
            maxsize=int(self.maxsize),
            output_file=_escape_filename(self.equation_file_),
            npopulations=int(self.populations),
            batching=self.batching,
            batch_size=int(min([batch_size, len(X)]) if self.batching else len(X)),
            mutation_weights=mutation_weights,
            tournament_selection_p=self.tournament_selection_p,
            tournament_selection_n=self.tournament_selection_n,
            # These have the same name:
            parsimony=self.parsimony,
            dimensional_constraint_penalty=self.dimensional_constraint_penalty,
            dimensionless_constants_only=self.dimensionless_constants_only,
            alpha=self.alpha,
            maxdepth=maxdepth,
            fast_cycle=self.fast_cycle,
            turbo=self.turbo,
            bumper=self.bumper,
            enable_autodiff=self.enable_autodiff,
            migration=self.migration,
            hof_migration=self.hof_migration,
            fraction_replaced_hof=self.fraction_replaced_hof,
            should_simplify=self.should_simplify,
            should_optimize_constants=self.should_optimize_constants,
            warmup_maxsize_by=warmup_maxsize_by,
            use_frequency=self.use_frequency,
            use_frequency_in_tournament=self.use_frequency_in_tournament,
            adaptive_parsimony_scaling=self.adaptive_parsimony_scaling,
            npop=self.population_size,
            ncycles_per_iteration=self.ncycles_per_iteration,
            fraction_replaced=self.fraction_replaced,
            topn=self.topn,
            print_precision=self.print_precision,
            optimizer_algorithm=self.optimizer_algorithm,
            optimizer_nrestarts=self.optimizer_nrestarts,
            optimizer_probability=self.optimize_probability,
            optimizer_iterations=self.optimizer_iterations,
            perturbation_factor=self.perturbation_factor,
            annealing=self.annealing,
            timeout_in_seconds=self.timeout_in_seconds,
            crossover_probability=self.crossover_probability,
            skip_mutation_failures=self.skip_mutation_failures,
            max_evals=self.max_evals,
            early_stop_condition=early_stop_condition,
            seed=seed,
            deterministic=self.deterministic,
            define_helper_functions=False,
        )

        self.julia_options_stream_ = jl_serialize(options)

        # Convert data to desired precision
        test_X = np.array(X)
        is_complex = np.issubdtype(test_X.dtype, np.complexfloating)
        is_real = not is_complex
        if is_real:
            np_dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self.precision]
        else:
            np_dtype = {32: np.complex64, 64: np.complex128}[self.precision]

        # This converts the data into a Julia array:
        jl_X = jl_array(np.array(X, dtype=np_dtype).T)
        if len(y.shape) == 1:
            jl_y = jl_array(np.array(y, dtype=np_dtype))
        else:
            jl_y = jl_array(np.array(y, dtype=np_dtype).T)
        if weights is not None:
            if len(weights.shape) == 1:
                jl_weights = jl_array(np.array(weights, dtype=np_dtype))
            else:
                jl_weights = jl_array(np.array(weights, dtype=np_dtype).T)
        else:
            jl_weights = None

        if self.procs == 0 and not multithreading:
            parallelism = "serial"
        elif multithreading:
            parallelism = "multithreading"
        else:
            parallelism = "multiprocessing"

        cprocs = (
            None if parallelism in ["serial", "multithreading"] else int(self.procs)
        )

        if len(y.shape) > 1:
            # We set these manually so that they respect Python's 0 indexing
            # (by default Julia will use y1, y2...)
            jl_y_variable_names = jl_array(
                [f"y{_subscriptify(i)}" for i in range(y.shape[1])]
            )
        else:
            jl_y_variable_names = None

        PythonCall.GC.disable()
        out = SymbolicRegression.equation_search(
            jl_X,
            jl_y,
            weights=jl_weights,
            niterations=int(self.niterations),
            variable_names=jl_array([str(v) for v in self.feature_names_in_]),
            display_variable_names=jl_array(
                [str(v) for v in self.display_feature_names_in_]
            ),
            y_variable_names=jl_y_variable_names,
            X_units=jl_array(self.X_units_),
            y_units=(
                jl_array(self.y_units_)
                if isinstance(self.y_units_, list)
                else self.y_units_
            ),
            options=options,
            numprocs=cprocs,
            parallelism=parallelism,
            saved_state=self.julia_state_,
            return_state=True,
            addprocs_function=cluster_manager,
            heap_size_hint_in_bytes=self.heap_size_hint_in_bytes,
            progress=progress and self.verbosity > 0 and len(y.shape) == 1,
            verbosity=int(self.verbosity),
        )
        PythonCall.GC.enable()

        self.julia_state_stream_ = jl_serialize(out)

        # Set attributes
        self.equations_ = self.get_hof()

        if self.delete_tempfiles:
            shutil.rmtree(self.tempdir_)

        ALREADY_RAN = True

        return self

    def fit(
        self,
        X,
        y,
        Xresampled=None,
        weights=None,
        variable_names: Optional[ArrayLike[str]] = None,
        complexity_of_variables: Optional[
            Union[int, float, List[Union[int, float]]]
        ] = None,
        X_units: Optional[ArrayLike[str]] = None,
        y_units: Optional[Union[str, ArrayLike[str]]] = None,
    ) -> "PySRRegressor":
        """
        Search for equations to fit the dataset and store them in `self.equations_`.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape (n_samples, n_features).
        y : ndarray | pandas.DataFrame
            Target values of shape (n_samples,) or (n_samples, n_targets).
            Will be cast to X's dtype if necessary.
        Xresampled : ndarray | pandas.DataFrame
            Resampled training data, of shape (n_resampled, n_features),
            to generate a denoised data on. This
            will be used as the training data, rather than `X`.
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `y`.
            Each element is how to weight the mean-square-error loss
            for that particular element of `y`. Alternatively,
            if a custom `loss` was set, it will can be used
            in arbitrary ways.
        variable_names : list[str]
            A list of names for the variables, rather than "x0", "x1", etc.
            If `X` is a pandas dataframe, the column names will be used
            instead of `variable_names`. Cannot contain spaces or special
            characters. Avoid variable names which are also
            function names in `sympy`, such as "N".
        X_units : list[str]
            A list of units for each variable in `X`. Each unit should be
            a string representing a Julia expression. See DynamicQuantities.jl
            https://symbolicml.org/DynamicQuantities.jl/dev/units/ for more
            information.
        y_units : str | list[str]
            Similar to `X_units`, but as a unit for the target variable, `y`.
            If `y` is a matrix, a list of units should be passed. If `X_units`
            is given but `y_units` is not, then `y_units` will be arbitrary.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Init attributes that are not specified in BaseEstimator
        if self.warm_start and hasattr(self, "julia_state_stream_"):
            pass
        else:
            if hasattr(self, "julia_state_stream_"):
                warnings.warn(
                    "The discovered expressions are being reset. "
                    "Please set `warm_start=True` if you wish to continue "
                    "to start a search where you left off.",
                )

            self.equations_ = None
            self.nout_ = 1
            self.selection_mask_ = None
            self.julia_state_stream_ = None
            self.julia_options_stream_ = None
            self.complexity_of_variables_ = None
            self.X_units_ = None
            self.y_units_ = None

        self._setup_equation_file()

        runtime_params = self._validate_and_modify_params()

        (
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        ) = self._validate_and_set_fit_params(
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        )

        if X.shape[0] > 10000 and not self.batching:
            warnings.warn(
                "Note: you are running with more than 10,000 datapoints. "
                "You should consider turning on batching (https://astroautomata.com/PySR/options/#batching). "
                "You should also reconsider if you need that many datapoints. "
                "Unless you have a large amount of noise (in which case you "
                "should smooth your dataset first), generally < 10,000 datapoints "
                "is enough to find a functional form with symbolic regression. "
                "More datapoints will lower the search speed."
            )

        random_state = check_random_state(self.random_state)  # For np random
        seed = cast(int, random_state.randint(0, 2**31 - 1))  # For julia random

        # Pre transformations (feature selection and denoising)
        X, y, variable_names, complexity_of_variables, X_units, y_units = (
            self._pre_transform_training_data(
                X,
                y,
                Xresampled,
                variable_names,
                complexity_of_variables,
                X_units,
                y_units,
                random_state,
            )
        )

        # Warn about large feature counts (still warn if feature count is large
        # after running feature selection)
        if self.n_features_in_ >= 10:
            warnings.warn(
                "Note: you are running with 10 features or more. "
                "Genetic algorithms like used in PySR scale poorly with large numbers of features. "
                "You should run PySR for more `niterations` to ensure it can find "
                "the correct variables, "
                "or, alternatively, do a dimensionality reduction beforehand. "
                "For example, `X = PCA(n_components=6).fit_transform(X)`, "
                "using scikit-learn's `PCA` class, "
                "will reduce the number of features to 6 in an interpretable way, "
                "as each resultant feature "
                "will be a linear combination of the original features. "
            )

        # Assertion checks
        use_custom_variable_names = variable_names is not None
        # TODO: this is always true.

        _check_assertions(
            X,
            use_custom_variable_names,
            variable_names,
            complexity_of_variables,
            weights,
            y,
            X_units,
            y_units,
        )

        # Initially, just save model parameters, so that
        # it can be loaded from an early exit:
        if not self.temp_equation_file:
            self._checkpoint()

        # Perform the search:
        self._run(X, y, runtime_params, weights=weights, seed=seed)

        # Then, after fit, we save again, so the pickle file contains
        # the equations:
        if not self.temp_equation_file:
            self._checkpoint()

        return self

    def refresh(self, checkpoint_file: Optional[PathLike] = None) -> None:
        """
        Update self.equations_ with any new options passed.

        For example, updating `extra_sympy_mappings`
        will require a `.refresh()` to update the equations.

        Parameters
        ----------
        checkpoint_file : str or Path
            Path to checkpoint hall of fame file to be loaded.
            The default will use the set `equation_file_`.
        """
        if checkpoint_file is not None:
            self.equation_file_ = checkpoint_file
            self.equation_file_contents_ = None
        check_is_fitted(self, attributes=["equation_file_"])
        self.equations_ = self.get_hof()

    def predict(self, X, index=None):
        """
        Predict y from input X using the equation chosen by `model_selection`.

        You may see what equation is used by printing this object. X should
        have the same columns as the training data.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape `(n_samples, n_features)`.
        index : int | list[int]
            If you want to compute the output of an expression using a
            particular row of `self.equations_`, you may specify the index here.
            For multiple output equations, you must pass a list of indices
            in the same order.

        Returns
        -------
        y_predicted : ndarray of shape (n_samples, nout_)
            Values predicted by substituting `X` into the fitted symbolic
            regression model.

        Raises
        ------
        ValueError
            Raises if the `best_equation` cannot be evaluated.
        """
        check_is_fitted(
            self, attributes=["selection_mask_", "feature_names_in_", "nout_"]
        )
        best_equation = self.get_best(index=index)

        # When X is an numpy array or a pandas dataframe with a RangeIndex,
        # the self.feature_names_in_ generated during fit, for the same X,
        # will cause a warning to be thrown during _validate_data.
        # To avoid this, convert X to a dataframe, apply the selection mask,
        # and then set the column/feature_names of X to be equal to those
        # generated during fit.
        if not isinstance(X, pd.DataFrame):
            X = check_array(X)
            X = pd.DataFrame(X)
        if isinstance(X.columns, pd.RangeIndex):
            if self.selection_mask_ is not None:
                # RangeIndex enforces column order allowing columns to
                # be correctly filtered with self.selection_mask_
                X = X[X.columns[self.selection_mask_]]
            X.columns = self.feature_names_in_
        # Without feature information, CallableEquation/lambda_format equations
        # require that the column order of X matches that of the X used during
        # the fitting process. _validate_data removes this feature information
        # when it converts the dataframe to an np array. Thus, to ensure feature
        # order is preserved after conversion, the dataframe columns must be
        # reordered/reindexed to match those of the transformed (denoised and
        # feature selected) X in fit.
        X = X.reindex(columns=self.feature_names_in_)
        X = self._validate_data_X(X)

        try:
            if isinstance(best_equation, list):
                assert self.nout_ > 1
                return np.stack(
                    [eq["lambda_format"](X) for eq in best_equation], axis=1
                )
            else:
                return best_equation["lambda_format"](X)
        except Exception as error:
            raise ValueError(
                "Failed to evaluate the expression. "
                "If you are using a custom operator, make sure to define it in `extra_sympy_mappings`, "
                "e.g., `model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})`, where "
                "`lambda x: 1/x` is a valid SymPy function defining the operator. "
                "You can then run `model.refresh()` to re-load the expressions."
            ) from error

    def sympy(self, index=None):
        """
        Return sympy representation of the equation(s) chosen by `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : str, list[str] of length nout_
            SymPy representation of the best equation.
        """
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            assert self.nout_ > 1
            return [eq["sympy_format"] for eq in best_equation]
        else:
            return best_equation["sympy_format"]

    def latex(self, index=None, precision=3):
        """
        Return latex representation of the equation(s) chosen by `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.
        precision : int
            The number of significant figures shown in the LaTeX
            representation.
            Default is `3`.

        Returns
        -------
        best_equation : str or list[str] of length nout_
            LaTeX expression of the best equation.
        """
        self.refresh()
        sympy_representation = self.sympy(index=index)
        if self.nout_ > 1:
            output = []
            for s in sympy_representation:
                latex = sympy2latex(s, prec=precision)
                output.append(latex)
            return output
        return sympy2latex(sympy_representation, prec=precision)

    def jax(self, index=None):
        """
        Return jax representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a dictionary
        containing {"callable": func, "parameters": params}. To call `func`, pass
        func(X, params). This function is differentiable using `jax.grad`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : dict[str, Any]
            Dictionary of callable jax function in "callable" key,
            and jax array of parameters as "parameters" key.
        """
        self.set_params(output_jax_format=True)
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            assert self.nout_ > 1
            return [eq["jax_format"] for eq in best_equation]
        else:
            return best_equation["jax_format"]

    def pytorch(self, index=None):
        """
        Return pytorch representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a PyTorch module
        containing the parameters as trainable attributes. You can use the module like
        any other PyTorch module: `module(X)`, where `X` is a tensor with the same
        column ordering as trained with.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : torch.nn.Module
            PyTorch module representing the expression.
        """
        self.set_params(output_torch_format=True)
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            return [eq["torch_format"] for eq in best_equation]
        else:
            return best_equation["torch_format"]

    def _read_equation_file(self):
        """Read the hall of fame file created by `SymbolicRegression.jl`."""

        try:
            if self.nout_ > 1:
                all_outputs = []
                for i in range(1, self.nout_ + 1):
                    cur_filename = str(self.equation_file_) + f".out{i}" + ".bkup"
                    if not os.path.exists(cur_filename):
                        cur_filename = str(self.equation_file_) + f".out{i}"
                    with open(cur_filename, "r", encoding="utf-8") as f:
                        buf = f.read()
                    buf = _preprocess_julia_floats(buf)

                    df = self._postprocess_dataframe(pd.read_csv(StringIO(buf)))

                    all_outputs.append(df)
            else:
                filename = str(self.equation_file_) + ".bkup"
                if not os.path.exists(filename):
                    filename = str(self.equation_file_)
                with open(filename, "r", encoding="utf-8") as f:
                    buf = f.read()
                buf = _preprocess_julia_floats(buf)
                all_outputs = [self._postprocess_dataframe(pd.read_csv(StringIO(buf)))]

        except FileNotFoundError:
            raise RuntimeError(
                "Couldn't find equation file! The equation search likely exited "
                "before a single iteration completed."
            )
        return all_outputs

    def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "Complexity": "complexity",
                "Loss": "loss",
                "Equation": "equation",
            },
        )

        return df

    def get_hof(self):
        """Get the equations from a hall of fame file.

        If no arguments entered, the ones used
        previously from a call to PySR will be used.
        """
        check_is_fitted(
            self,
            attributes=[
                "nout_",
                "equation_file_",
                "selection_mask_",
                "feature_names_in_",
            ],
        )
        if (
            not hasattr(self, "equation_file_contents_")
        ) or self.equation_file_contents_ is None:
            self.equation_file_contents_ = self._read_equation_file()

        # It is expected extra_jax/torch_mappings will be updated after fit.
        # Thus, validation is performed here instead of in _validate_init_params
        extra_jax_mappings = self.extra_jax_mappings
        extra_torch_mappings = self.extra_torch_mappings
        if extra_jax_mappings is not None:
            for value in extra_jax_mappings.values():
                if not isinstance(value, str):
                    raise ValueError(
                        "extra_jax_mappings must have keys that are strings! "
                        "e.g., {sympy.sqrt: 'jnp.sqrt'}."
                    )
        else:
            extra_jax_mappings = {}
        if extra_torch_mappings is not None:
            for value in extra_torch_mappings.values():
                if not callable(value):
                    raise ValueError(
                        "extra_torch_mappings must be callable functions! "
                        "e.g., {sympy.sqrt: torch.sqrt}."
                    )
        else:
            extra_torch_mappings = {}

        ret_outputs = []

        equation_file_contents = copy.deepcopy(self.equation_file_contents_)

        for output in equation_file_contents:
            scores = []
            lastMSE = None
            lastComplexity = 0
            sympy_format = []
            lambda_format = []
            jax_format = []
            torch_format = []

            for _, eqn_row in output.iterrows():
                eqn = pysr2sympy(
                    eqn_row["equation"],
                    feature_names_in=self.feature_names_in_,
                    extra_sympy_mappings=self.extra_sympy_mappings,
                )
                sympy_format.append(eqn)

                # NumPy:
                sympy_symbols = create_sympy_symbols(self.feature_names_in_)
                lambda_format.append(
                    sympy2numpy(
                        eqn,
                        sympy_symbols,
                        selection=self.selection_mask_,
                    )
                )

                # JAX:
                if self.output_jax_format:
                    func, params = sympy2jax(
                        eqn,
                        sympy_symbols,
                        selection=self.selection_mask_,
                        extra_jax_mappings=self.extra_jax_mappings,
                    )
                    jax_format.append({"callable": func, "parameters": params})

                # Torch:
                if self.output_torch_format:
                    module = sympy2torch(
                        eqn,
                        sympy_symbols,
                        selection=self.selection_mask_,
                        extra_torch_mappings=self.extra_torch_mappings,
                    )
                    torch_format.append(module)

                curMSE = eqn_row["loss"]
                curComplexity = eqn_row["complexity"]

                if lastMSE is None:
                    cur_score = 0.0
                else:
                    if curMSE > 0.0:
                        # TODO Move this to more obvious function/file.
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

        if self.nout_ > 1:
            return ret_outputs
        return ret_outputs[0]

    def latex_table(
        self,
        indices=None,
        precision=3,
        columns=["equation", "complexity", "loss", "score"],
    ):
        """Create a LaTeX/booktabs table for all, or some, of the equations.

        Parameters
        ----------
        indices : list[int] | list[list[int]]
            If you wish to select a particular subset of equations from
            `self.equations_`, give the row numbers here. By default,
            all equations will be used. If there are multiple output
            features, then pass a list of lists.
        precision : int
            The number of significant figures shown in the LaTeX
            representations.
            Default is `3`.
        columns : list[str]
            Which columns to include in the table.
            Default is `["equation", "complexity", "loss", "score"]`.

        Returns
        -------
        latex_table_str : str
            A string that will render a table in LaTeX of the equations.
        """
        self.refresh()

        if isinstance(self.equations_, list):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], list)
                assert len(indices) == self.nout_

            table_string = sympy2multilatextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        elif isinstance(self.equations_, pd.DataFrame):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], int)

            table_string = sympy2latextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        else:
            raise ValueError(
                "Invalid type for equations_ to pass to `latex_table`. "
                "Expected a DataFrame or a list of DataFrames."
            )

        return with_preamble(table_string)


def idx_model_selection(equations: pd.DataFrame, model_selection: str):
    """Select an expression and return its index."""
    if model_selection == "accuracy":
        chosen_idx = equations["loss"].idxmin()
    elif model_selection == "best":
        threshold = 1.5 * equations["loss"].min()
        filtered_equations = equations.query(f"loss <= {threshold}")
        chosen_idx = filtered_equations["score"].idxmax()
    elif model_selection == "score":
        chosen_idx = equations["score"].idxmax()
    else:
        raise NotImplementedError(
            f"{model_selection} is not a valid model selection strategy."
        )
    return chosen_idx


def _mutate_parameter(param_name: str, param_value):
    if param_name in ["binary_operators", "unary_operators"] and isinstance(
        param_value, str
    ):
        return [param_value]

    if param_name == "batch_size" and param_value < 1:
        warnings.warn(
            "Given `batch_size` must be greater than or equal to one. "
            "`batch_size` has been increased to equal one."
        )
        return 1

    if (
        param_name == "progress"
        and param_value == True
        and "buffer" not in sys.stdout.__dir__()
    ):
        warnings.warn(
            "Note: it looks like you are running in Jupyter. "
            "The progress bar will be turned off."
        )
        return False

    return param_value
