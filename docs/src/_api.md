# PySRRegressor Reference

`PySRRegressor` has many options for controlling a symbolic regression search.
Let's look at them below.

## PySRRegressor Parameters
### The Algorithm

#### Creating the Search Space

  - **`binary_operators`**

    List of strings for binary operators used in the search.
    See the [operators page](https://ai.damtp.cam.ac.uk/PySR/operators/)
    for more details.

    *Default:* `["+", "-", "*", "/"]`

  - **`unary_operators`**

    Operators which only take a single scalar as input.
    For example, `"cos"` or `"exp"`.

    *Default:* `None`

  - **`maxsize`**

    Max complexity of an equation.

    *Default:* `20`

  - **`maxdepth`**

    Max depth of an equation. You can use both `maxsize` and
    `maxdepth`. `maxdepth` is by default not used.

    *Default:* `None`

#### Setting the Search Size

  - **`niterations`**

    Number of iterations of the algorithm to run. The best
    equations are printed and migrate between populations at the
    end of each iteration.

    *Default:* `40`

  - **`populations`**

    Number of populations running.

    *Default:* `15`

  - **`population_size`**

    Number of individuals in each population.

    *Default:* `33`

  - **`ncycles_per_iteration`**

    Number of total mutations to run, per 10 samples of the
    population, per iteration.

    *Default:* `550`

#### The Objective

  - **`elementwise_loss`**

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

    *Default:* `"L2DistLoss()"`

  - **`loss_function`**

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

    *Default:* `None`

  - **`model_selection`**

    Model selection criterion when selecting a final expression from
    the list of best expression at each complexity.
    Can be `'accuracy'`, `'best'`, or `'score'`.
    `'accuracy'` selects the candidate model with the lowest loss
    (highest accuracy).
    `'score'` selects the candidate model with the highest score.
    Score is defined as the negated derivative of the log-loss with
    respect to complexity - if an expression has a much better
    loss at a slightly higher complexity, it is preferred.
    `'best'` selects the candidate model with the highest score
    among expressions with a loss better than at least 1.5x the
    most accurate model.

    *Default:* `'best'`

  - **`dimensional_constraint_penalty`**

    Additive penalty for if dimensional analysis of an expression fails.
    By default, this is `1000.0`.

#### Working with Complexities

  - **`parsimony`**

    Multiplicative factor for how much to punish complexity.

    *Default:* `0.0032`

  - **`constraints`**

    Dictionary of int (unary) or 2-tuples (binary), this enforces
    maxsize constraints on the individual arguments of operators.
    E.g., `'pow': (-1, 1)` says that power laws can have any
    complexity left argument, but only 1 complexity in the right
    argument. Use this to force more interpretable solutions.

    *Default:* `None`

  - **`nested_constraints`**

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

    *Default:* `None`

  - **`complexity_of_operators`**

    If you would like to use a complexity other than 1 for an
    operator, specify the complexity here. For example,
    `{"sin": 2, "+": 1}` would give a complexity of 2 for each use
    of the `sin` operator, and a complexity of 1 for each use of
    the `+` operator (which is the default). You may specify real
    numbers for a complexity, and the total complexity of a tree
    will be rounded to the nearest integer after computing.

    *Default:* `None`

  - **`complexity_of_constants`**

    Complexity of constants.

    *Default:* `1`

  - **`complexity_of_variables`**

    Complexity of variables.

    *Default:* `1`

  - **`warmup_maxsize_by`**

    Whether to slowly increase max size from a small number up to
    the maxsize (if greater than 0).  If greater than 0, says the
    fraction of training time at which the current maxsize will
    reach the user-passed maxsize.

    *Default:* `0.0`

  - **`use_frequency`**

    Whether to measure the frequency of complexities, and use that
    instead of parsimony to explore equation space. Will naturally
    find equations of all complexities.

    *Default:* `True`

  - **`use_frequency_in_tournament`**

    Whether to use the frequency mentioned above in the tournament,
    rather than just the simulated annealing.

    *Default:* `True`

  - **`adaptive_parsimony_scaling`**

    If the adaptive parsimony strategy (`use_frequency` and
    `use_frequency_in_tournament`), this is how much to (exponentially)
    weight the contribution. If you find that the search is only optimizing
    the most complex expressions while the simpler expressions remain stagnant,
    you should increase this value.

    *Default:* `20.0`

  - **`should_simplify`**

    Whether to use algebraic simplification in the search. Note that only
    a few simple rules are implemented.

    *Default:* `True`

#### Mutations

  - **`weight_add_node`**

    Relative likelihood for mutation to add a node.

    *Default:* `0.79`

  - **`weight_insert_node`**

    Relative likelihood for mutation to insert a node.

    *Default:* `5.1`

  - **`weight_delete_node`**

    Relative likelihood for mutation to delete a node.

    *Default:* `1.7`

  - **`weight_do_nothing`**

    Relative likelihood for mutation to leave the individual.

    *Default:* `0.21`

  - **`weight_mutate_constant`**

    Relative likelihood for mutation to change the constant slightly
    in a random direction.

    *Default:* `0.048`

  - **`weight_mutate_operator`**

    Relative likelihood for mutation to swap an operator.

    *Default:* `0.47`

  - **`weight_swap_operands`**

    Relative likehood for swapping operands in binary operators.

    *Default:* `0.0`

  - **`weight_randomize`**

    Relative likelihood for mutation to completely delete and then
    randomly generate the equation

    *Default:* `0.00023`

  - **`weight_simplify`**

    Relative likelihood for mutation to simplify constant parts by evaluation

    *Default:* `0.0020`

  - **`weight_optimize`**

    Constant optimization can also be performed as a mutation, in addition to
    the normal strategy controlled by `optimize_probability` which happens
    every iteration. Using it as a mutation is useful if you want to use
    a large `ncycles_periteration`, and may not optimize very often.

    *Default:* `0.0`

  - **`crossover_probability`**

    Absolute probability of crossover-type genetic operation, instead of a mutation.

    *Default:* `0.066`

  - **`annealing`**

    Whether to use annealing.

    *Default:* `False`

  - **`alpha`**

    Initial temperature for simulated annealing
    (requires `annealing` to be `True`).

    *Default:* `0.1`

  - **`perturbation_factor`**

    Constants are perturbed by a max factor of
    (perturbation_factor*T + 1). Either multiplied by this or
    divided by this.

    *Default:* `0.076`

  - **`skip_mutation_failures`**

    Whether to skip mutation and crossover failures, rather than
    simply re-sampling the current member.

    *Default:* `True`

#### Tournament Selection

  - **`tournament_selection_n`**

    Number of expressions to consider in each tournament.

    *Default:* `10`

  - **`tournament_selection_p`**

    Probability of selecting the best expression in each
    tournament. The probability will decay as p*(1-p)^n for other
    expressions, sorted by loss.

    *Default:* `0.86`

#### Constant Optimization

  - **`optimizer_algorithm`**

    Optimization scheme to use for optimizing constants. Can currently
    be `NelderMead` or `BFGS`.

    *Default:* `"BFGS"`

  - **`optimizer_nrestarts`**

    Number of time to restart the constants optimization process with
    different initial conditions.

    *Default:* `2`

  - **`optimize_probability`**

    Probability of optimizing the constants during a single iteration of
    the evolutionary algorithm.

    *Default:* `0.14`

  - **`optimizer_iterations`**

    Number of iterations that the constants optimizer can take.

    *Default:* `8`

  - **`should_optimize_constants`**

    Whether to numerically optimize constants (Nelder-Mead/Newton)
    at the end of each iteration.

    *Default:* `True`

#### Migration between Populations

  - **`fraction_replaced`**

    How much of population to replace with migrating equations from
    other populations.

    *Default:* `0.000364`

  - **`fraction_replaced_hof`**

    How much of population to replace with migrating equations from
    hall of fame.

    *Default:* `0.035`

  - **`migration`**

    Whether to migrate.

    *Default:* `True`

  - **`hof_migration`**

    Whether to have the hall of fame migrate.

    *Default:* `True`

  - **`topn`**

    How many top individuals migrate from each population.

    *Default:* `12`

### Data Preprocessing

  - **`denoise`**

    Whether to use a Gaussian Process to denoise the data before
    inputting to PySR. Can help PySR fit noisy data.

    *Default:* `False`

  - **`select_k_features`**

    Whether to run feature selection in Python using random forests,
    before passing to the symbolic regression code. None means no
    feature selection; an int means select that many features.

    *Default:* `None`

### Stopping Criteria

  - **`max_evals`**

    Limits the total number of evaluations of expressions to
    this number.

    *Default:* `None`

  - **`timeout_in_seconds`**

    Make the search return early once this many seconds have passed.

    *Default:* `None`

  - **`early_stop_condition`**

    Stop the search early if this loss is reached. You may also
    pass a string containing a Julia function which
    takes a loss and complexity as input, for example:
    `"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`.

    *Default:* `None`

### Performance and Parallelization

  - **`procs`**

    Number of processes (=number of populations running).

    *Default:* `cpu_count()`

  - **`multithreading`**

    Use multithreading instead of distributed backend.
    Using procs=0 will turn off both.

    *Default:* `True`

  - **`cluster_manager`**

    For distributed computing, this sets the job queue system. Set
    to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or
    "htc". If set to one of these, PySR will run in distributed
    mode, and use `procs` to figure out how many processes to launch.

    *Default:* `None`

  - **`heap_size_hint_in_bytes`**

    For multiprocessing, this sets the `--heap-size-hint` parameter
    for new Julia processes. This can be configured when using
    multi-node distributed compute, to give a hint to each process
    about how much memory they can use before aggressive garbage
    collection.

  - **`batching`**

    Whether to compare population members on small batches during
    evolution. Still uses full dataset for comparing against hall
    of fame.

    *Default:* `False`

  - **`batch_size`**

    The amount of data to use if doing batching.

    *Default:* `50`

  - **`precision`**

    What precision to use for the data. By default this is `32`
    (float32), but you can select `64` or `16` as well, giving
    you 64 or 16 bits of floating point precision, respectively.
    If you pass complex data, the corresponding complex precision
    will be used (i.e., `64` for complex128, `32` for complex64).

    *Default:* `32`

  - **`fast_cycle`**

    Batch over population subsamples. This is a slightly different
    algorithm than regularized evolution, but does cycles 15%
    faster. May be algorithmically less efficient.

    *Default:* `False`

  - **`turbo`**

    (Experimental) Whether to use LoopVectorization.jl to speed up the
    search evaluation. Certain operators may not be supported.
    Does not support 16-bit precision floats.

    *Default:* `False`

  - **`enable_autodiff`**

    Whether to create derivative versions of operators for automatic
    differentiation. This is only necessary if you wish to compute
    the gradients of an expression within a custom loss function.

    *Default:* `False`

### Determinism

  - **`random_state`**

    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

    *Default:* `None`

  - **`deterministic`**

    Make a PySR search give the same result every run.
    To use this, you must turn off parallelism
    (with `procs`=0, `multithreading`=False),
    and set `random_state` to a fixed seed.

    *Default:* `False`

  - **`warm_start`**

    Tells fit to continue from where the last call to fit finished.
    If false, each call to fit will be fresh, overwriting previous results.

    *Default:* `False`

### Monitoring

  - **`verbosity`**

    What verbosity level to use. 0 means minimal print statements.

    *Default:* `1`

  - **`update_verbosity`**

    What verbosity level to use for package updates.
    Will take value of `verbosity` if not given.

    *Default:* `None`

  - **`print_precision`**

    How many significant digits to print for floats.

    *Default:* `5`

  - **`progress`**

    Whether to use a progress bar instead of printing to stdout.

    *Default:* `True`

### Environment

  - **`temp_equation_file`**

    Whether to put the hall of fame file in the temp directory.
    Deletion is then controlled with the `delete_tempfiles`
    parameter.

    *Default:* `False`

  - **`tempdir`**

    directory for the temporary files.

    *Default:* `None`

  - **`delete_tempfiles`**

    Whether to delete the temporary files after finishing.

    *Default:* `True`

  - **`update`**

    Whether to automatically update Julia packages when `fit` is called.
    You should make sure that PySR is up-to-date itself first, as
    the packaged Julia packages may not necessarily include all
    updated dependencies.

    *Default:* `False`

### Exporting the Results

  - **`equation_file`**

    Where to save the files (.csv extension).

    *Default:* `None`

  - **`output_jax_format`**

    Whether to create a 'jax_format' column in the output,
    containing jax-callable functions and the default parameters in
    a jax array.

    *Default:* `False`

  - **`output_torch_format`**

    Whether to create a 'torch_format' column in the output,
    containing a torch module with trainable parameters.

    *Default:* `False`

  - **`extra_sympy_mappings`**

    Provides mappings between custom `binary_operators` or
    `unary_operators` defined in julia strings, to those same
    operators defined in sympy.
    E.G if `unary_operators=["inv(x)=1/x"]`, then for the fitted
    model to be export to sympy, `extra_sympy_mappings`
    would be `{"inv": lambda x: 1/x}`.

    *Default:* `None`

  - **`extra_torch_mappings`**

    The same as `extra_jax_mappings` but for model export
    to pytorch. Note that the dictionary keys should be callable
    pytorch expressions.
    For example: `extra_torch_mappings={sympy.sin: torch.sin}`.

    *Default:* `None`

  - **`extra_jax_mappings`**

    Similar to `extra_sympy_mappings` but for model export
    to jax. The dictionary maps sympy functions to jax functions.
    For example: `extra_jax_mappings={sympy.sin: "jnp.sin"}` maps
    the `sympy.sin` function to the equivalent jax expression `jnp.sin`.

    *Default:* `None`

## PySRRegressor Functions

::: pysr.PySRRegressor.fit
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.predict
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.from_file
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.sympy
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.latex
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.pytorch
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.jax
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.latex_table
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.refresh
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false
