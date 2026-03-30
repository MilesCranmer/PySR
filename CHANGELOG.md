# Changelog

## [1.5.9] (2025-07-15)

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/MilesCranmer/PySR/pull/853
* Fix type error in feature selection code by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/952
* chore(deps): update juliacall requirement from <0.9.26,>=0.9.24 to >=0.9.24,<0.9.27 by @dependabot[bot] in https://github.com/MilesCranmer/PySR/pull/980


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.8...v1.5.9

## [1.5.8] (2025-05-20)

### What's Changed
* fix: compat with python 3.8 by removing beartype by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/935
* ci: update workflows to test 3.13 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/929
* style: fix newline in warning by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/931
* ci: switch to codecov by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/932
* deps: fix local conda env versions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/933


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.7...v1.5.8

## [1.5.7] (2025-05-19)

### What's Changed
* Enable negative losses by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/928
* Recommend TemplateExpressionSpec over ParametricExpressionSpec @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/920
* Fix multi-output template expressions by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/921
* build: switch to hatchling by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/888
* chore(deps): bump juliacall from 0.9.24 to 0.9.25 by @dependabot in https://github.com/MilesCranmer/PySR/pull/925
* fix: turn off double warning for ParametricExpressionSpec by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/930


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.6...v1.5.7

## [1.5.6] (2025-05-04)

### What's Changed
* Added paper contribution and image by @manuel-morales-a in https://github.com/MilesCranmer/PySR/pull/824
* fix: pickling of inv by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/910
* Automated update to backend: v1.10.0 by @github-actions in https://github.com/MilesCranmer/PySR/pull/890

### New Contributors
* @manuel-morales-a made their first contribution in https://github.com/MilesCranmer/PySR/pull/824

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.5...v1.5.6

## [1.5.5] (2025-04-02)

### What's Changed
* fix: typing extensions dependency by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/885


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.4...v1.5.5

## [1.5.4] (2025-04-01)

### What's Changed
* Compat with older Python by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/884


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.3...v1.5.4

## [1.5.3] (2025-03-28)

### What's Changed
* fix: change sympy mappings ordering by @romanovzky in https://github.com/MilesCranmer/PySR/pull/868

### New Contributors
* @romanovzky made their first contribution in https://github.com/MilesCranmer/PySR/pull/868

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.2...v1.5.3

## [1.5.2] (2025-03-05)

### What's Changed
* fix: mapping of cbrt by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/858


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.1...v1.5.2

## [1.5.1] (2025-03-01)

### What's Changed
* fix: comparison operator parsing by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/845


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.5.0...v1.5.1

## [1.5.0] (2025-02-25)

### Backend Changes

#### Major Changes

* Change behavior of batching to resample only every iteration; not every eval in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/421
  * This result in a speed improvement for code with `batching=true`
  * It should also result in improved search results with batching, because comparison within a single population is more stable during evolution. In other words, there is no _lucky batch_ phenomenon.
  * This also refactors the batching interface to be cleaner. There is a `SubDataset <: Dataset` rather than passing around an array `idx` explicitly.
  * Note that other than the slight behaviour change, this is otherwise backwards compatible - the old way to write custom loss functions that take `idx` will still be handled.

#### Other changes

* feat: better error for mismatched eltypes by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/414
* CompatHelper: bump compat for Optim to 1, (keep existing compat) by @github-actions in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/403
* feat: explicitly monitor errors in workers by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/417
* feat: allow recording crossovers by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/415
* add script for converting record to graphml by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/416
* ci: redistribute part 1 of test suite by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/424
* refactor: rename to `.cost` by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/423
* fix: batched dataset for optimisation by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/426
* refactor: task local storage instead of thread local by @MilesCranmer in https://github.com/MilesCranmer/SymbolicRegression.jl/pull/427

### Frontend Changes

* Update backend to v1.8.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/833
* test: update deprecated sklearn test syntax by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/834
* chore(deps): bump juliacall from 0.9.23 to 0.9.24 by @dependabot in https://github.com/MilesCranmer/PySR/pull/815
* use standard library logging by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/835
* Remove warning about many features, as not really relevant anymore by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/837
* chore(deps): update beartype requirement from <0.20,>=0.19 to >=0.19,<0.21 by @dependabot in https://github.com/MilesCranmer/PySR/pull/838
* chore(deps): update jax[cpu] requirement from <0.5,>=0.4 to >=0.4,<0.6 by @dependabot in https://github.com/MilesCranmer/PySR/pull/810


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.4.0...v1.5.0

## [1.4.0] (2025-02-13)

### What's Changed

[#823](https://github.com/MilesCranmer/PySR/pull/823) adds support for _parameters in template expressions_, allowing you to learn expressions under a template, that have custom coefficients which can be optimized.

Along with this, the `TemplateExpressionSpec` API has changed. (The old API will continue to function, but will not have parametric expressions available).

```python
spec = TemplateExpressionSpec(
    "fx = f(x); p[1] + p[2] * fx + p[3] * fx^2",
    expressions=["f"],
    variable_names=["x"],
    parameters={"p": 3},
)
```

This would learn three parameters, for the expression $y = p_1 + p_2 f(x) + p_3 f(x)^2.$

You can have multiple parameter vectors, and these parameter vectors can also be indexed by categorical features. For example:

```python
### Learn different parameters for each class:
spec = TemplateExpressionSpec(
    "p1[category] * f(x1, x2) + p2[1] * g(x1^2)",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "category"],
    parameters={"p1": 3, "p2": 1},
)
```

This will learn an equation of the form:
$$y = \alpha_c\,f(x_1,x_2) + \beta g(x_1 ^2)$$
where $c$ is the category, $\alpha_c$ is a learned parameter specific to each category, and $\beta$ is a normal scalar category. Note that **unlike ParametricExpressionSpec**, this feature of TemplateExpressionSpec would have you pass the `category` variable _in_ `X` rather than as a category keyword (floating point versions of the categories). This difference means that in a TemplateExpressionSpec, you can actually have _multiple_ categories!

* Added support for expression-level loss functions via `loss_function_expression`, which allows you to specify custom loss functions that operate on the full expression object rather than just its evaluated output. This is particularly useful when working with template expressions.

* Note that the old template expression syntax using function-style definitions is deprecated. Use the new, cleaner syntax instead:

```python
### # Old:
### spec = TemplateExpressionSpec(
###     function_symbols=["f", "g"],
###     combine="((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)"
### )

### New:
spec = TemplateExpressionSpec(
    "sin(f(x1, x2)) + g(x3)"
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
)
```


**Full Changelog:** [v1.3.1...v1.4.0](https://github.com/MilesCranmer/PySR/compare/v1.3.1...v1.4.0)

## [1.3.1] (2024-12-27)

### What's Changed
* Automated update to backend: v1.5.1 by @github-actions in https://github.com/MilesCranmer/PySR/pull/790


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.3.0...v1.3.1

## [1.3.0] (2024-12-15)

### What's Changed

- Expanded support for differential operators via backend 1.5.0 by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/782

e.g., say we wish to integrate $\frac{1}{x^2 \sqrt{x^2 - 1}}$ for $x > 1$:

```python
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec

x = np.random.uniform(1, 10, (1000,))  # Integrand sampling points
y = 1 / (x**2 * np.sqrt(x**2 - 1))     # Evaluation of the integrand

expression_spec = TemplateExpressionSpec(
    ["f"], "((; f), (x,)) -> D(f, 1)(x)"
)

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    expression_spec=expression_spec,
    maxsize=20,
)
model.fit(x[:, np.newaxis], y)
```

which should correctly find $\frac{\sqrt{x^2 - 1}}{x}$.


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.2.0...v1.3.0

## [1.2.0] (2024-12-14)

### What's Changed
* Compatibility with new scikit-learn API and test suite by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/776
* Add differential operators and input stream specification by @MilesCranmer in https://github.com/MilesCranmer/PySR/pull/780
  * (Note: the differential operators aren't yet in a stable state, and are not yet documented. However, they do work!)
  * This PR also adds various GC allocation improvements in the backend.

**Frontend Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.1.0...v1.2.0

**Backend Changelog**: https://github.com/MilesCranmer/SymbolicRegression.jl/compare/v1.2.0...v1.4.0

## [1.1.0] (2024-12-09)

### What's Changed
* Automated update to backend: v1.2.0 by @github-actions in https://github.com/MilesCranmer/PySR/pull/770


**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.0.2...v1.1.0

## [1.0.2] (2024-12-07)

### What's Changed
* logger fixes: close streams and persist during warm start by @BrotherHa in https://github.com/MilesCranmer/PySR/pull/763
* Let sympy use log2(x) instead of log(x)/log(2) by @nerai in https://github.com/MilesCranmer/PySR/pull/712

### New Contributors
* @BrotherHa made their first contribution in https://github.com/MilesCranmer/PySR/pull/763
* @nerai made their first contribution in https://github.com/MilesCranmer/PySR/pull/712

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.0.1...v1.0.2

## [1.0.1] (2024-12-06)

### What's Changed
* Automated update to backend: v1.1.0 by @github-actions in https://github.com/MilesCranmer/PySR/pull/762
* Fall back to `eager` registry when needed by @DilumAluthge in https://github.com/MilesCranmer/PySR/pull/765

### New Contributors
* @DilumAluthge made their first contribution in https://github.com/MilesCranmer/PySR/pull/765

**Full Changelog**: https://github.com/MilesCranmer/PySR/compare/v1.0.0...v1.0.1

## [1.0.0] (2024-12-01)

### PySR v1.0.0 Release Notes

PySR 1.0.0 introduces new features for imposing specific functional forms and finding parametric expressions. It also includes TensorBoard support, along with significant updates to the core algorithm, including some important bug fixes. The default hyperparameters have also been updated based on extensive tuning, with a maxsize of 30 rather than 20.

### Major New Features

#### Expression Specifications

PySR 1.0.0 introduces new ways to specify the structure of equations through "Expression Specifications", that expose the new backend feature of `AbstractExpression`:

#### Template Expressions
`TemplateExpressionSpec` allows you to define a specific structure for your equations. For example:

```python
expression_spec = TemplateExpressionSpec(["f", "g"], "((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)")
```

#### Parametric Expressions
`ParametricExpressionSpec` enables fitting expressions that can adapt to different categories of data with per-category parameters:

```python
expression_spec = ParametricExpressionSpec(max_parameters=2)
model = PySRRegressor(
    expression_spec=expression_spec
    binary_operators=["+", "*", "-", "/"],
)
model.fit(X, y, category=category)  # Pass category labels
```

#### Improved Logging with TensorBoard

The new `TensorBoardLoggerSpec` enables logging of the search process, as well as hyperparameter recording, which exposes the `AbstractSRLogger` feature of the backend:

```python
logger_spec = TensorBoardLoggerSpec(
    log_dir="logs/run",
    log_interval=10,  # Log every 10 iterations
)
model = PySRRegressor(logger_spec=logger_spec)
```

Features logged include:

- Loss curves over time at each complexity level
- Population statistics
- Pareto "volume" logging (measures performance over all complexities with a single scalar)
- The min loss over time

### Algorithm Improvements

#### Updated Default Parameters

The default hyperparameters have been significantly revised based on testing:

- Increased default `maxsize` from 20 to 30, as I noticed that many people use the defaults, and this maxsize would allow for more accurate expressions.
- New mutation operator weights optimized for better performance, along the new mutation "rotate tree."
- Improved search parameters tuned using Pareto front volume calculations.
- Default `niterations` increased from 40 to 100, also to support better accuracy (at the expense of slightly longer default search times).

#### Core Changes

- New output organization: Results are now stored in `outputs/<run_id>/` rather than in the directory of execution.
- Improved performance with better parallelism handling
- Support for Python 3.10+
- Updated Julia backend to version 1.10+
- Fix for aliasing issues in crossover operations

### Breaking Changes

- Minimum Python version is now 3.10, and minimum Julia version is 1.10
- Output file structure has changed to use directories
- Parameter name updates:
  - `equation_file` → `output_directory` + `run_id`
  - Added clearer naming for parallelism options, such as `parallelism="serial"` rather than the old `multithreading=False, procs=0` which was unclear

### Documentation

The documentation has a new home at https://ai.damtp.cam.ac.uk/pysr/
