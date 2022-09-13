# Customization

If you have explored the [options](options.md) and [PySRRegressor reference](api.md), and still haven't figured out how to specify a constraint or objective required for your problem, you might consider editing the backend.
The backend of PySR is written as a pure Julia package under the name [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl).
This package is accessed with [`PyJulia`](https://github.com/JuliaPy/pyjulia), which allows us to transfer objects back and forth between the Python and Julia runtimes.

PySR gives you access to everything in SymbolicRegression.jl, but there are some specific use-cases which require modifications to the backend itself.
Generally you can do this as follows:

1. Clone a copy of the backend:
```
git clone https://github.com/MilesCranmer/SymbolicRegression.jl
```
2. Edit the source code in `src/` to your requirements:
    -  The documentation for the backend is given [here](https://astroautomata.com/SymbolicRegression.jl/dev/). 
    - Throughout the package, you will often see template functions which typically use a symbol `T` (such as in the string `where {T<:Real}`). Here, `T` is simply the datatype of the input data and stored constants, such as `Float32` or `Float64`. Writing functions in this way lets us write functions generic to types, while still having access to the specific type specified at compilation time.
    - Expressions are stored as binary trees, using the `Node{T}` type, described [here](https://astroautomata.com/SymbolicRegression.jl/dev/types/#SymbolicRegression.CoreModule.EquationModule.Node).
    - Parts of the code which are typically edited by users include:
        - [`src/LossFunctions.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/LossFunctions.jl), particularly the function `eval_loss`. This function assigns a loss to a given expression, using `eval_tree_array` to evaluate it, and `loss` to compute the loss with respect to the dataset.
        - [`src/CheckConstraints.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/CheckConstraints.jl), particularly the function `check_constraints`. This function checks whether a given expression satisfies constraints, such as having a complexity lower than `maxsize`, and whether it contains any forbidden nestings of functions.
            - Note that all expressions, *even intermediate expressions*, must comply with constraints. Therefore, make sure that evolution can still reach your desired expression (with one mutation at a time), before setting a hard constraint. In other cases you might want to instead put in the loss function.
        - [`src/Options.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/Options.jl), as well as the struct definition in [`src/OptionsStruct.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/OptionsStruct.jl). This file specifies all the options used in the search: an instance of `Options` is typically available throughout every function in `SymbolicRegression.jl`. If you add new functionality to the backend, and wish to make it parameterizable (including from PySR), you should specify it in the options.
        - For reference, the main loop itself is found in the `EquationSearch` function inside [`src/SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/SymbolicRegression.jl).
3. Specify the directory of `SymbolicRegression.jl` to PySR by setting `julia_project` in the `PySRRegressor` object, and run `.fit` when you're ready. That's it! No compilation or build steps required.
    - Note that it will automatically update your project by default; to turn this off, set `update=False`.

If you get comfortable enough with the backend, you might consider using the Julia package directly: the API is given on the [SymbolicRegression.jl documentation](https://astroautomata.com/SymbolicRegression.jl/dev/).

If you make a change that you think could be useful to other users, don't hesitate to open a pull request on either the PySR or SymbolicRegression.jl repositories! Contributions are very appreciated.