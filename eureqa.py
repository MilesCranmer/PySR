import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple
import pathlib
import numpy as np
import pandas as pd

# Dumped from hyperparam optimization
default_alpha =                      2.288229
default_annealing =                  1.000000
default_fractionReplaced =           0.121271
default_fractionReplacedHof =        0.065129
default_ncyclesperiteration =    15831.000000
default_niterations =               11.000000
default_npop =                     105.000000
default_parsimony =                  0.000465
default_topn =                       6.000000
default_weightAddNode =              0.454050
default_weightDeleteNode =           0.603670
default_weightDoNothing =            0.141223
default_weightMutateConstant =       3.680211
default_weightMutateOperator =       0.660488
default_weightRandomize =            6.759691
default_weightSimplify =             0.010442
default_result =                     0.687007

def eureqa(X=None, y=None, threads=4,
            niterations=20,
            ncyclesperiteration=int(default_ncyclesperiteration),
            binary_operators=["plus", "mult"],
            unary_operators=["cos", "exp", "sin"],
            alpha=default_alpha,
            annealing=True,
            fractionReplaced=default_fractionReplaced,
            fractionReplacedHof=default_fractionReplacedHof,
            npop=int(default_npop),
            parsimony=default_parsimony,
            migration=True,
            hofMigration=True,
            shouldOptimizeConstants=True,
            topn=int(default_topn),
            weightAddNode=default_weightAddNode,
            weightDeleteNode=default_weightDeleteNode,
            weightDoNothing=default_weightDoNothing,
            weightMutateConstant=default_weightMutateConstant,
            weightMutateOperator=default_weightMutateOperator,
            weightRandomize=default_weightRandomize,
            weightSimplify=default_weightSimplify,
            timeout=None,
            equation_file='hall_of_fame.csv',
            test='simple1',
            maxsize=20,
        ):
    """Run symbolic regression to fit f(X[i, :]) ~ y[i] for all i.
    Note: most default parameters have been tuned over several example
    equations, but you should adjust `threads`, `niterations`,
    `binary_operators`, `unary_operators` to your requirements.

    :param X: 2D array. Rows are examples, columns are features.
    :type X: np.ndarray, optional
    :param y: 1D array. Rows are examples.
    :type y: np.ndarray, optional
    :param threads: Number of threads (=number of populations running).
        You can have more threads than cores - it actually makes it more
        efficient.
    :type threads: int, optional
    :param niterations: Number of iterations of the algorithm to run. The best
        equations are printed, and migrate between populations, at the
        end of each.
    :type niterations: int, optional
    :param ncyclesperiteration: Number of total mutations to run, per 10
        samples of the population, per iteration.
    :type ncyclesperiteration: int, optional
    :param binary_operators: List of strings giving the binary operators
        in Julia's Base, or in `operator.jl`.
    :type binary_operators: list, optional
    :param unary_operators: Same but for operators taking a single `Float32`.
    :type unary_operators: list, optional
    :param alpha: Initial temperature.
    :type alpha: float, optional
    :param annealing: Whether to use annealing. You should (and it is default).
    :type annealing: bool, optional
    :param fractionReplaced: How much of population to replace with migrating
        equations from other populations.
    :type fractionReplaced: float, optional
    :param fractionReplacedHof: How much of population to replace with migrating
        equations from hall of fame.
    :type fractionReplacedHof: float, optional
    :param npop: Number of individuals in each population
    :type npop: int, optional
    :param parsimony: Multiplicative factor for how much to punish complexity.
    :type parsimony: float, optional
    :param migration: Whether to migrate.
    :type migration: bool, optional
    :param hofMigration: Whether to have the hall of fame migrate.
    :type hofMigration: bool, optional
    :param shouldOptimizeConstants: Whether to numerically optimize
        constants (Nelder-Mead/Newton) at the end of each iteration.
    :type shouldOptimizeConstants: bool, optional
    :param topn: How many top individuals migrate from each population.
    :type topn: int, optional
    :param weightAddNode: Relative likelihood for mutation to add a node
    :type weightAddNode: float, optional
    :param weightDeleteNode: Relative likelihood for mutation to delete a node
    :type weightDeleteNode: float, optional
    :param weightDoNothing: Relative likelihood for mutation to leave the individual
    :type weightDoNothing: float, optional
    :param weightMutateConstant: Relative likelihood for mutation to change
        the constant slightly in a random direction.
    :type weightMutateConstant: float, optional
    :param weightMutateOperator: Relative likelihood for mutation to swap
        an operator.
    :type weightMutateOperator: float, optional
    :param weightRandomize: Relative likelihood for mutation to completely
        delete and then randomly generate the equation
    :type weightRandomize: float, optional
    :param weightSimplify: Relative likelihood for mutation to simplify
        constant parts by evaluation
    :type weightSimplify: float, optional
    :param timeout: Time in seconds to timeout search
    :type timeout: float, optional
    :param equation_file: Where to save the files (.csv separated by |)
    :type equation_file: str, optional
    :param test: What test to run, if X,y not passed.
    :type test: str, optional
    :param maxsize: Max size of an equation.
    :type maxsize: int, optional
    :returns: Results dataframe, giving complexity, MSE, and equations
        (as strings).
    :rtype: pd.DataFrame

    """

    rand_string = f'{"".join([str(np.random.rand())[2] for i in range(20)])}'

    if isinstance(binary_operators, str): binary_operators = [binary_operators]
    if isinstance(unary_operators, str): unary_operators = [unary_operators]

    if X is None:
        if test == 'simple1':
            eval_str = "np.sign(X[:, 2])*np.abs(X[:, 2])**2.5 + 5*np.cos(X[:, 3]) - 5"
        elif test == 'simple2':
            eval_str = "np.sign(X[:, 2])*np.abs(X[:, 2])**3.5 + 1/np.abs(X[:, 0])"
        elif test == 'simple3':
            eval_str = "np.exp(X[:, 0]/2) + 12.0 + np.log(np.abs(X[:, 0])*10 + 1)"
        elif test == 'simple4':
            eval_str = "1.0 + 3*X[:, 0]**2 - 0.5*X[:, 0]**3 + 0.1*X[:, 0]**4"
        elif test == 'simple5':
            eval_str = "(np.exp(X[:, 3]) + 3)/(X[:, 1] + np.cos(X[:, 0]))"

        X = np.random.randn(100, 5)*3
        y = eval(eval_str)
        print("Running on", eval_str)

    def_hyperparams = f"""include("operators.jl")
const binops = {'[' + ', '.join(binary_operators) + ']'}
const unaops = {'[' + ', '.join(unary_operators) + ']'}
const ns=10;
const parsimony = {parsimony:f}f0
const alpha = {alpha:f}f0
const maxsize = {maxsize:d}
const migration = {'true' if migration else 'false'}
const hofMigration = {'true' if hofMigration else 'false'}
const fractionReplacedHof = {fractionReplacedHof}f0
const shouldOptimizeConstants = {'true' if shouldOptimizeConstants else 'false'}
const hofFile = "{equation_file}"
const nthreads = {threads:d}
const mutationWeights = [
    {weightMutateConstant:f},
    {weightMutateOperator:f},
    {weightAddNode:f},
    {weightDeleteNode:f},
    {weightSimplify:f},
    {weightRandomize:f},
    {weightDoNothing:f}
]
    """

    assert len(X.shape) == 2
    assert len(y.shape) == 1

    X_str = str(X.tolist()).replace('],', '];').replace(',', '')
    y_str = str(y.tolist())

    def_datasets = """const X = convert(Array{Float32, 2}, """f"{X_str})""""
const y = convert(Array{Float32, 1}, """f"{y_str})""""
    """

    starting_path = f'cd {pathlib.Path().absolute()}'
    code_path = f'cd {pathlib.Path(__file__).parent.absolute()}' #Move to filepath of code

    os.system(code_path)

    with open(f'.hyperparams_{rand_string}.jl', 'w') as f:
        print(def_hyperparams, file=f)

    with open(f'.dataset_{rand_string}.jl', 'w') as f:
        print(def_datasets, file=f)

    command = [
        'julia -O3',
        f'--threads {threads}',
        '-e',
        f'\'include(".hyperparams_{rand_string}.jl"); include(".dataset_{rand_string}.jl"); include("eureqa.jl"); fullRun({niterations:d}, npop={npop:d}, annealing={"true" if annealing else "false"}, ncyclesperiteration={ncyclesperiteration:d}, fractionReplaced={fractionReplaced:f}f0, verbosity=round(Int32, 1e9), topn={topn:d})\'',
        ]
    if timeout is not None:
        command = [f'timeout {timeout}'] + command
    cur_cmd = ' '.join(command)
    print("Running on", cur_cmd)
    os.system(cur_cmd)
    try:
        output = pd.read_csv(equation_file, sep="|")
    except FileNotFoundError:
        print("Couldn't find equation file!")
        output = pd.DataFrame()
    os.system(starting_path)
    return output



if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--parsimony", type=float, default=default_parsimony, help="How much to punish complexity")
    parser.add_argument("--alpha", type=float, default=default_alpha, help="Scaling of temperature")
    parser.add_argument("--maxsize", type=int, default=20, help="Max size of equation")
    parser.add_argument("--niterations", type=int, default=20, help="Number of total migration periods")
    parser.add_argument("--npop", type=int, default=int(default_npop), help="Number of members per population")
    parser.add_argument("--ncyclesperiteration", type=int, default=int(default_ncyclesperiteration), help="Number of evolutionary cycles per migration")
    parser.add_argument("--topn", type=int, default=int(default_topn), help="How many best species to distribute from each population")
    parser.add_argument("--fractionReplacedHof", type=float, default=default_fractionReplacedHof, help="Fraction of population to replace with hall of fame")
    parser.add_argument("--fractionReplaced", type=float, default=default_fractionReplaced, help="Fraction of population to replace with best from other populations")
    parser.add_argument("--migration", type=bool, default=True, help="Whether to migrate")
    parser.add_argument("--hofMigration", type=bool, default=True, help="Whether to have hall of fame migration")
    parser.add_argument("--shouldOptimizeConstants", type=bool, default=True, help="Whether to use classical optimization on constants before every migration (doesn't impact performance that much)")
    parser.add_argument("--annealing", type=bool, default=True, help="Whether to use simulated annealing")
    parser.add_argument("--equation_file", type=str, default='hall_of_fame.csv', help="File to dump best equations to")
    parser.add_argument("--test", type=str, default='simple1', help="Which test to run")

    parser.add_argument(
            "--binary-operators", type=str, nargs="+", default=["plus", "mult"],
            help="Binary operators. Make sure they are defined in operators.jl")
    parser.add_argument(
            "--unary-operators", type=str, nargs="+", default=["exp", "sin", "cos"],
            help="Unary operators. Make sure they are defined in operators.jl")
    args = vars(parser.parse_args()) #dict

    eureqa(**args)
