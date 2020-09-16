import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple
import pathlib
import numpy as np
import pandas as pd


def eureqa(X=None, y=None, threads=4, parsimony=1e-3, alpha=10,
        maxsize=20, migration=True,
        hofMigration=True, fractionReplacedHof=0.1,
        shouldOptimizeConstants=True,
        binary_operators=["plus", "mult"],
        unary_operators=["cos", "exp", "sin"],
        niterations=20, npop=100, annealing=True,
        ncyclesperiteration=5000, fractionReplaced=0.1,
        topn=10, equation_file='hall_of_fame.csv',
        test='simple1'
        ):
    """Either provide a 2D numpy array for X, 1D array for y, or declare a test to run.

     --threads THREADS     Number of threads (default: 4)
     --parsimony PARSIMONY
                           How much to punish complexity (default: 0.001)
     --alpha ALPHA         Scaling of temperature (default: 10)
     --maxsize MAXSIZE     Max size of equation (default: 20)
     --niterations NITERATIONS
                           Number of total migration periods (default: 20)
     --npop NPOP           Number of members per population (default: 100)
     --ncyclesperiteration NCYCLESPERITERATION
                           Number of evolutionary cycles per migration (default:
                           5000)
     --topn TOPN           How many best species to distribute from each
                           population (default: 10)
     --fractionReplacedHof FRACTIONREPLACEDHOF
                           Fraction of population to replace with hall of fame
                           (default: 0.1)
     --fractionReplaced FRACTIONREPLACED
                           Fraction of population to replace with best from other
                           populations (default: 0.1)
     --migration MIGRATION
                           Whether to migrate (default: True)
     --hofMigration HOFMIGRATION
                           Whether to have hall of fame migration (default: True)
     --shouldOptimizeConstants SHOULDOPTIMIZECONSTANTS
                           Whether to use classical optimization on constants
                           before every migration (doesn't impact performance
                           that much) (default: True)
     --annealing ANNEALING
                           Whether to use simulated annealing (default: True)
     --equation_file EQUATION_FILE
                           File to dump best equations to (default:
                           hall_of_fame.csv)
     --test TEST           Which test to run (default: simple1)
     --binary-operators BINARY_OPERATORS [BINARY_OPERATORS ...]
                           Binary operators. Make sure they are defined in
                           operators.jl (default: ['plus', 'mul'])
     --unary-operators UNARY_OPERATORS
                           Unary operators. Make sure they are defined in
                           operators.jl (default: ['exp', 'sin', 'cos'])
    """

    if isinstance(binary_operators, str): binary_operators = [binary_operators]
    if isinstance(unary_operators, str): unary_operators = [unary_operators]

    if X is None:
        if test == 'simple1':
            eval_str = "X[:, 2]**2 + np.cos(X[:, 3]) - 5"
        elif test == 'simple2':
            eval_str = "X[:, 2]**3.5 + 1/abs(X[:, 0])"

        X = np.random.randn(100, 5)*3
        y = eval(eval_str)
        print("Runing on", eval_str)

    def_hyperparams = f"""
    include("operators.jl")
    ##########################
    # # Allowed operators
    # (Apparently using const for globals helps speed)
    const binops = {'[' + ', '.join(binary_operators) + ']'}
    const unaops = {'[' + ', '.join(unary_operators) + ']'}
    ##########################
    
    # How many equations to search when replacing
    const ns=10;
    
    ##################
    # Hyperparameters
    # How much to punish complexity
    const parsimony = {parsimony:f}f0
    # How much to scale temperature by (T between 0 and 1)
    const alpha = {alpha:f}f0
    # Max size of an equation (too large will slow program down)
    const maxsize = {maxsize:d}
    # Whether to migrate between threads (you should)
    const migration = {'true' if migration else 'false'}
    # Whether to re-introduce best examples seen (helps a lot)
    const hofMigration = {'true' if hofMigration else 'false'}
    # Fraction of population to replace with hall of fame
    const fractionReplacedHof = {fractionReplacedHof}f0
    # Optimize constants
    const shouldOptimizeConstants = {'true' if shouldOptimizeConstants else 'false'}
    # File to put operators
    const hofFile = "{equation_file}"
    ##################
    """

    assert len(X.shape) == 2
    assert len(y.shape) == 1

    X_str = str(X.tolist()).replace('],', '];').replace(',', '')
    y_str = str(y.tolist())

    def_datasets = """
    # Here is the function we want to learn (x2^2 + cos(x3) + 5)
    ##########################
    # # Dataset to learn
    const X = convert(Array{Float32, 2}, """f"{X_str})""""
    const y = convert(Array{Float32, 1}, """f"{y_str})""""
    ##########################
    """

    with open('.hyperparams.jl', 'w') as f:
        print(def_hyperparams, file=f)

    with open('.dataset.jl', 'w') as f:
        print(def_datasets, file=f)

    command = [
        f'cd {pathlib.Path(__file__).parent.absolute()}', #Move to filepath of code
        '&&',
        'julia -O3',
        f'--threads {threads}',
        '-e',
        f'\'include("eureqa.jl"); fullRun({niterations:d}, npop={npop:d}, annealing={"true" if annealing else "false"}, ncyclesperiteration={ncyclesperiteration:d}, fractionReplaced={fractionReplaced:f}f0, verbosity=round(Int32, 1e9), topn={topn:d})\'',
        '&&',
        f'cd {pathlib.Path().absolute()}',
        ]
    import os
    cur_cmd = ' '.join(command[:-2])
    print("Running on", cur_cmd)
    os.system(cur_cmd)
    output = pd.read_csv(equation_file, sep="|")
    os.system(command[-1])
    return output





if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--parsimony", type=float, default=0.001, help="How much to punish complexity")
    parser.add_argument("--alpha", type=int, default=10, help="Scaling of temperature")
    parser.add_argument("--maxsize", type=int, default=20, help="Max size of equation")
    parser.add_argument("--niterations", type=int, default=20, help="Number of total migration periods")
    parser.add_argument("--npop", type=int, default=100, help="Number of members per population")
    parser.add_argument("--ncyclesperiteration", type=int, default=5000, help="Number of evolutionary cycles per migration")
    parser.add_argument("--topn", type=int, default=10, help="How many best species to distribute from each population")
    parser.add_argument("--fractionReplacedHof", type=float, default=0.1, help="Fraction of population to replace with hall of fame")
    parser.add_argument("--fractionReplaced", type=float, default=0.1, help="Fraction of population to replace with best from other populations")
    parser.add_argument("--migration", type=bool, default=True, help="Whether to migrate")
    parser.add_argument("--hofMigration", type=bool, default=True, help="Whether to have hall of fame migration")
    parser.add_argument("--shouldOptimizeConstants", type=bool, default=True, help="Whether to use classical optimization on constants before every migration (doesn't impact performance that much)")
    parser.add_argument("--annealing", type=bool, default=True, help="Whether to use simulated annealing")
    parser.add_argument("--equation_file", type=str, default='hall_of_fame.csv', help="File to dump best equations to")
    parser.add_argument("--test", type=str, default='simple1', help="Which test to run")

    parser.add_argument(
            "--binary-operators", type=str, nargs="+", default=["plus", "mul"],
            help="Binary operators. Make sure they are defined in operators.jl")
    parser.add_argument(
            "--unary-operators", type=str, default=["exp", "sin", "cos"],
            help="Unary operators. Make sure they are defined in operators.jl")
    args = vars(parser.parse_args()) #dict

    eureqa(**args)
