import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import namedtuple


def eureqa(threads=4, parsimony=1e-3, alpha=10,
        maxsize=20, migration=True,
        hofMigration=True, fractionReplacedHof=0.1,
        shouldOptimizeConstants=True,
        binary_operators=["plus", "mult"],
        unary_operators=["cos", "exp", "sin"],
        niterations=20, npop=100, annealing=True,
        ncyclesperiteration=5000, fractionReplaced=0.1,
        topn=10, equation_file='hall_of_fame.csv'
        ):

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

    def_datasets = """
    # Here is the function we want to learn (x2^2 + cos(x3) + 5)
    ##########################
    # # Dataset to learn
    const X = convert(Array{Float32, 2}, randn(100, 5)*2)
    const y = convert(Array{Float32, 1}, ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]) .- 5)
    ##########################
    """

    with open('.hyperparams.jl', 'w') as f:
        print(def_hyperparams, file=f)

    with open('.dataset.jl', 'w') as f:
        print(def_datasets, file=f)

    command = ' '.join([
        'julia -O3',
        f'--threads {threads}',
        '-e',
        f'\'include("eureqa.jl"); fullRun({niterations:d}, npop={npop:d}, annealing={"true" if annealing else "false"}, ncyclesperiteration={ncyclesperiteration:d}, fractionReplaced={fractionReplaced:f}f0, verbosity=round(Int32, 1e9), topn={topn:d})\''
        ])
    import os
    os.system(command)


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

    parser.add_argument(
            "--binary-operators", type=str, nargs="+", default=["plus", "mul"],
            help="Binary operators. Make sure they are defined in operators.jl")
    parser.add_argument(
            "--unary-operators", type=str, default=["exp", "sin", "cos"],
            help="Unary operators. Make sure they are defined in operators.jl")
    args = vars(parser.parse_args()) #dict

    eureqa(**args)
