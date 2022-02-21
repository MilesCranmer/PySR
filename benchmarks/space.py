import numpy as np
from hyperopt import hp, fmin, tpe, Trials

binary_operators = ["*", "/", "+", "-"]
unary_operators = ["sin", "cos", "exp", "log"]

space = dict(
    #     model_selection="best",
    model_selection=hp.choice("model_selection", ["accuracy"]),
    #     binary_operators=None,
    binary_operators=hp.choice("binary_operators", [binary_operators]),
    #     unary_operators=None,
    unary_operators=hp.choice("unary_operators", [unary_operators]),
    #     populations=100,
    populations=hp.qloguniform("populations", np.log(10), np.log(1000), 1),
    #     niterations=4,
    niterations=hp.choice(
        "niterations", [10000]
    ),  # We will quit automatically based on a clock.
    #     ncyclesperiteration=100,
    ncyclesperiteration=hp.qloguniform(
        "ncyclesperiteration", np.log(10), np.log(5000), 1
    ),
    #     alpha=0.1,
    alpha=hp.loguniform("alpha", np.log(0.0001), np.log(1000)),
    #     annealing=False,
    annealing=hp.choice("annealing", [False, True]),
    #     fractionReplaced=0.01,
    fractionReplaced=hp.loguniform("fractionReplaced", np.log(0.0001), np.log(0.5)),
    #     fractionReplacedHof=0.005,
    fractionReplacedHof=hp.loguniform(
        "fractionReplacedHof", np.log(0.0001), np.log(0.5)
    ),
    #     npop=100,
    npop=hp.qloguniform("npop", np.log(20), np.log(1000), 1),
    #     parsimony=1e-4,
    parsimony=hp.loguniform("parsimony", np.log(0.0001), np.log(0.5)),
    #     topn=10,
    topn=hp.qloguniform("topn", np.log(2), np.log(50), 1),
    #     weightAddNode=1,
    weightAddNode=hp.loguniform("weightAddNode", np.log(0.0001), np.log(100)),
    #     weightInsertNode=3,
    weightInsertNode=hp.loguniform("weightInsertNode", np.log(0.0001), np.log(100)),
    #     weightDeleteNode=3,
    weightDeleteNode=hp.loguniform("weightDeleteNode", np.log(0.0001), np.log(100)),
    #     weightDoNothing=1,
    weightDoNothing=hp.loguniform("weightDoNothing", np.log(0.0001), np.log(100)),
    #     weightMutateConstant=10,
    weightMutateConstant=hp.loguniform(
        "weightMutateConstant", np.log(0.0001), np.log(100)
    ),
    #     weightMutateOperator=1,
    weightMutateOperator=hp.loguniform(
        "weightMutateOperator", np.log(0.0001), np.log(100)
    ),
    #     weightRandomize=1,
    weightRandomize=hp.loguniform("weightRandomize", np.log(0.0001), np.log(100)),
    #     weightSimplify=0.002,
    weightSimplify=hp.choice("weightSimplify", [0.002]),  # One of these is fixed.
    #     crossoverProbability=0.01,
    crossoverProbability=hp.loguniform(
        "crossoverProbability", np.log(0.00001), np.log(0.2)
    ),
    #     perturbationFactor=1.0,
    perturbationFactor=hp.loguniform("perturbationFactor", np.log(0.0001), np.log(100)),
    #     maxsize=20,
    maxsize=hp.choice("maxsize", [30]),
    #     warmupMaxsizeBy=0.0,
    warmupMaxsizeBy=hp.uniform("warmupMaxsizeBy", 0.0, 0.5),
    #     useFrequency=True,
    useFrequency=hp.choice("useFrequency", [True, False]),
    #     optimizer_nrestarts=3,
    optimizer_nrestarts=hp.quniform("optimizer_nrestarts", 1, 10, 1),
    #     optimize_probability=1.0,
    optimize_probability=hp.uniform("optimize_probability", 0.0, 1.0),
    #     optimizer_iterations=10,
    optimizer_iterations=hp.quniform("optimizer_iterations", 1, 10, 1),
    #     tournament_selection_p=1.0,
    tournament_selection_p=hp.uniform("tournament_selection_p", 0.0, 1.0),
)
