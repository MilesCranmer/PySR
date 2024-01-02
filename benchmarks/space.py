import numpy as np
from hyperopt import Trials, fmin, hp, tpe

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
    #     fraction_replaced=0.01,
    fraction_replaced=hp.loguniform("fraction_replaced", np.log(0.0001), np.log(0.5)),
    #     fraction_replaced_hof=0.005,
    fraction_replaced_hof=hp.loguniform(
        "fraction_replaced_hof", np.log(0.0001), np.log(0.5)
    ),
    #     population_size=100,
    population_size=hp.qloguniform("population_size", np.log(20), np.log(1000), 1),
    #     parsimony=1e-4,
    parsimony=hp.loguniform("parsimony", np.log(0.0001), np.log(0.5)),
    #     topn=10,
    topn=hp.qloguniform("topn", np.log(2), np.log(50), 1),
    #     weight_add_node=1,
    weight_add_node=hp.loguniform("weight_add_node", np.log(0.0001), np.log(100)),
    #     weight_insert_node=3,
    weight_insert_node=hp.loguniform("weight_insert_node", np.log(0.0001), np.log(100)),
    #     weight_delete_node=3,
    weight_delete_node=hp.loguniform("weight_delete_node", np.log(0.0001), np.log(100)),
    #     weight_do_nothing=1,
    weight_do_nothing=hp.loguniform("weight_do_nothing", np.log(0.0001), np.log(100)),
    #     weight_mutate_constant=10,
    weight_mutate_constant=hp.loguniform(
        "weight_mutate_constant", np.log(0.0001), np.log(100)
    ),
    #     weight_mutate_operator=1,
    weight_mutate_operator=hp.loguniform(
        "weight_mutate_operator", np.log(0.0001), np.log(100)
    ),
    #     weight_swap_operands=1,
    weight_swap_operands=hp.loguniform(
        "weight_swap_operands", np.log(0.0001), np.log(100)
    ),
    #     weight_randomize=1,
    weight_randomize=hp.loguniform("weight_randomize", np.log(0.0001), np.log(100)),
    #     weight_simplify=0.002,
    weight_simplify=hp.choice("weight_simplify", [0.002]),  # One of these is fixed.
    #     crossover_probability=0.01,
    crossover_probability=hp.loguniform(
        "crossover_probability", np.log(0.00001), np.log(0.2)
    ),
    #     perturbation_factor=1.0,
    perturbation_factor=hp.loguniform(
        "perturbation_factor", np.log(0.0001), np.log(100)
    ),
    #     maxsize=20,
    maxsize=hp.choice("maxsize", [30]),
    #     warmup_maxsize_by=0.0,
    warmup_maxsize_by=hp.uniform("warmup_maxsize_by", 0.0, 0.5),
    #     use_frequency=True,
    use_frequency=hp.choice("use_frequency", [True, False]),
    #     optimizer_nrestarts=3,
    optimizer_nrestarts=hp.quniform("optimizer_nrestarts", 1, 10, 1),
    #     optimize_probability=1.0,
    optimize_probability=hp.uniform("optimize_probability", 0.0, 1.0),
    #     optimizer_iterations=10,
    optimizer_iterations=hp.quniform("optimizer_iterations", 1, 10, 1),
    #     tournament_selection_p=1.0,
    tournament_selection_p=hp.uniform("tournament_selection_p", 0.0, 1.0),
)
