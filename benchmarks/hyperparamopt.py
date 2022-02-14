"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
from pysr import PySRRegressor
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.fmin import generate_trials_to_calculate

# Change the following code to your file
################################################################################
TRIALS_FOLDER = "trials"
NUMBER_TRIALS_PER_RUN = 1
timeout_in_minutes = 5

# Test run to compile everything:
binary_operators = ["*", "/", "+", "-"]
unary_operators = ["sin", "cos", "exp", "log"]
julia_project = None
procs = 4
model = PySRRegressor(
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    timeout_in_seconds=30,
    julia_project=julia_project,
    procs=procs,
)
model.fit(np.random.randn(100, 3), np.random.randn(100))


def run_trial(args):
    """Evaluate the model loss using the hyperparams in args

    :args: A dictionary containing all hyperparameters
    :returns: Dict with status and loss from cross-validation

    """
    # The arguments which are integers:
    integer_args = [
        "populations",
        "niterations",
        "ncyclesperiteration",
        "npop",
        "topn",
        "maxsize",
        "optimizer_nrestarts",
        "optimizer_iterations",
    ]
    # Set these to int types:
    for k, v in args.items():
        if k in integer_args:
            args[k] = int(v)

    # Duplicate this argument:
    args["tournament_selection_n"] = args["topn"]

    # Invalid hyperparams:
    invalid = args["npop"] < args["topn"]
    if invalid:
        return dict(status="fail", loss=float("inf"))

    args["timeout_in_seconds"] = timeout_in_minutes * 60
    args["julia_project"] = julia_project
    args["procs"] = procs

    print(f"Running trial with args: {args}")

    # Create the dataset:
    ntrials = 3
    losses = []

    # Old datasets:
    eval_str = [
        "np.cos(2.3 * X[:, 0]) * np.sin(2.3 * X[:, 0] * X[:, 1] * X[:, 2]) - 10.0",
        "(np.exp(X[:, 3]*0.3) + 3)/(np.exp(X[:, 1]*0.2) + np.cos(X[:, 0]) + 1.1)",
        # "np.sign(X[:, 2])*np.abs(X[:, 2])**2.5 + 5*np.cos(X[:, 3]) - 5",
        # "np.exp(X[:, 0]/2) + 12.0 + np.log(np.abs(X[:, 0])*10 + 1)",
        # "X[:, 0] * np.sin(2*np.pi * (X[:, 1] * X[:, 2] - X[:, 3] / X[:, 4])) + 3.0",
    ]

    for expression in eval_str:
        expression_losses = []
        for i in range(ntrials):
            rstate = np.random.RandomState(i)
            X = 3 * rstate.randn(200, 5)
            y = eval(expression)

            # Normalize y so that losses are fair:
            y = (y - np.average(y)) / np.std(y)

            # Create the model:
            model = PySRRegressor(**args)

            # Run the model:
            try:
                model.fit(X, y)
            except RuntimeError:
                return dict(status="fail", loss=float("inf"))

            # Compute loss:
            cur_loss = float(model.get_best()["loss"])
            expression_losses.append(cur_loss)

        losses.append(np.median(expression_losses))

    loss = np.average(losses)
    print(f"Finished with {loss}", str(args))

    return dict(status="ok", loss=loss)


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
    #     perturbationFactor=1.0,
    perturbationFactor=hp.loguniform("perturbationFactor", np.log(0.0001), np.log(100)),
    #     maxsize=20,
    maxsize=hp.choice("maxsize", [20]),
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

init_vals = [
    dict(
        model_selection=0,  # 0 means first choice
        binary_operators=0,
        unary_operators=0,
        populations=100.0,
        niterations=0,
        ncyclesperiteration=100.0,
        alpha=0.1,
        annealing=0,
        #     fractionReplaced=0.01,
        fractionReplaced=0.01,
        #     fractionReplacedHof=0.005,
        fractionReplacedHof=0.005,
        #     npop=100,
        npop=100.0,
        #     parsimony=1e-4,
        parsimony=1e-4,
        #     topn=10,
        topn=10.0,
        #     weightAddNode=1,
        weightAddNode=1.0,
        #     weightInsertNode=3,
        weightInsertNode=3.0,
        #     weightDeleteNode=3,
        weightDeleteNode=3.0,
        #     weightDoNothing=1,
        weightDoNothing=1.0,
        #     weightMutateConstant=10,
        weightMutateConstant=10.0,
        #     weightMutateOperator=1,
        weightMutateOperator=1.0,
        #     weightRandomize=1,
        weightRandomize=1.0,
        #     weightSimplify=0.002,
        weightSimplify=0,  # One of these is fixed.
        #     perturbationFactor=1.0,
        perturbationFactor=1.0,
        #     maxsize=20,
        maxsize=0,
        #     warmupMaxsizeBy=0.0,
        warmupMaxsizeBy=0.0,
        #     useFrequency=True,
        useFrequency=1,
        #     optimizer_nrestarts=3,
        optimizer_nrestarts=3.0,
        #     optimize_probability=1.0,
        optimize_probability=1.0,
        #     optimizer_iterations=10,
        optimizer_iterations=10.0,
        #     tournament_selection_p=1.0,
        tournament_selection_p=0.999,
    )
]

################################################################################


def merge_trials(trials1, trials2_slice):
    """Merge two hyperopt trials objects

    :trials1: The primary trials object
    :trials2_slice: A slice of the trials object to be merged,
        obtained with, e.g., trials2.trials[:10]
    :returns: The merged trials object

    """
    max_tid = 0
    if len(trials1.trials) > 0:
        max_tid = max([trial["tid"] for trial in trials1.trials])

    for trial in trials2_slice:
        tid = trial["tid"] + max_tid + 1
        local_hyperopt_trial = Trials().new_trial_docs(
            tids=[None], specs=[None], results=[None], miscs=[None]
        )
        local_hyperopt_trial[0] = trial
        local_hyperopt_trial[0]["tid"] = tid
        local_hyperopt_trial[0]["misc"]["tid"] = tid
        for key in local_hyperopt_trial[0]["misc"]["idxs"].keys():
            local_hyperopt_trial[0]["misc"]["idxs"][key] = [tid]
        trials1.insert_trial_docs(local_hyperopt_trial)
        trials1.refresh()
    return trials1


loaded_fnames = []
trials = generate_trials_to_calculate(init_vals)
i = 0
n = NUMBER_TRIALS_PER_RUN

# Run new hyperparameter trials until killed
while True:
    np.random.seed()

    # Load up all runs:
    import glob

    if i > 0:
        path = TRIALS_FOLDER + "/*.pkl"
        for fname in glob.glob(path):
            if fname in loaded_fnames:
                continue

            trials_obj = pkl.load(open(fname, "rb"))
            n_trials = trials_obj["n"]
            trials_obj = trials_obj["trials"]
            if len(loaded_fnames) == 0:
                trials = trials_obj
            else:
                print("Merging trials")
                trials = merge_trials(trials, trials_obj.trials[-n_trials:])

            loaded_fnames.append(fname)

        print("Loaded trials", len(loaded_fnames))
        if len(loaded_fnames) == 0:
            trials = Trials()

        try:
            best = fmin(
                run_trial,
                space=space,
                algo=tpe.suggest,
                max_evals=n + len(trials.trials),
                trials=trials,
                verbose=1,
                rstate=np.random.default_rng(np.random.randint(1, 10**6)),
            )
        except hyperopt.exceptions.AllTrialsFailed:
            continue
    else:
        best = fmin(
            run_trial,
            space=space,
            algo=tpe.suggest,
            max_evals=2,
            trials=trials,
            points_to_evaluate=init_vals,
        )

    print("current best", best)
    hyperopt_trial = Trials()

    # Merge with empty trials dataset:
    save_trials = merge_trials(hyperopt_trial, trials.trials[-n:])
    new_fname = TRIALS_FOLDER + "/" + str(np.random.randint(0, sys.maxsize)) + ".pkl"
    pkl.dump({"trials": save_trials, "n": n}, open(new_fname, "wb"))
    loaded_fnames.append(new_fname)

    i += 1
