"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
import pysr
import time

import contextlib


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# Change the following code to your file
################################################################################
TRIALS_FOLDER = "trials"
NUMBER_TRIALS_PER_RUN = 1


def run_trial(args):
    """Evaluate the model loss using the hyperparams in args

    :args: A dictionary containing all hyperparameters
    :returns: Dict with status and loss from cross-validation

    """

    print("Running on", args)
    args["niterations"] = 100
    args["npop"] = 100
    args["ncyclesperiteration"] = 1000
    args["topn"] = 10
    args["parsimony"] = 0.0
    args["useFrequency"] = True
    args["annealing"] = True

    if args["npop"] < 20 or args["ncyclesperiteration"] < 3:
        print("Bad parameters")
        return {"status": "ok", "loss": np.inf}

    args["weightDoNothing"] = 1.0
    ntrials = 3

    with temp_seed(0):
        X = np.random.randn(100, 10) * 3

    eval_str = [
        "np.sign(X[:, 2])*np.abs(X[:, 2])**2.5 + 5*np.cos(X[:, 3]) - 5",
        "np.exp(X[:, 0]/2) + 12.0 + np.log(np.abs(X[:, 0])*10 + 1)",
        "(np.exp(X[:, 3]) + 3)/(np.abs(X[:, 1]) + np.cos(X[:, 0]) + 1.1)",
        "X[:, 0] * np.sin(2*np.pi * (X[:, 1] * X[:, 2] - X[:, 3] / X[:, 4])) + 3.0",
    ]

    print(f"Starting", str(args))
    try:
        local_trials = []
        for i in range(len(eval_str)):
            print(f"Starting test {i}")
            for j in range(ntrials):
                print(f"Starting trial {j}")
                y = eval(eval_str[i])
                trial = pysr.pysr(
                    X,
                    y,
                    procs=4,
                    populations=20,
                    binary_operators=["plus", "mult", "pow", "div"],
                    unary_operators=["cos", "exp", "sin", "logm", "abs"],
                    maxsize=25,
                    constraints={"pow": (-1, 1)},
                    **args,
                )
                if len(trial) == 0:
                    raise ValueError
                local_trials.append(
                    np.min(trial["MSE"]) ** 0.5 / np.std(eval(eval_str[i - 1]))
                )
                print(f"Test {i} trial {j} with", str(args), f"got {local_trials[-1]}")

    except ValueError:
        print(f"Broken", str(args))
        return {"status": "ok", "loss": np.inf}  # or 'fail' if nan loss
    loss = np.average(local_trials)
    print(f"Finished with {loss}", str(args))

    return {"status": "ok", "loss": loss}  # or 'fail' if nan loss


space = {
    "alpha": hp.lognormal("alpha", np.log(10.0), 1.0),
    "fractionReplacedHof": hp.lognormal("fractionReplacedHof", np.log(0.1), 1.0),
    "fractionReplaced": hp.lognormal("fractionReplaced", np.log(0.1), 1.0),
    "perturbationFactor": hp.lognormal("perturbationFactor", np.log(1.0), 1.0),
    "weightMutateConstant": hp.lognormal("weightMutateConstant", np.log(4.0), 1.0),
    "weightMutateOperator": hp.lognormal("weightMutateOperator", np.log(0.5), 1.0),
    "weightAddNode": hp.lognormal("weightAddNode", np.log(0.5), 1.0),
    "weightInsertNode": hp.lognormal("weightInsertNode", np.log(0.5), 1.0),
    "weightDeleteNode": hp.lognormal("weightDeleteNode", np.log(0.5), 1.0),
    "weightSimplify": hp.lognormal("weightSimplify", np.log(0.05), 1.0),
    "weightRandomize": hp.lognormal("weightRandomize", np.log(0.25), 1.0),
}

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
trials = None
# Run new hyperparameter trials until killed
while True:
    np.random.seed()

    # Load up all runs:
    import glob

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

    n = NUMBER_TRIALS_PER_RUN
    try:
        best = fmin(
            run_trial,
            space=space,
            algo=tpe.suggest,
            max_evals=n + len(trials.trials),
            trials=trials,
            verbose=1,
            rstate=np.random.RandomState(np.random.randint(1, 10 ** 6)),
        )
    except hyperopt.exceptions.AllTrialsFailed:
        continue

    print("current best", best)
    hyperopt_trial = Trials()

    # Merge with empty trials dataset:
    save_trials = merge_trials(hyperopt_trial, trials.trials[-n:])
    new_fname = (
        TRIALS_FOLDER
        + "/"
        + str(np.random.randint(0, sys.maxsize))
        + str(time.time())
        + ".pkl"
    )
    pkl.dump({"trials": save_trials, "n": n}, open(new_fname, "wb"))
    loaded_fnames.append(new_fname)
