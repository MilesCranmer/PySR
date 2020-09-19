"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
import pysr
import time

import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


#Change the following code to your file
################################################################################
TRIALS_FOLDER = 'trials'
NUMBER_TRIALS_PER_RUN = 1

def run_trial(args):
    """Evaluate the model loss using the hyperparams in args

    :args: A dictionary containing all hyperparameters
    :returns: Dict with status and loss from cross-validation

    """

    print("Running on", args)
    for key in 'niterations npop'.split(' '):
        args[key] = int(args[key])


    total_steps = 10*100*1000
    niterations = args['niterations']
    npop = args['npop']
    if niterations == 0 or npop == 0: 
        print("Bad parameters")
        return {'status': 'ok', 'loss': np.inf}
        
    args['ncyclesperiteration'] = int(total_steps / (niterations * npop))
    args['topn'] = 10
    args['parsimony'] = 1e-3
    args['annealing'] = True

    if args['npop'] < 20 or args['ncyclesperiteration'] < 3:
        print("Bad parameters")
        return {'status': 'ok', 'loss': np.inf}


    args['weightDoNothing'] = 1.0

    maxTime = 30
    ntrials = 2
    equation_file = f'.hall_of_fame_{np.random.rand():f}.csv'

    with temp_seed(0):
        X = np.random.randn(100, 5)*3

    eval_str = ["np.sign(X[:, 2])*np.abs(X[:, 2])**2.5 + 5*np.cos(X[:, 3]) - 5",
    "np.sign(X[:, 2])*np.abs(X[:, 2])**3.5 + 1/(np.abs(X[:, 0])+1)",
    "np.exp(X[:, 0]/2) + 12.0 + np.log(np.abs(X[:, 0])*10 + 1)",
    "1.0 + 3*X[:, 0]**2 - 0.5*X[:, 0]**3 + 0.1*X[:, 0]**4",
    "(np.exp(X[:, 3]) + 3)/(np.abs(X[:, 1]) + np.cos(X[:, 0]) + 1.1)"]

    print(f"Starting", str(args))
    try:
        trials = []
        for i in range(3, 6):
            print(f"Starting test {i}")
            for j in range(ntrials):
                print(f"Starting trial {j}")
                trial = pysr.pysr(
                    test=f"simple{i}",
                    threads=4,
                    binary_operators=["plus", "mult", "pow", "div"],
                    unary_operators=["cos", "exp", "sin", "loga", "abs"],
                    equation_file=equation_file,
                    timeout=maxTime,
                    maxsize=25,
                    verbosity=0,
                    **args)
                if len(trial) == 0: raise ValueError
                trials.append(
                        np.min(trial['MSE'])**0.5 / np.std(eval(eval_str[i-1]))
                )
                print(f"Test {i} trial {j} with", str(args), f"got {trials[-1]}")

    except ValueError:
        print(f"Broken", str(args))
        return {
            'status': 'ok', # or 'fail' if nan loss
            'loss': np.inf
        }
    loss = np.average(trials)
    print(f"Finished with {loss}", str(args))

    return {
        'status': 'ok', # or 'fail' if nan loss
        'loss': loss
    }


space = {
    'niterations': hp.qlognormal('niterations', np.log(10), 1.0, 1),
    'npop': hp.qlognormal('npop', np.log(100), 1.0, 1),
    'alpha': hp.lognormal('alpha', np.log(10.0), 1.0),
    'fractionReplacedHof': hp.lognormal('fractionReplacedHof', np.log(0.1), 1.0),
    'fractionReplaced': hp.lognormal('fractionReplaced', np.log(0.1), 1.0),
    'perturbationFactor': hp.lognormal('perturbationFactor', np.log(1.0), 1.0),
    'weightMutateConstant': hp.lognormal('weightMutateConstant', np.log(4.0), 1.0),
    'weightMutateOperator': hp.lognormal('weightMutateOperator', np.log(0.5), 1.0),
    'weightAddNode': hp.lognormal('weightAddNode', np.log(0.5), 1.0),
    'weightInsertNode': hp.lognormal('weightInsertNode', np.log(0.5), 1.0),
    'weightDeleteNode': hp.lognormal('weightDeleteNode', np.log(0.5), 1.0),
    'weightSimplify': hp.lognormal('weightSimplify', np.log(0.05), 1.0),
    'weightRandomize': hp.lognormal('weightRandomize', np.log(0.25), 1.0),
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
        max_tid = max([trial['tid'] for trial in trials1.trials])

    for trial in trials2_slice:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial) 
        trials1.refresh()
    return trials1

loaded_fnames = []
trials = None
# Run new hyperparameter trials until killed
while True:
    np.random.seed()

    # Load up all runs:
    import glob
    path = TRIALS_FOLDER + '/*.pkl'
    for fname in glob.glob(path):
        if fname in loaded_fnames:
            continue

        trials_obj = pkl.load(open(fname, 'rb'))
        n_trials = trials_obj['n']
        trials_obj = trials_obj['trials']
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
        best = fmin(run_trial,
            space=space,
            algo=tpe.suggest,
            max_evals=n + len(trials.trials),
            trials=trials,
            verbose=1,
            rstate=np.random.RandomState(np.random.randint(1,10**6))
            )
    except hyperopt.exceptions.AllTrialsFailed:
        continue

    print('current best', best)
    hyperopt_trial = Trials()

    # Merge with empty trials dataset:
    save_trials = merge_trials(hyperopt_trial, trials.trials[-n:])
    new_fname = TRIALS_FOLDER + '/' + str(np.random.randint(0, sys.maxsize)) + str(time.time()) + '.pkl'
    pkl.dump({'trials': save_trials, 'n': n}, open(new_fname, 'wb'))
    loaded_fnames.append(new_fname)

