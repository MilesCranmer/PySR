"""Start a hyperoptimization from a single node"""
import sys
import numpy as np
import pickle as pkl
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
import eureqa


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
    for key in 'niterations npop ncyclesperiteration topn'.split(' '):
        args[key] = int(args[key])

    if args['npop'] < 50 or args['ncyclesperiteration'] < 3:
        print("Bad parameters")
        return {'status': 'ok', 'loss': np.inf}

    def handler(signum, frame):
        print("Took too long. Skipping.")
        raise ValueError("Takes too long")

    maxTime = 120
    ntrials = 3
    equation_file = f'.hall_of_fame_{np.random.rand():f}.csv'

    try:
        trials = []
        for i in range(1, 4):
            subtrials = []
            for j in range(ntrials):
                trial = eureqa.eureqa(
                    test=f"simple{i}",
                    threads=4,
                    binary_operators=["plus", "mult", "pow", "div"],
                    unary_operators=["cos", "exp", "sin", "log"],
                    equation_file=equation_file,
                    timeout=maxTime,
                    **args)
                if len(trial) == 0: raise ValueError
                subtrials.append(np.min(trial['MSE']))
            trials.append(np.log(np.median(subtrials) + 0.1))
    except ValueError:
        return {
            'status': 'ok', # or 'fail' if nan loss
            'loss': np.inf
        }

    loss = np.average(trials)
    print(args, "got", loss)

    return {
        'status': 'ok', # or 'fail' if nan loss
        'loss': loss
    }


space = {
    'niterations': hp.qlognormal('niterations', np.log(10), 0.5, 1),
    'npop': hp.qlognormal('npop', np.log(100), 0.5, 1),
    'ncyclesperiteration': hp.qlognormal('ncyclesperiteration', np.log(5000), 0.5, 1),
    'topn': hp.quniform('topn', 1, 30, 1),
    'annealing': hp.choice('annealing', [False, True]),
    'alpha': hp.lognormal('alpha', np.log(10.0), 0.5),
    'parsimony': hp.lognormal('parsimony', np.log(1e-3), 0.5),
    'fractionReplacedHof': hp.lognormal('fractionReplacedHof', np.log(0.1), 0.5),
    'fractionReplaced': hp.lognormal('fractionReplaced', np.log(0.1), 0.5),
    'weightMutateConstant': hp.lognormal('weightMutateConstant', np.log(4.0), 0.5),
    'weightMutateOperator': hp.lognormal('weightMutateOperator', np.log(0.5), 0.5),
    'weightAddNode': hp.lognormal('weightAddNode', np.log(0.5), 0.5),
    'weightDeleteNode': hp.lognormal('weightDeleteNode', np.log(0.5), 0.5),
    'weightSimplify': hp.lognormal('weightSimplify', np.log(0.05), 0.5),
    'weightRandomize': hp.lognormal('weightRandomize', np.log(0.25), 0.5),
    'weightDoNothing': hp.lognormal('weightDoNothing', np.log(1.0), 0.5),
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
    new_fname = TRIALS_FOLDER + '/' + str(np.random.randint(0, sys.maxsize)) + '.pkl'
    pkl.dump({'trials': save_trials, 'n': n}, open(new_fname, 'wb'))
    loaded_fnames.append(new_fname)

