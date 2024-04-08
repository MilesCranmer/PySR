"""Print the best model parameters and loss"""

import pickle as pkl
from pprint import PrettyPrinter

import hyperopt
import numpy as np
from hyperopt import Trials, fmin, hp, tpe
from space import space

# Change the following code to your file
################################################################################
# TODO: Declare a folder to hold all trials objects
TRIALS_FOLDER = "trials2"
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
        hyperopt_trial = Trials().new_trial_docs(
            tids=[None], specs=[None], results=[None], miscs=[None]
        )
        hyperopt_trial[0] = trial
        hyperopt_trial[0]["tid"] = tid
        hyperopt_trial[0]["misc"]["tid"] = tid
        for key in hyperopt_trial[0]["misc"]["idxs"].keys():
            hyperopt_trial[0]["misc"]["idxs"][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial)
        trials1.refresh()
    return trials1


np.random.seed()

# Load up all runs:
import glob

path = TRIALS_FOLDER + "/*.pkl"
files = 0
for fname in glob.glob(path):
    trials_obj = pkl.load(open(fname, "rb"))
    n_trials = trials_obj["n"]
    trials_obj = trials_obj["trials"]
    if files == 0:
        trials = trials_obj
    else:
        trials = merge_trials(trials, trials_obj.trials[-n_trials:])
    files += 1


print(files, "trials merged")


best_loss = np.inf
best_trial = None
try:
    trials
except NameError:
    raise NameError("No trials loaded. Be sure to set the right folder")

# for trial in trials:
# if trial['result']['status'] == 'ok':
# loss = trial['result']['loss']
# if loss < best_loss:
# best_loss = loss
# best_trial = trial

# print(best_loss, best_trial['misc']['vals'])
# trials = sorted(trials, key=lambda x: (x['result']['loss'] if trials['result']['status'] == 'ok' else float('inf')))

clean_trials = []
for trial in trials:
    clean_trials.append((trial["result"]["loss"], trial["misc"]["vals"]))

clean_trials = sorted(clean_trials, key=lambda x: x[0])

pp = PrettyPrinter(indent=4)

for trial in clean_trials:
    loss, params = trial
    for k, value in params.items():
        value = value[0]
        if isinstance(value, int):
            possible_args = space[k].pos_args[1:]
            try:
                value = possible_args[value].obj
            except AttributeError:
                value = [arg.obj for arg in possible_args[value].pos_args]

        params[k] = value

    pp.pprint({"loss": loss, "params": params})
