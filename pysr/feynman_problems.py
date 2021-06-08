import numpy as np
import csv
import traceback
from .sr import pysr, best
from pathlib import Path
from functools import partial

PKG_DIR = Path(__file__).parents[1]
FEYNMAN_DATASET = PKG_DIR / "datasets" / "FeynmanEquations.csv"


class Problem:
    """
    Problem API to work with PySR.

    Has attributes: X, y as pysr accepts, form which is a string representing the correct equation and variable_names

    Should be able to call pysr(problem.X, problem.y, var_names=problem.var_names) and have it work
    """

    def __init__(self, X, y, form=None, variable_names=None):
        self.X = X
        self.y = y
        self.form = form
        self.variable_names = variable_names


class FeynmanProblem(Problem):
    """
    Stores the data for the problems from the 100 Feynman Equations on Physics.
    This is the benchmark used in the AI Feynman Paper
    """

    def __init__(self, row, gen=False, dp=500):
        """
        row: a row read as a dict from the FeynmanEquations dataset provided in the datasets folder of the repo
        gen: If true the problem will have dp X and y values randomly generated else they will be None
        """
        self.eq_id = row["Filename"]
        self.n_vars = int(row["# variables"])
        super(FeynmanProblem, self).__init__(
            None,
            None,
            form=row["Formula"],
            variable_names=[row[f"v{i + 1}_name"] for i in range(self.n_vars)],
        )
        self.low = [float(row[f"v{i+1}_low"]) for i in range(self.n_vars)]
        self.high = [float(row[f"v{i+1}_high"]) for i in range(self.n_vars)]
        self.dp = dp
        if gen:
            self.X = np.random.uniform(0.01, 25, size=(self.dp, self.n_vars))
            d = {}
            for var in range(len(self.variable_names)):
                d[self.variable_names[var]] = self.X[:, var]
            d["exp"] = np.exp
            d["sqrt"] = np.sqrt
            d["pi"] = np.pi
            d["cos"] = np.cos
            d["sin"] = np.sin
            d["tan"] = np.tan
            d["tanh"] = np.tanh
            d["ln"] = np.log
            d["log"] = np.log  # Quite sure the Feynman dataset has no base 10 logs
            d["arcsin"] = np.arcsin
            self.y = eval(self.form, d)

    def __str__(self):
        return f"Feynman Equation: {self.eq_id}|Form: {self.form}"

    def __repr__(self):
        return str(self)

    def mk_problems(self, first=100, gen=False, dp=500, data_dir=FEYNMAN_DATASET):
        """

        first: the first "first" equations from the dataset will be made into problems
        data_dir: the path pointing to the Feynman Equations csv
        returns: list of FeynmanProblems
        """
        ret = []
        with open(data_dir) as csvfile:
            ind = 0
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if ind > first:
                    break
                if row["Filename"] == "":
                    continue
                try:
                    p = FeynmanProblem(row, gen=gen, dp=dp)
                    ret.append(p)
                except Exception as e:
                    traceback.print_exc()
                    print(f"FAILED ON ROW {i} with {e}")
                ind += 1
        return ret


def run_on_problem(problem, verbosity=0, multiprocessing=True):
    """
    Takes in a problem and returns a tuple: (equations, best predicted equation, actual equation)
    """
    from time import time

    starting = time()
    equations = pysr(
        problem.X,
        problem.y,
        variable_names=problem.variable_names,
        verbosity=verbosity,
    )
    timing = time() - starting
    others = {"time": timing, "problem": problem}
    if not multiprocessing:
        others["equations"] = equations
    return str(best(equations)), problem.form, others


def do_feynman_experiments_parallel(
    first=100,
    verbosity=0,
    dp=500,
    output_file_path="FeynmanExperiment.csv",
    data_dir=FEYNMAN_DATASET,
):
    import multiprocessing as mp
    from tqdm import tqdm

    problems = FeynmanProblem.mk_problems(
        first=first, gen=True, dp=dp, data_dir=data_dir
    )
    ids = []
    predictions = []
    true_equations = []
    time_takens = []
    pool = mp.Pool()
    results = []
    with tqdm(total=len(problems)) as pbar:
        f = partial(run_on_problem, verbosity=verbosity)
        for i, res in enumerate(pool.imap(f, problems)):
            results.append(res)
            pbar.update()
    for res in results:
        prediction, true_equation, others = res
        problem = others["problem"]
        ids.append(problem.eq_id)
        predictions.append(prediction)
        true_equations.append(true_equation)
        time_takens.append(others["time"])
    with open(output_file_path, "a") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["ID", "Predicted", "True", "Time"])
        for i in range(len(ids)):
            writer.writerow([ids[i], predictions[i], true_equations[i], time_takens[i]])


def do_feynman_experiments(
    first=100,
    verbosity=0,
    dp=500,
    output_file_path="FeynmanExperiment.csv",
    data_dir=FEYNMAN_DATASET,
):
    from tqdm import tqdm

    problems = FeynmanProblem.mk_problems(
        first=first, gen=True, dp=dp, data_dir=data_dir
    )
    ids = []
    predictions = []
    true_equations = []
    time_takens = []
    for problem in tqdm(problems):
        prediction, true_equation, others = run_on_problem(problem, verbosity)
        ids.append(problem.eq_id)
        predictions.append(prediction)
        true_equations.append(true_equation)
        time_takens.append(others["time"])
    with open(output_file_path, "a") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["ID", "Predicted", "True", "Time"])
        for i in range(len(ids)):
            writer.writerow([ids[i], predictions[i], true_equations[i], time_takens[i]])
