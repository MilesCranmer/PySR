import numpy as np
import pandas as pd
import tempfile, os, pdb, csv, traceback,random, time


class Problem:
    """
    Problem API to work with PySR.

    Should be able to call pysr(problem.X, problem.y, var_names=problem.var_names) and have it work
    """
    def __init__(self, X, y, var_names=None):
        self.X = X
        self.y = y
        self.var_names = var_names


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
        self.eq_id      = row['Filename']
        self.form       = row['Formula']
        self.n_vars     = int(row['# variables'])
        super(FeynmanProblem, self).__init__(None, None, var_names=[row[f'v{i + 1}_name'] for i in range(self.n_vars)])
        #self.var_names  = [row[f'v{i+1}_name']  for i in range(self.n_vars)]
        self.low        = [float(row[f'v{i+1}_low'])   for i in range(self.n_vars)]
        self.high       = [float(row[f'v{i+1}_high'])  for i in range(self.n_vars)]
        self.dp         = dp#int(row[f'datapoints'])
        #self.X = None
        #self.Y = None
        if gen:
            self.X = np.random.uniform(0.01, 25, size=(self.dp, self.n_vars))
            d = {}
            for var in range(len(self.var_names)):
                d[self.var_names[var]] = self.X[:, var]
            d['exp'] = np.exp
            d['sqrt'] = np.sqrt
            d['pi'] = np.pi
            d['cos'] = np.cos
            d['sin'] = np.sin
            d['tan'] = np.tan
            d['tanh'] = np.tanh
            d['ln']   = np.log
            d['log'] = np.log # Quite sure the Feynman dataset has no base 10 logs
            d['arcsin'] = np.arcsin
            self.Y = eval(self.form,d)
        return

    def __str__(self):
        return f"Feynman Equation: {self.eq_id}|Form: {self.form}"

    def __repr__(self):
        return str(self)

    def mk_problems(first=100, gen=False, dp=500, data_dir="datasets/FeynmanEquations.csv"):
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
                if row['Filename'] == '': continue
                try:
                    p = FeynmanProblem(row, gen=gen, dp=dp)
                    ret.append(p)
                except Exception as e:
                    #traceback.print_exc()
                    #print(row)
                    print(f"FAILED ON ROW {i}")
                ind += 1
        return ret


if __name__ == "__main__":
    ret = FeynmanProblem.mk_problems(first=100, gen=True)
    print(ret)