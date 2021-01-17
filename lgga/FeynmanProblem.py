import numpy as np
import pandas as pd
import tempfile, os, pdb, csv, traceback,random, time


class FeynmanProblem:
    def __init__(self, row, gen=False):
        self.eq_id      = row['Filename']
        self.form       = row['Formula']
        self.n_vars     = int(row['# variables'])
        self.var_names  = [row[f'v{i+1}_name']  for i in range(self.n_vars)]
        self.low        = [float(row[f'v{i+1}_low'])   for i in range(self.n_vars)]
        self.high       = [float(row[f'v{i+1}_high'])  for i in range(self.n_vars)]
        self.dp         = 500#int(row[f'datapoints'])
        self.X = None
        self.Y = None
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
            d['arcsin'] = np.arcsin
            self.Y = eval(self.form,d)
        return

    def __str__(self):
        return f"Feynman Equation: {self.eq_id}|Form: {self.form}"

    def __repr__(self):
        return str(self)

    def mk_problems(first=100, gen=False, data_dir="datasets/FeynmanEquations.csv"):
        ret = []
        with open(data_dir) as csvfile:
            ind = 0
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if ind > first:
                    break
                if row['Filename'] == '': continue
                try:
                    p = FeynmanProblem(row, gen=gen)
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