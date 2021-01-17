import numpy as np


class Oracle:
    oracle_d = {'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi, 'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
                'tanh': np.tanh, 'ln': np.log, 'arcsin': np.arcsin}

    def __init__(self, nvariables, f=None, form=None, variable_names=None, range_restriction={}, id=None):
        """
        nvariables: is the number of variables the function takes in
        f: takes in an X of shape (n, nvariables) and returns f(X) of shape (n,)
        form: String Def of the function
        variable_names: variable names used in form
        Range_restrictions: Dictionary of form {variable_index: (low, high)}
        """
        self.nvariables = nvariables
        if f is None and form is None:
            raise ValueError("f and form are both none in Oracle initialization. Specify at least one")
        if f is not None and form is not None:
            raise ValueError("f and form are both not none, pick only one")
        if form is not None and variable_names is None:
            raise ValueError("If form is provided then variable_names must also be provided")
        if form is not None:
            self.form = form
            self.variable_names = variable_names
            self.use_func = False
            self.d = Oracle.oracle_d.copy()
            for var_name in variable_names:
                self.d[var_name] = None
        else:
            # f is not None
            self.func = f
            self.use_func = True

        self.ranges = []
        for i in range(nvariables):
            if i in range_restriction:
                self.ranges.append(range_restriction[i])
            else:
                self.ranges.append(None)

        if id is not None:
            self.id = id
        return

    def f(self, X):
        """
        X is of shape (n, nvariables)
        """
        if self.invalid_input(X):
            raise ValueError("Invalid input to Oracle")
        if self.use_func:
            return self.func(X)
        else:
            return self.form_f(X)

    def form_f(self, X):
        """
        Returns the function output using form
        """
        for i, var in enumerate(self.variable_names):
            self.d[var] = X[:, i]
        return eval(self.form, self.d)

    def invalid_input(self, X):
        """
        Returns true if any of the following are true
            X has more or less variables than nvariables
            X has a value in a restricted range variable outside said range
        """
        if X.shape[1] != self.nvariables:
            return True
        for i, r in enumerate(self.ranges):
            if r is None:
                continue
            else:
                low = r[0]
                high = r[1]
                low_check = all(low <= X[:, i])
                high_check = all(X[:, i] <= high)
                if not low_check or not high_check:
                    return True

    def __str__(self):
        if self.id:
            return str(self.id)
        elif self.form:
            return str(self.form)
        else:
            return "<Un named Oracle>"

    def from_problem(problem):
        """
        Static function to return an oracle when given an instance of class problem.
        """
        return Oracle(nvariables=problem.n_vars, f=None, form=problem.form, variable_names=problem.var_names,
                      id=problem.eq_id)