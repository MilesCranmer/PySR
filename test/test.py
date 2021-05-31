import unittest
import numpy as np
from pysr import pysr, get_hof, best, best_tex, best_callable
import sympy
import pandas as pd

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.default_test_kwargs = dict(
            niterations=10,
            populations=4,
            user_input=False,
            annealing=True,
            useFrequency=False,
        )
        np.random.seed(0)
        self.X = np.random.randn(100, 5)
    
    def test_linear_relation(self):
        y = self.X[:, 0]
        equations = pysr(self.X, y, **self.default_test_kwargs)
        print(equations)
        self.assertLessEqual(equations.iloc[-1]['MSE'], 1e-4)

    def test_multioutput_custom_operator(self):
        y = self.X[:, [0, 1]]**2
        equations = pysr(self.X, y,
                         unary_operators=["sq(x) = x^2"], binary_operators=["plus"],
                         extra_sympy_mappings={'square': lambda x: x**2},
                         **self.default_test_kwargs)
        print(equations)
        self.assertLessEqual(equations[0].iloc[-1]['MSE'], 1e-4)
        self.assertLessEqual(equations[1].iloc[-1]['MSE'], 1e-4)

    def test_empty_operators_single_input(self):
        X = np.random.randn(100, 1)
        y = X[:, 0] + 3.0
        equations = pysr(X, y,
                         unary_operators=[], binary_operators=["plus"],
                         **self.default_test_kwargs)

        print(equations)
        self.assertLessEqual(equations.iloc[-1]['MSE'], 1e-4)

class TestBest(unittest.TestCase):
    def setUp(self):
        equations = pd.DataFrame({
            'Equation': ['1.0', 'cos(x0)', 'square(cos(x0))'],
            'MSE': [1.0, 0.1, 1e-5],
            'Complexity': [1, 2, 3]
            })

        equations['Complexity MSE Equation'.split(' ')].to_csv(
                'equation_file.csv.bkup', sep='|')

        self.equations = get_hof(
                'equation_file.csv', n_features=2,
                variables_names='x0 x1'.split(' '),
                extra_sympy_mappings={}, output_jax_format=False,
                multioutput=False, nout=1)

    def test_best(self):
        self.assertEqual(best(self.equations), sympy.cos(sympy.Symbol('x0'))**2)
        self.assertEqual(best(), sympy.cos(sympy.Symbol('x0'))**2)

    def test_best_tex(self):
        self.assertEqual(best_tex(self.equations), '\\cos^{2}{\\left(x_{0} \\right)}')
        self.assertEqual(best_tex(), '\\cos^{2}{\\left(x_{0} \\right)}')

    def test_best_lambda(self):
        X = np.random.randn(10, 2)
        y = np.cos(X[:, 0])**2
        for f in [best_callable(), best_callable(self.equations))]:
            np.testing.assert_almost_equal(f(X), y)
