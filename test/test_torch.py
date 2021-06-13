import unittest
import numpy as np
import pandas as pd
from pysr import sympy2torch, get_hof
import torch
import sympy


class TestTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sympy2torch(self):
        x, y, z = sympy.symbols("x y z")
        cosx = 1.0 * sympy.cos(x) + y
        X = torch.tensor(np.random.randn(1000, 3))
        true = 1.0 * torch.cos(X[:, 0]) + X[:, 1]
        torch_module = sympy2torch(cosx, [x, y, z])
        self.assertTrue(
            np.all(np.isclose(torch_module(X).detach().numpy(), true.detach().numpy()))
        )

    def test_pipeline(self):
        X = np.random.randn(100, 10)
        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "cos(x0)", "square(cos(x0))"],
                "MSE": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        equations["Complexity MSE Equation".split(" ")].to_csv(
            "equation_file.csv.bkup", sep="|"
        )

        equations = get_hof(
            "equation_file.csv",
            n_features=2,
            variables_names="x1 x2 x3".split(" "),
            extra_sympy_mappings={},
            output_torch_format=True,
            multioutput=False,
            nout=1,
            selection=[1, 2, 3],
        )

        tformat = equations.iloc[-1].torch_format
        np.testing.assert_almost_equal(
            tformat(torch.tensor(X)).detach().numpy(),
            np.square(np.cos(X[:, 1])),  # Selection 1st feature
            decimal=4,
        )

    def test_mod_mapping(self):
        x, y, z = sympy.symbols("x y z")
        expression = x ** 2 + sympy.atanh(sympy.Mod(y + 1, 2) - 1) * 3.2 * z

        module = sympy2torch(expression, [x, y, z])

        X = torch.rand(100, 3).float() * 10

        true_out = (
            X[:, 0] ** 2 + torch.atanh(torch.fmod(X[:, 1] + 1, 2) - 1) * 3.2 * X[:, 2]
        )
        torch_out = module(X)

        np.testing.assert_array_almost_equal(
            true_out.detach(), torch_out.detach(), decimal=4
        )
