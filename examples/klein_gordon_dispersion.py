"""Klein-Gordon dispersion relation discovery using PySR.

Demonstrates symbolic regression rediscovering the relativistic dispersion
relation from wave simulation data.

The Klein-Gordon equation describes a free relativistic scalar field:
    d^2 psi / dt^2 = c^2 * nabla^2 psi - m^2 * psi

where c is the wave speed and m is the mass parameter.

Plane wave solutions psi ~ exp(i*(k*x - omega*t)) satisfy the dispersion
relation:
    omega^2 = c^2 * k^2 + m^2

This script generates (k, m) -> omega^2 data along with a small amount of
noise, then lets PySR rediscover the exact symbolic form.
"""

import numpy as np
from pysr import PySRRegressor


def generate_dispersion_data(
    n_samples: int = 500,
    c: float = 1.0,
    noise_level: float = 0.01,
    seed: int = 0,
) -> tuple:
    """Generate Klein-Gordon dispersion data.

    Parameters
    ----------
    n_samples : int
        Number of data points.
    c : float
        Wave speed.
    noise_level : float
        Relative noise amplitude.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        Columns: [wavenumber k, mass parameter m].
    y : np.ndarray of shape (n_samples,)
        Squared angular frequency omega^2.
    """
    rng = np.random.default_rng(seed)

    # Sample wavenumber and mass uniformly
    k = rng.uniform(0.1, 10.0, n_samples)
    m = rng.uniform(0.0, 5.0, n_samples)

    omega_sq = c**2 * k**2 + m**2
    omega_sq += noise_level * rng.normal(size=n_samples) * omega_sq

    X = np.column_stack([k, m])
    return X, omega_sq


def main():
    # ---- Data generation ----
    X, y = generate_dispersion_data(n_samples=500, c=1.0, noise_level=0.01)
    print(f"Generated {len(y)} data points.")
    print(f"  k range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  m range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    print(f"  omega^2 range: [{y.min():.2f}, {y.max():.2f}]")

    # ---- Symbolic regression: discover omega^2 = f(k, m) ----
    model = PySRRegressor(
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        niterations=40,
        populations=30,
        maxsize=15,
        variable_names=["k", "m"],
        extra_sympy_mappings={"square": lambda x: x**2},
    )

    model.fit(X, y)

    # ---- Results ----
    print("\n--- Discovered Equations ---")
    print(model)

    best = model.sympy()
    print(f"\nBest expression (sympy): {best}")
    print(f"Expected:               k**2 + m**2")

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f"Mean squared error: {mse:.6e}")


if __name__ == "__main__":
    main()
