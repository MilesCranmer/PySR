"""Functions for denoising data during preprocessing."""
import numpy as np


def denoise(X, y, Xresampled=None, random_state=None):
    """Denoise the dataset using a Gaussian process."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    gp_kernel = RBF(np.ones(X.shape[1])) + WhiteKernel(1e-1) + ConstantKernel()
    gpr = GaussianProcessRegressor(
        kernel=gp_kernel, n_restarts_optimizer=50, random_state=random_state
    )
    gpr.fit(X, y)

    if Xresampled is not None:
        return Xresampled, gpr.predict(Xresampled)

    return X, gpr.predict(X)


def multi_denoise(X, y, Xresampled=None, random_state=None):
    """Perform `denoise` along each column of `y` independently."""
    y = np.stack(
        [
            denoise(X, y[:, i], Xresampled=Xresampled, random_state=random_state)[1]
            for i in range(y.shape[1])
        ],
        axis=1,
    )

    if Xresampled is not None:
        return Xresampled, y

    return X, y
