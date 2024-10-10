import os

import numpy as np
import paddle
from paddle import nn
from paddle.io import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.environ["PYTHON_JULIACALL_THREADS"] = "1"

rstate = np.random.RandomState(0)

N = 100000
Nt = 10
X = 6 * rstate.rand(N, Nt, 5) - 3
y_i = X[..., 0] ** 2 + 6 * np.cos(2 * X[..., 2])
y = np.sum(y_i, axis=1) / y_i.shape[1]
z = y**2


hidden = 128
total_steps = 50_000


def mlp(size_in, size_out, act=nn.ReLU):
    return nn.Sequential(
        nn.Linear(size_in, hidden),
        act(),
        nn.Linear(hidden, hidden),
        act(),
        nn.Linear(hidden, hidden),
        act(),
        nn.Linear(hidden, size_out),
    )


class SumNet(nn.Layer):
    def __init__(self):
        super().__init__()

        ########################################################
        # The same inductive bias as above!
        self.g = mlp(5, 1)
        self.f = mlp(1, 1)

    def forward(self, x):
        y_i = self.g(x)[:, :, 0]
        y = paddle.sum(y_i, axis=1, keepdim=True) / y_i.shape[1]
        z = self.f(y)
        return z[:, 0]


Xt = paddle.to_tensor(X).astype("float32")
zt = paddle.to_tensor(z).astype("float32")
X_train, X_test, z_train, z_test = train_test_split(Xt, zt, random_state=0)
train_set = TensorDataset([X_train, z_train])
train = DataLoader(train_set, batch_size=128, shuffle=True)
test_set = TensorDataset([X_test, z_test])
test = DataLoader(test_set, batch_size=256)

paddle.seed(0)

model = SumNet()
max_lr = 1e-2
model = paddle.Model(model)
scheduler = paddle.optimizer.lr.OneCycleLR(
    max_learning_rate=max_lr, total_steps=total_steps, divide_factor=1e4
)
optim = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
model.prepare(optim, paddle.nn.MSELoss())
model.fit(train, test, num_iters=total_steps, eval_freq=1000)

np.random.seed(0)
idx = np.random.randint(0, 10000, size=1000)

X_for_pysr = Xt[idx]
y_i_for_pysr = model.network.g(X_for_pysr)[:, :, 0]
y_for_pysr = paddle.sum(y_i_for_pysr, axis=1) / y_i_for_pysr.shape[1]
z_for_pysr = zt[idx]  # Use true values.


nnet_recordings = {
    "g_input": X_for_pysr.detach().cpu().numpy().reshape(-1, 5),
    "g_output": y_i_for_pysr.detach().cpu().numpy().reshape(-1),
    "f_input": y_for_pysr.detach().cpu().numpy().reshape(-1, 1),
    "f_output": z_for_pysr.detach().cpu().numpy().reshape(-1),
}

# Save the data for later use:
import pickle as pkl

with open("nnet_recordings.pkl", "wb") as f:
    pkl.dump(nnet_recordings, f)

import pickle as pkl

nnet_recordings = pkl.load(open("nnet_recordings.pkl", "rb"))
f_input = nnet_recordings["f_input"]
f_output = nnet_recordings["f_output"]
g_input = nnet_recordings["g_input"]
g_output = nnet_recordings["g_output"]


rstate = np.random.RandomState(0)
f_sample_idx = rstate.choice(f_input.shape[0], size=500, replace=False)
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=50,
    binary_operators=["+", "-", "*"],
    unary_operators=["cos", "square"],
)
model.fit(g_input[f_sample_idx], g_output[f_sample_idx])

model.equations_[["complexity", "loss", "equation"]]
