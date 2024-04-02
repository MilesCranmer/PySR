import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from data import generate_data, read_csv

EMPTY_DF = lambda: pd.DataFrame(
    {
        "Equation": [],
        "Loss": [],
        "Complexity": [],
    }
)


def pysr_fit(queue: mp.Queue, out_queue: mp.Queue):
    import pysr

    while True:
        # Get the arguments from the queue, if available
        args = queue.get()
        if args is None:
            break
        X = args["X"]
        y = args["y"]
        kwargs = args["kwargs"]
        model = pysr.PySRRegressor(
            progress=False,
            timeout_in_seconds=1000,
            **kwargs,
        )
        model.fit(X, y)
        out_queue.put(None)


def pysr_predict(queue: mp.Queue, out_queue: mp.Queue):
    import pysr

    while True:
        args = queue.get()

        if args is None:
            break

        X = args["X"]
        equation_file = str(args["equation_file"])
        index = args["index"]

        equation_file_pkl = equation_file.replace(".csv", ".pkl")
        equation_file_bkup = equation_file + ".bkup"

        equation_file_copy = equation_file.replace(".csv", "_copy.csv")
        equation_file_pkl_copy = equation_file.replace(".csv", "_copy.pkl")

        # TODO: See if there is way to get lock on file
        os.system(f"cp {equation_file_bkup} {equation_file_copy}")
        os.system(f"cp {equation_file_pkl} {equation_file_pkl_copy}")

        try:
            model = pysr.PySRRegressor.from_file(equation_file_pkl_copy, verbosity=0)
        except pd.errors.EmptyDataError:
            continue

        ypred = model.predict(X, index)

        # Rename the columns to uppercase
        equations = model.equations_[["complexity", "loss", "equation"]].copy()

        # Remove any row that has worse loss than previous row:
        equations = equations[equations["loss"].cummin() == equations["loss"]]
        # TODO: Why is this needed? Are rows not being removed?

        equations.columns = ["Complexity", "Loss", "Equation"]
        out_queue.put(dict(ypred=ypred, equations=equations))


class ProcessWrapper:
    def __init__(self, target: Callable[[mp.Queue, mp.Queue], None]):
        self.queue = mp.Queue(maxsize=1)
        self.out_queue = mp.Queue(maxsize=1)
        self.process = mp.Process(target=target, args=(self.queue, self.out_queue))
        self.process.start()


PERSISTENT_WRITER = None
PERSISTENT_READER = None


def processing(
    file_input,
    force_run,
    test_equation,
    num_points,
    noise_level,
    data_seed,
    niterations,
    maxsize,
    binary_operators,
    unary_operators,
    plot_update_delay,
    parsimony,
    populations,
    population_size,
    ncycles_per_iteration,
    elementwise_loss,
    adaptive_parsimony_scaling,
    optimizer_algorithm,
    optimizer_iterations,
    batching,
    batch_size,
):
    """Load data, then spawn a process to run the greet function."""
    global PERSISTENT_WRITER
    global PERSISTENT_READER

    if PERSISTENT_WRITER is None:
        print("Starting PySR fit process")
        PERSISTENT_WRITER = ProcessWrapper(pysr_fit)

    if PERSISTENT_READER is None:
        print("Starting PySR predict process")
        PERSISTENT_READER = ProcessWrapper(pysr_predict)

    if file_input is not None:
        try:
            X, y = read_csv(file_input, force_run)
        except ValueError as e:
            return (EMPTY_DF(), str(e))
    else:
        X, y = generate_data(test_equation, num_points, noise_level, data_seed)

    tmpdirname = tempfile.mkdtemp()
    base = Path(tmpdirname)
    equation_file = base / "hall_of_fame.csv"
    # Check if queue is empty, if not, kill the process
    # and start a new one
    if not PERSISTENT_WRITER.queue.empty():
        print("Restarting PySR fit process")
        if PERSISTENT_WRITER.process.is_alive():
            PERSISTENT_WRITER.process.terminate()
            PERSISTENT_WRITER.process.join()

        PERSISTENT_WRITER = ProcessWrapper(pysr_fit)

    if not PERSISTENT_READER.queue.empty():
        print("Restarting PySR predict process")
        if PERSISTENT_READER.process.is_alive():
            PERSISTENT_READER.process.terminate()
            PERSISTENT_READER.process.join()

        PERSISTENT_READER = ProcessWrapper(pysr_predict)

    PERSISTENT_WRITER.queue.put(
        dict(
            X=X,
            y=y,
            kwargs=dict(
                niterations=niterations,
                maxsize=maxsize,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                equation_file=equation_file,
                parsimony=parsimony,
                populations=populations,
                population_size=population_size,
                ncycles_per_iteration=ncycles_per_iteration,
                elementwise_loss=elementwise_loss,
                adaptive_parsimony_scaling=adaptive_parsimony_scaling,
                optimizer_algorithm=optimizer_algorithm,
                optimizer_iterations=optimizer_iterations,
                batching=batching,
                batch_size=batch_size,
            ),
        )
    )
    while PERSISTENT_WRITER.out_queue.empty():
        if equation_file.exists():
            # First, copy the file to a the copy file
            PERSISTENT_READER.queue.put(
                dict(
                    X=X,
                    equation_file=equation_file,
                    index=-1,
                )
            )
            out = PERSISTENT_READER.out_queue.get()
            equations = out["equations"]
            yield equations[["Complexity", "Loss", "Equation"]]

        time.sleep(0.1)
