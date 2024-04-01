import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

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


class PySRProcess:
    def __init__(self):
        self.queue = mp.Queue()
        self.out_queue = mp.Queue()
        self.process = mp.Process(target=pysr_fit, args=(self.queue, self.out_queue))
        self.process.start()


PERSISTENT_WRITER = None


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
    if PERSISTENT_WRITER is None:
        print("Starting PySR process")
        PERSISTENT_WRITER = PySRProcess()

    if file_input is not None:
        try:
            X, y = read_csv(file_input, force_run)
        except ValueError as e:
            return (EMPTY_DF(), str(e))
    else:
        X, y = generate_data(test_equation, num_points, noise_level, data_seed)

    with tempfile.TemporaryDirectory() as tmpdirname:
        base = Path(tmpdirname)
        equation_file = base / "hall_of_fame.csv"
        equation_file_bkup = base / "hall_of_fame.csv.bkup"
        # Check if queue is empty, if not, kill the process
        # and start a new one
        if not PERSISTENT_WRITER.queue.empty():
            print("Restarting PySR process")
            if PERSISTENT_WRITER.process.is_alive():
                PERSISTENT_WRITER.process.terminate()
                PERSISTENT_WRITER.process.join()

            PERSISTENT_WRITER = PySRProcess()
        # Write these to queue instead:
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
        last_yield_time = None
        while PERSISTENT_WRITER.out_queue.empty():
            if equation_file_bkup.exists():
                try:
                    # First, copy the file to a the copy file
                    equation_file_copy = base / "hall_of_fame_copy.csv"
                    os.system(f"cp {equation_file_bkup} {equation_file_copy}")
                    equations = pd.read_csv(equation_file_copy)
                    # Ensure it is pareto dominated, with more complex expressions
                    # having higher loss. Otherwise remove those rows.
                    # TODO: Not sure why this occurs; could be the result of a late copy?
                    equations.sort_values("Complexity", ascending=True, inplace=True)
                    equations.reset_index(inplace=True)
                    bad_idx = []
                    min_loss = None
                    for i in equations.index:
                        if min_loss is None or equations.loc[i, "Loss"] < min_loss:
                            min_loss = float(equations.loc[i, "Loss"])
                        else:
                            bad_idx.append(i)
                    equations.drop(index=bad_idx, inplace=True)

                    while (
                        last_yield_time is not None
                        and time.time() - last_yield_time < plot_update_delay
                    ):
                        time.sleep(0.1)

                    yield equations[["Complexity", "Loss", "Equation"]]

                    last_yield_time = time.time()
                except pd.errors.EmptyDataError:
                    pass
