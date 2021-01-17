from sklearn.linear_model import LinearRegression
from Transformation import *
from Truth import *
import itertools
import warnings
import traceback


def gen_valid_points(oracle, npoints=20, default_min=0.5, default_max=30):
    """
    Generates valid dataset (npoints, dim)
    """
    dim = oracle.nvariables
    # print(f"Dim {dim}, {oracle.nvariables}")
    # print(f"Oracle has {oracle} {oracle.variable_names}")
    mins = []
    maxes = []
    for r in oracle.ranges:
        if r is None:
            mins.append(default_min)
            maxes.append(default_max)
        else:
            mins.append(r[0])
            maxes.append(r[1])
    return np.random.uniform(low=mins, high=maxes, size=(npoints, dim))


def discover(transformation, oracle, npoints=20, threshold=0.98, timeout=5):
    """
    Constraint is a class child of the Class parent Constraint

    Oracle is a class which has a variable nvariables i.e number of inputs and a function f which performs f(X)
        f(X) must be of shape (n, nvariables)

    npoints: number of data points to train the weak model with

    threshold: minimum accuracy of weak model to say that a constraint has been found

    timeout: If the random generator cannot find a valid input in timeout seconds we quit
    """
    # Get random 10 points from some range
    start = time()
    sat = False
    while not sat and time() - start < timeout:
        try:
            points = gen_valid_points(oracle, npoints)
            y_original = oracle.f(points)
            if any(np.isnan(y_original)) or any(np.isinf(y_original)):
                print(points, points.shape, oracle)
                print(y_original)
                break
                raise ValueError()
            sat = True
        except:
            traceback.print_stack()
    if not sat:
        warnings.warn(f"Could not find an input that worked for oracle - ({oracle})")
        return False, None
    # print(points)
    X = transformation.transform(points)
    try:
        y = oracle.f(X)
        if any(np.isnan(y)) or any(np.isinf(y)):
            raise ValueError()
    except:
        # If the oracle cannot evaluate this input because of an out of domain error
        return False, None
    model, score = weak_learner(X, y, y_original)
    if score > threshold:
        return True, Truth(transformation, model)
    else:
        return False, Truth(transformation, model)


def weak_learner(X, y, y_original):
    """
    Takes in X, y and returns a weak learner that tries to fit the training data and its associated R^2 score as well as the model itself
    """

    y_original = np.reshape(y_original, newshape=(len(y_original), 1))
    # print(X.shape, y_original.shape)
    new_X = np.append(X, y_original, axis=1)

    model = LinearRegression()
    model.fit(new_X, y)
    # Force the model to be simple by rounding coefficients to 2 decimal points
    model.coef_ = np.round(model.coef_, 2)
    model.intercept_ = np.round(model.intercept_, 2)

    score = model.score(new_X, y)
    return model, score


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def multiprocess_task(transformation, oracle):
    """
    Takes in a constraint and oracle and returns (constraint, model) if the value from discover is true else returns None
    """
    value, truth = discover(transformation, oracle)
    if value == True:
        return truth
    else:
        return None


def naive_procedure(oracle):
    """
    Takes in an oracle and gives out an exhaustive list of form [(constraint, model)] for all true constraints
    """
    nvariables = oracle.nvariables
    var_list = range(nvariables)
    pairs = itertools.combinations(var_list, r=2)
    sets = [x for x in powerset(var_list) if len(x) > 0]
    final = []
    transformations = []
    for pair in pairs:
        transformations.append(SymTransformation(pair[0], pair[1]))
        pass
    for smallset in sets:
        if len(smallset) > 1:
            transformations.append(ValueTransformation(smallset))
        transformations.append(ZeroTransformation(smallset))

        pass
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #    args = [(constraint, oracle) for constraint in constraints]
    #    results = executor.map(lambda x: multiprocess_task(*x), args)

    temp = [multiprocess_task(transformation, oracle) for transformation in transformations]
    for t in temp:
        if t is not None:
            final.append(t)
    return final


def process_from_problems(problems):
    ids = []
    forms = []
    ns = []
    for problem in problems:
        nvariables = problem.n_vars
        form = problem.form
        variable_names = problem.var_names
        id = problem.eq_id

        oracle = Oracle(nvariables, form=form, variable_names=variable_names, id=id)
        ids.append(oracle.id)
        forms.append(oracle.form)
        ns = len(naive_procedure(oracle))
    d = {"id": ids, "form": forms, "Number of Constraints": ns}
    return d


def process_from_form_and_names(form, variable_names):
    """
    Returns a julia string which declares an array called TRUTHS
    """
    if form is None or variable_names is None:
        return "TRUTHS = []"
    nvars = len(variable_names)
    oracle = Oracle(nvariables=nvars, form=form, variable_names=variable_names)
    truths = naive_procedure(oracle)
    print("Discovered the following Auxiliary Truths")
    for truth in truths:
        print(truth)
    julia_string = "TRUTHS = ["
    for truth in truths:
        addition = truth.julia_string()
        julia_string = julia_string + addition + ", "
    julia_string = julia_string + "]"
    return julia_string


if __name__ == "__main__":
    from Transformation import  SymTransformation
    from Oracle import Oracle
    from time import time

    variable_names = ["alpha", "beta"]
    form = "alpha * beta"
    nvariables = len(variable_names)
    # range_restriction={2: (1, 20)}
    oracle = Oracle(nvariables, form=form, variable_names=variable_names)
    now = time()
    finals = naive_procedure(oracle)
    end = time()
    print(finals)
    print(end - now)
