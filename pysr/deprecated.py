"""Various functions to deprecate features."""


def make_deprecated_kwargs_for_pysr_regressor():
    """Create dict of deprecated kwargs."""

    deprecation_string = """
        fractionReplaced => fraction_replaced
        fractionReplacedHof => fraction_replaced_hof
        npop => population_size
        hofMigration => hof_migration
        shouldOptimizeConstants => should_optimize_constants
        weightAddNode => weight_add_node
        weightDeleteNode => weight_delete_node
        weightDoNothing => weight_do_nothing
        weightInsertNode => weight_insert_node
        weightMutateConstant => weight_mutate_constant
        weightMutateOperator => weight_mutate_operator
        weightRandomize => weight_randomize
        weightSimplify => weight_simplify
        crossoverProbability => crossover_probability
        perturbationFactor => perturbation_factor
        batchSize => batch_size
        warmupMaxsizeBy => warmup_maxsize_by
        useFrequency => use_frequency
        useFrequencyInTournament => use_frequency_in_tournament
    """
    # Turn this into a dict:
    deprecated_kwargs = {}
    for line in deprecation_string.splitlines():
        line = line.replace(" ", "")
        if line == "":
            continue
        old, new = line.split("=>")
        deprecated_kwargs[old] = new

    return deprecated_kwargs
