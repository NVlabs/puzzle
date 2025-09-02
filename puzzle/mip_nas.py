from collections import defaultdict
import warnings
from typing import TypeAlias, Optional, Hashable, Union, Any, Iterable

from mip import Model, xsum, maximize, BINARY, minimize
import math
from tqdm import tqdm

DEFAULT_MAX_SECONDS_PER_SOLUTION = 60.0


# fmt: off
BlockConfig: TypeAlias = Hashable  # for example "attention = 8 kv heads (parent * 100%) && ffn = intermediate size 14e3 (parent * 50%)" or "attention = no_op (parent * 0%) && ffn = no_op (parent * 0%)"
BlockVariantStats: TypeAlias = dict[str, Any]  # the scores and costs of replacing a specific parent layer with this block variant
LayerOptions: TypeAlias = dict[BlockConfig, BlockVariantStats]  # which block variants are available for this layer

LayerId: TypeAlias = Hashable  # e.g. layer_22
BlockLibrary: TypeAlias = dict[LayerId, LayerOptions]  # each layer and its available block variants
ChosenBlockVariants: TypeAlias = dict[LayerId, BlockConfig]  # which block variant was chosen for each layer
# fmt: on


def mip_nas(
    block_library: BlockLibrary,
    objective: str,
    constraints: dict[str, float | tuple[float, float]],
    bigger_is_better: bool,
    previous_solutions: Optional[list[ChosenBlockVariants]] = None,
    minimal_diversity: int = 1,
    max_seconds_per_solution: Optional[float] = DEFAULT_MAX_SECONDS_PER_SOLUTION,
) -> tuple[ChosenBlockVariants, float, dict[str, float]]:
    """
    Given a block library of possible block variants per layer, an objective metric and a set of constraints,
    this function solves a Mixed Integer Program (MIP) problem to find a valid model architecture that satisfies
    the constraints and optimizes the objective.

    Args:
        block_library (BlockLibrary): Mapping from layer id to the available block variants and their statistics.
            See examples/data/Llama-3.3-70B-Instruct/block_library.json
            Example:
            {
                "layer_0": {
                    "attention = 8 kv heads (parent * 100%) && ffn = intermediate size 14e3 (parent * 50%)": {
                        "stats": {
                            "runtime_ms": 580.1637696533203,
                            "memory_mib": 1101.109375,
                        },
                        "metrics": {
                            "kl_div": 0.06432564748683944,
                            "lm_loss": 1.6424870013725013,
                        },
                    }
                    # ...  more variants for layer_0
                }
                # ... more layers
            }
        objective (str): Dot-separated key into each block's stats indicating
            the scalar value to optimize, e.g. metrics.kl_div or metrics.lm_loss.
        constraints (dict[str, float | tuple[float, float]]): For each
            constraint that you're interested in, provide either a single float
            interpreted as a maximum (<=), or a (min_cost, max_cost) tuple to
            enforce both bounds.
        bigger_is_better (bool): If True, maximize the objective (for accuracy-like objectives);
            If False, minimize the objective (for loss-like objectives).
        previous_solutions (Optional[list[ChosenBlockVariants]]): Previously
            found solutions. When provided, the solver enforces that the new
            solution differs from each previous one in at least `minimal_diversity` layers.
            No need to use directly, let multi_solution_mip_nas() handle this. 
        minimal_diversity (int): Minimum number of layers that must differ from
            each solution in `previous_solutions`. Only applies when
            `previous_solutions` is not None. Defaults to 1.
        max_seconds_per_solution (Optional[float]): Time limit for the solver.
            Set to None to disable the time limit. Defaults to
            DEFAULT_MAX_SECONDS_PER_SOLUTION.

    Returns:
        tuple[ChosenBlockVariants, float, dict[str, float]]: A tuple containing:
            - chosen_block_variants: Selected block config per layer.
            - total_value: Sum of the objective values over chosen blocks.
            - total_costs: Aggregated cost per constraint key.

    Raises:
        InfeasibleError: If no feasible solution is found within the constraints and/or time limit.
        AssertionError: If post-solve verification finds a violated constraint.
                        Should not happen in practice, consider this a built-in test.
    """

    ############
    ## Declare optimization variables, gather score terms and cost terms
    ############

    mip_model = Model()
    mip_model.verbose = 0

    # indicator_vars: for each layer i, a dict of x_i,j that determine whether block variant j is chosen for layer i.
    indicator_vars = defaultdict(dict)

    # score_terms - score(i,j) * x_i,j: their sum is what we try to optimize.
    score_terms = []

    # cost_terms - cost(i,j) * x_i,j: for each type of cost (e.g. memory, latency),
    # we will later force the sum of its cost_terms to be between [min_cost, max_cost].
    cost_terms = defaultdict(list)

    # declare indicator_vars and gather score_terms and cost_terms:
    for layer_id, layer_options in block_library.items():
        for block_config, block_variant_stats in layer_options.items():
            is_chosen = mip_model.add_var(var_type=BINARY)
            indicator_vars[layer_id][block_config] = is_chosen

            score = get_nested_key(block_variant_stats, objective)
            score_term = score * is_chosen
            score_terms.append(score_term)

            for constraint_key in constraints.keys():
                cost = get_nested_key(block_variant_stats, constraint_key)
                cost_term = cost * is_chosen
                cost_terms[constraint_key].append(cost_term)

    ############
    ## Build the MIP model: objective and constraints
    ############

    # Objective
    mip_model.objective = (
        maximize(xsum(score_terms)) if bigger_is_better else minimize(xsum(score_terms))
    )

    # Layer validity constraints: exactly 1 block variant chosen per layer
    for layer_id in block_library.keys():
        layer_indicator_vars = list(indicator_vars[layer_id].values())
        mip_model += xsum(layer_indicator_vars) == 1

    # Cost constraints: sum of costs for each constraint must be between [min_cost, max_cost]
    for constraint_key, cost_range in constraints.items():
        if isinstance(cost_range, Iterable):
            min_cost, max_cost = cost_range
        else:
            min_cost = None
            max_cost = cost_range

        if min_cost is not None:
            mip_model += xsum(cost_terms[constraint_key]) >= min_cost
        if max_cost is not None:
            mip_model += xsum(cost_terms[constraint_key]) <= max_cost

    # Diversity constraints: at least minimal_diversity layers must be different from all previous solutions
    if previous_solutions is not None:
        for previous_chosen_block_variants in previous_solutions:
            corresponding_vars = [
                indicator_vars[layer_id][block_config]
                for layer_id, block_config in previous_chosen_block_variants.items()
            ]
            mip_model += (
                xsum(corresponding_vars) <= len(block_library) - minimal_diversity
            )

    ############
    ## Run the MIP search
    ############

    if max_seconds_per_solution is not None:
        mip_model.max_seconds = max_seconds_per_solution

    mip_model.optimize()

    if is_chosen.x is None:
        raise InfeasibleError()

    ############
    ## Infer the solution from the MIP output and calculate total value and costs
    ############

    total_value = 0.0
    total_costs = {constraint_key: 0 for constraint_key in constraints.keys()}
    chosen_block_variants: ChosenBlockVariants = dict()
    for layer_id, layer_options in block_library.items():
        for block_config, block_variant_stats in layer_options.items():
            is_chosen = indicator_vars[layer_id][block_config]
            is_chosen = is_chosen.x >= 0.99
            if is_chosen:
                assert layer_id not in chosen_block_variants, (
                    "The layer validity constraint was violated, this shouldn't happen"
                )
                chosen_block_variants[layer_id] = block_config
                score = get_nested_key(block_variant_stats, objective)
                total_value += score
                for constraint_key in constraints.keys():
                    cost = get_nested_key(block_variant_stats, constraint_key)
                    total_costs[constraint_key] += cost

    ############
    ## Verify solution validity
    ############

    # Verify that every layer exists in the solution
    if len(chosen_block_variants) != len(block_library):
        layers_missing_from_library = set(chosen_block_variants.keys()) - set(
            block_library.keys()
        )  # really shouldn't happen
        layers_missing_from_solution = set(block_library.keys()) - set(
            chosen_block_variants.keys()
        )
        block_library_of_missing_layers = [
            block_library[key] for key in layers_missing_from_solution
        ]
        raise AssertionError(
            f"""
        Different number of layers in 'chosen_block_variants' and 'block_library': {len(chosen_block_variants)=}  {len(block_library)=}
        {layers_missing_from_solution=}
        {layers_missing_from_library=}
        {block_library_of_missing_layers=}
        """
        )

    # Verify that the solution satisfies the cost constraints
    for constraint_key, cost_range in constraints.items():
        if isinstance(cost_range, Iterable):
            min_cost, max_cost = cost_range
        else:
            min_cost = None
            max_cost = cost_range

        cost = total_costs[constraint_key]

        if max_cost is not None:
            assert cost < max_cost or math.isclose(cost, max_cost, rel_tol=1e-9), (
                f"A max_cost was violated: {constraint_key=}, {cost=} > {max_cost=}"
            )
        if min_cost is not None:
            assert cost > min_cost or math.isclose(cost, min_cost, rel_tol=1e-9), (
                f"A min_cost was violated: {constraint_key=}, {cost=} < {min_cost=}"
            )

    # Verify that the solution is diverse from all previous solutions
    if previous_solutions is not None:
        for previous_chosen_block_variants in previous_solutions:
            num_differences = 0
            for layer_id in block_library.keys():
                num_differences += (
                    previous_chosen_block_variants[layer_id]
                    != chosen_block_variants[layer_id]
                )
            assert num_differences >= minimal_diversity

    ######
    ## All done!
    ######

    return chosen_block_variants, total_value, total_costs


def multi_solution_mip_nas(
    block_library: BlockLibrary,
    objective: str,
    constraints: dict[str, float],
    bigger_is_better: bool,
    num_solutions: int = 1,
    minimal_diversity: int = 1,
    max_seconds_per_solution: Optional[float] = DEFAULT_MAX_SECONDS_PER_SOLUTION,
) -> list[dict[str, Union[ChosenBlockVariants, float]]]:
    """
    Runs mip_nas() multiple times to obtain multiple diverse solutions to the same optimization problem.
    The solutions are all different from each other by at least minimal_diversity layers.
    """

    solutions = []
    previous_solutions: list[ChosenBlockVariants] = []
    for i_run in tqdm(range(num_solutions), desc="multi_solution_mip_nas"):
        try:
            chosen_block_variants, total_value, total_costs = mip_nas(
                block_library,
                objective,
                constraints,
                bigger_is_better,
                previous_solutions,
                minimal_diversity,
                max_seconds_per_solution,
            )
        except InfeasibleError:
            warnings.warn(
                f"Found only {i_run} feasible solutions (requested {num_solutions})"
            )
            break
        previous_solutions.append(chosen_block_variants)
        solutions.append(
            {
                "chosen_block_variants": chosen_block_variants,
                "total_value": total_value,
                "total_costs": total_costs,
            }
        )
    return solutions


class InfeasibleError(Exception):
    pass


def get_nested_key(dictionary: dict[str, Any], nested_key: str) -> Any:
    """
    If nested_key is "a.b.c" returns dictionary["a"]["b"]["c"]
    """
    value = dictionary
    for key in nested_key.split("."):
        value = value[key]
    return value
