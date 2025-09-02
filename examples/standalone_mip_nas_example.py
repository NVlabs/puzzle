"""
Usage:
python -m examples.standalone_mip_nas_example
"""

from random import random
from puzzle.mip_nas import multi_solution_mip_nas


DEFAULT_NUM_LAYERS = 32
DEFAULT_NUM_CONFIGS_PER_BLOCK = 100
DEFAULT_MEAN_MEMORY_MIB = 100.0
DEFAULT_MEAN_RUNTIME_MS = 10.0


def mip_nas_example():
    block_library = generate_random_block_library()

    minimal_diversity = 10
    constraints = {
        "stats.memory_mib": DEFAULT_NUM_LAYERS * DEFAULT_MEAN_MEMORY_MIB,
        "stats.runtime_ms": DEFAULT_NUM_LAYERS * DEFAULT_MEAN_RUNTIME_MS,
    }
    solutions = multi_solution_mip_nas(
        block_library,
        objective="metrics.loss",
        constraints=constraints,
        bigger_is_better=False,
        num_solutions=10,
        minimal_diversity=minimal_diversity,
    )

    print()
    print(f"{constraints=}")

    print()
    print(
        "Printing solutions. Note that the total_value is increasing from solution to solution (lower is better), "
        "and that constraints are respected."
    )

    for i_solution, solution in enumerate(solutions):
        print()
        print(f"solution {i_solution}:")
        for var_name, var_value in solution.items():
            print(f"{var_name}={var_value}")

    print(
        f"\nChecking differences between solutions, should be at least {minimal_diversity}:"
    )
    for a in range(len(solutions)):
        for b in range(a + 1, len(solutions)):
            num_differences = 0
            for layer_id in block_library.keys():
                num_differences += (
                    solutions[a]["chosen_block_variants"][layer_id]
                    != solutions[b]["chosen_block_variants"][layer_id]
                )

            print(a, "<>", b, "=", num_differences)


def generate_random_block_library(
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_configs_per_block: int = DEFAULT_NUM_CONFIGS_PER_BLOCK,
    mean_memory_mib: float = DEFAULT_MEAN_MEMORY_MIB,
    mean_runtime_ms: float = DEFAULT_MEAN_RUNTIME_MS,
) -> dict:
    return {
        f"layer_{i_layer}": {
            f"block_config_{i_config}": {
                "metrics": {"loss": random()},
                "stats": {
                    "memory_mib": random() * 2 * mean_memory_mib,
                    "runtime_ms": random() * 2 * mean_runtime_ms,
                },
            }
            for i_config in range(num_configs_per_block)
        }
        for i_layer in range(num_layers)
    }


if __name__ == "__main__":
    mip_nas_example()
