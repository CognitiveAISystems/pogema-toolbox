from typing import Any, Dict, Generator, Tuple
import copy
from itertools import product


def generate_variants(spec: Dict) -> Generator[Tuple[Dict, Dict], None, None]:
    for variant, changes in _expand_grid_search(spec):
        if not _contains_unresolved_values(variant):
            yield changes, variant


def _expand_grid_search(spec: Dict) -> Generator[Tuple[Dict, Dict], None, None]:
    grid_params = _identify_grid_params(spec)
    if not grid_params:
        yield spec, {}
        return

    for combination in _generate_combinations(grid_params):
        variant, changes = _apply_combination(copy.deepcopy(spec), combination)
        yield variant, changes


def _identify_grid_params(spec: Dict, path=()) -> Dict:
    grid_params = {}
    for key, value in spec.items():
        current_path = path + (key,)
        if isinstance(value, dict):
            if "grid_search" in value:
                grid_params[current_path] = value["grid_search"]
            else:
                grid_params.update(_identify_grid_params(value, current_path))
    return grid_params


def _generate_combinations(grid_params: Dict) -> Generator[Dict, None, None]:
    keys, value_lists = zip(*grid_params.items())
    for values in product(*value_lists):
        yield dict(zip(keys, values))


def _apply_combination(spec: Dict, combination: Dict) -> Tuple[Dict, Dict]:
    changes = {}
    for path, value in combination.items():
        _set_dict_value(spec, path, value)
        changes[path] = value
    return spec, changes


def _set_dict_value(d: Dict, path: Tuple, value: Any):
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _contains_unresolved_values(spec: Dict) -> bool:
    return any("grid_search" in value for _, value in _traverse_dict(spec) if isinstance(value, dict))


def _traverse_dict(d: Dict, path=()) -> Generator[Tuple[Tuple, Any], None, None]:
    for key, value in d.items():
        current_path = path + (key,)
        if isinstance(value, dict):
            yield from _traverse_dict(value, current_path)
        else:
            yield current_path, value


def main():
    environment_spec = {
        "name": "DecMAPF-v0",
        "grid_config": {
            "on_target": "restart",
            "max_episode_steps": 512,
            "map_name": {"grid_search": ["map-a", "map-b", "map-c", ]},
            "num_agents": {"grid_search": [8, 16, 32]},
            "seed": {"grid_search": [0, 1, 2]}
        }
    }

    # Generate and print the first 3 variant pairs to demonstrate the functionality
    variant_pairs = list(generate_variants(environment_spec))
    for changes, variant in variant_pairs[:3]:
        print(changes, variant)


if __name__ == '__main__':
    main()
