import numpy as np

from dataclasses import dataclass

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass
class MapRangeSettings:
    width_min: int = 17
    width_max: int = 21
    height_min: int = 17
    height_max: int = 21
    obstacle_density_min: float = 0.1
    obstacle_density_max: float = 0.3

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "width": rng.integers(self.width_min, self.width_max + 1),
            "height": rng.integers(self.height_min, self.height_max + 1),
            "obstacle_density": rng.uniform(self.obstacle_density_min, self.obstacle_density_max),
            "seed": seed
        }


def generate_map(settings):
    rng = np.random.default_rng(settings["seed"])
    width, height, obstacle_density = settings["width"], settings["height"], settings["obstacle_density"]
    map_data = [['.' for _ in range(width)] for _ in range(height)]
    total_tiles = width * height
    total_obstacles = int(total_tiles * obstacle_density)

    obstacles_placed = 0
    while obstacles_placed < total_obstacles:
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        if map_data[y][x] == '.':
            map_data[y][x] = '#'
            obstacles_placed += 1

    return '\n'.join(''.join(row) for row in map_data)


def generate_and_save_maps(name_prefix, seed_range):
    test_maps = {}
    max_digits = len(str(max(seed_range)))
    settings_generator = MapRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = generate_map(settings)
        map_name = f"{name_prefix}-seed-{str(seed).zfill(max_digits)}"
        test_maps[map_name] = map_data

    maps_dict_to_yaml(f'{name_prefix}.yaml', test_maps)


def main():
    generate_and_save_maps("validation-random", range(0, 128))
    generate_and_save_maps("training-random", range(128, 128 + 512))


if __name__ == "__main__":
    main()
