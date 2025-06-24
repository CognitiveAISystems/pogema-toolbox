from collections import deque

from dataclasses import dataclass
import numpy as np

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass
class HouseRangeSettings:
    width_min: int = 10
    width_max: int = 20

    height_min: int = 10
    height_max: int = 20

    obstacle_ratio_min: int = 4
    obstacle_ratio_max: int = 8

    remove_edge_ratio_min: int = 4
    remove_edge_ratio_max: int = 10

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        width = rng.integers(self.width_min, self.width_max + 1)
        height = rng.integers(self.height_min, self.height_max + 1)
        obstacle_ratio = rng.integers(self.obstacle_ratio_min, self.obstacle_ratio_max + 1)
        remove_edge_ratio = rng.integers(self.remove_edge_ratio_min, self.remove_edge_ratio_max + 1)

        return {
            "width": width,
            "height": height,
            "obstacle_ratio": obstacle_ratio,
            "remove_edge_ratio": remove_edge_ratio,
            "seed": seed,
        }


def array_to_string(array_maze):
    result = []
    for line in array_maze:
        result.append("".join(['#' if x == -1 else '.' for x in line]))

    map_str = '\n'.join([''.join(row) for row in result])
    return map_str


# The code is adapted from ALPHA paper: https://github.com/marmotlab/ALPHA/blob/main/map_generator.py
class HouseGenerator:

    @staticmethod
    def _generate_sparse_coordinates(valid_coordinates, count, rng):
        coordinates = []
        valid_coordinates = list(valid_coordinates)  # Ensure indexable
        while len(coordinates) < count:
            candidate = rng.choice(valid_coordinates)
            if all(abs(candidate - existing) > 1 for existing in coordinates):
                coordinates.append(candidate)
        return coordinates

    @staticmethod
    def _build_edges(world, primary, secondary, is_row, edge_list):
        height, width = world.shape
        size = width if is_row else height  # Respect the axis orientation

        for i in primary:
            edge = []
            for j in range(size):
                x, y = (i, j) if is_row else (j, i)
                world[x][y] = -1
                if j not in secondary:
                    edge.append([x, y])
                elif edge:
                    edge_list.append(edge)
                    edge = []
            if edge:
                edge_list.append(edge)

    @staticmethod
    def label_connected_regions(grid, background=-1):
        visited = np.zeros_like(grid, dtype=bool)
        height, width = grid.shape
        count = 0

        for i in range(height):
            for j in range(width):
                if grid[i, j] == background or visited[i, j]:
                    continue

                queue = deque()
                queue.append((i, j))
                visited[i, j] = True

                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < height and 0 <= ny < width and
                                not visited[nx, ny] and grid[nx, ny] != background):
                            visited[nx, ny] = True
                            queue.append((nx, ny))

                count += 1

        return count

    @classmethod
    def generate(cls, width, height, obstacle_ratio, remove_edge_ratio, seed=None):
        width += 2
        height += 2

        rng = np.random.default_rng(seed)

        house_grid = np.zeros((height, width))

        valid_x = range(2, height - 2)
        valid_y = range(2, width - 2)

        num_obstacles = min(width, height) // obstacle_ratio

        obs_corner_x = cls._generate_sparse_coordinates(valid_x, num_obstacles, rng) + [0, height - 1]
        obs_corner_y = cls._generate_sparse_coordinates(valid_y, num_obstacles, rng) + [0, width - 1]

        obs_edge = []
        cls._build_edges(house_grid, obs_corner_x, obs_corner_y, is_row=True, edge_list=obs_edge)
        cls._build_edges(house_grid, obs_corner_y, obs_corner_x, is_row=False, edge_list=obs_edge)

        removable = rng.choice(len(obs_edge), len(obs_edge) // remove_edge_ratio, replace=False)
        for idx in removable:
            for x, y in obs_edge[idx]:
                house_grid[x][y] = 0

        for edge in obs_edge:
            if len(edge) <= max(1, min(width, height) // 20):
                for x, y in edge:
                    house_grid[x][y] = 0

        count = cls.label_connected_regions(house_grid, background=-1)

        while count != 1 and obs_edge:
            idx = rng.integers(len(obs_edge))
            edge = obs_edge[idx]

            door_idx = rng.integers(len(edge))
            door = edge[door_idx]

            house_grid[door[0]][door[1]] = 0
            count = cls.label_connected_regions(house_grid, background=-1)
            obs_edge.remove(edge)

        house_grid[:, [0, -1]] = -1
        house_grid[[0, -1], :] = -1

        results = cls.array_to_string(house_grid[1:-1, 1:-1])
        return results


    @classmethod
    def array_to_string(cls, array_maze):
        result = []
        for line in array_maze:
            result.append("".join(['#' if x == -1 else '.' for x in line]))

        map_str = '\n'.join([''.join(row) for row in result])
        return map_str

def generate_and_save_houses(name_prefix, seed_range):
    test_mazes = {}
    # Determine the number of digits in the largest seed to set the zero-padding correctly
    max_digits = len(str(max(seed_range)))

    for seed in seed_range:
        settings = HouseRangeSettings().sample(seed)
        maze = HouseGenerator.generate(**settings)
        # Apply dynamic zero-padding based on the number of digits in the largest seed
        map_name = f"{name_prefix}-seed-{str(seed).zfill(max_digits)}"
        test_mazes[map_name] = maze
    maps_dict_to_yaml(f'{name_prefix}.yaml', test_mazes)


def main():
    generate_and_save_houses("validation-houses", range(0, 128))
    generate_and_save_houses("training-houses", range(128, 128 + 2048))


if __name__ == "__main__":
    main()
