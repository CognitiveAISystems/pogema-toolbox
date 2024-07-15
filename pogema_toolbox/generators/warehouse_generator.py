from dataclasses import dataclass

import numpy as np

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass
class WarehouseConfig:
    wall_width: int = 5
    wall_height: int = 2
    walls_in_row: int = 5
    walls_rows: int = 5
    bottom_gap: int = 5
    horizontal_gap: int = 1
    vertical_gap: int = 3
    # wfi_instances: bool = True


def generate_warehouse(cfg: WarehouseConfig):
    height = cfg.vertical_gap * (cfg.walls_rows + 1) + cfg.wall_height * cfg.walls_rows
    width = cfg.bottom_gap * 2 + cfg.wall_width * cfg.walls_in_row + cfg.horizontal_gap * (cfg.walls_in_row - 1)

    grid = np.zeros((height, width), dtype=int)

    for row in range(cfg.walls_rows):
        row_start = cfg.vertical_gap * (row + 1) + cfg.wall_height * row
        for col in range(cfg.walls_in_row):
            col_start = cfg.bottom_gap + col * (cfg.wall_width + cfg.horizontal_gap)
            grid[row_start:row_start + cfg.wall_height, col_start:col_start + cfg.wall_width] = 1

    return '\n'.join(''.join('!' if cell == 0 else '#' for cell in row) for row in grid)


def generate_wfi_positions(grid_str, bottom_gap, vertical_gap):
    if vertical_gap == 1:
        raise ValueError("Cannot generate WFI instance with vertical_gap of 1.")

    grid = [list(row) for row in grid_str.strip().split('\n')]
    height = len(grid)
    width = len(grid[0])

    start_locations = []
    goal_locations = []

    for row in range(1, height - 1):
        if row % 3 == 0:
            continue
        for col in range(bottom_gap - 1):
            if grid[row][col] == '!':
                start_locations.append((row, col))
        for col in range(width - bottom_gap + 1, width):
            if grid[row][col] == '!':
                start_locations.append((row, col))

    if vertical_gap == 2:
        for row in range(1, height):
            for col in range(width):
                if grid[row][col] == '!' and grid[row - 1][col] == '#':
                    goal_locations.append((row, col))
    else:
        for row in range(height):
            for col in range(width):
                if grid[row][col] == '!':
                    if (row > 0 and grid[row - 1][col] == '#') or (row < height - 1 and grid[row + 1][col] == '#'):
                        goal_locations.append((row, col))

    return start_locations, goal_locations


def generate_wfi_warehouse(cfg: WarehouseConfig = WarehouseConfig()):
    grid = generate_warehouse(cfg)
    start_locations, goal_locations = generate_wfi_positions(grid, cfg.bottom_gap, cfg.vertical_gap)
    grid_list = [list(row) for row in grid.split('\n')]

    for s in start_locations:
        grid_list[s[0]][s[1]] = '@'
    for s in goal_locations:
        if grid_list[s[0]][s[1]] == '$':
            grid_list[s[0]][s[1]] = '!'
        else:
            grid_list[s[0]][s[1]] = '$'
    str_grid = '\n'.join([''.join(row) for row in grid_list])

    return str_grid


def main():
    maps_dict_to_yaml(f'warehouse.yaml', {'warehouse': generate_wfi_warehouse(WarehouseConfig(wall_width=10))})


if __name__ == '__main__':
    main()
