import re
from copy import deepcopy

import numpy as np
from pogema import pogema_v0, GridConfig, AnimationMonitor, AnimationConfig
from gymnasium import Wrapper
from pydantic import BaseModel

from pogema_toolbox.registry import ToolboxRegistry


class Environment(GridConfig, ):
    with_animation: bool = False
    use_maps: bool = True


class ProvideGlobalObstacles(Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()

    def get_global_targets_xy(self):
        return self.grid.get_targets_xy()


class MultiMapWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            maps = ToolboxRegistry.get_maps()
            for map_name in sorted(maps):
                if re.match(f'^{pattern}$', map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = maps[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def reset(self, seed=None, **kwargs):
        if seed is None:
            seed = self.grid_config.seed
        self._rnd = np.random.default_rng(seed)
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs))
            cfg = deepcopy(self._configs[map_idx])
            self.env.unwrapped.grid_config = cfg
            self.env.unwrapped.grid_config.seed = seed
        return self.env.reset(seed=seed, **kwargs)


def create_env_base(config: Environment):
    env = pogema_v0(grid_config=config)
    env = ProvideGlobalObstacles(env)
    if config.use_maps:
        env = MultiMapWrapper(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(directory='experiments/renders', save_every_idx_episode=None))

    return env
