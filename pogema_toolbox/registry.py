from pathlib import Path
import yaml
import sys
from loguru import logger as base_logger

from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.run_episode import run_episode


class ToolboxRegistry:
    _instances = {}
    _maps = None
    _algorithms = {}
    _envs = {}
    _logger = None
    _logger_config = {'level': 'INFO', 'sink': sys.stderr}
    _run_episode_funcs = {'default': run_episode}

    # ----- Logging section -----
    @classmethod
    def setup_logger(cls, level="INFO", sink=None):
        sink = sink or sys.stderr
        format_string = "<level>{time:YYYY-MM-DD at HH:mm:ss} | Toolbox {level} | {message}</level>"
        cls._logger = base_logger.bind(name="ToolboxRegistry")
        cls._logger.remove()
        cls._logger.add(sink, level=level, format=format_string, colorize=True)
        cls._logger_config = {'level': level, 'sink': sink}

    @classmethod
    def _ensure_logger(cls):
        if cls._logger is None:
            cls.setup_logger()

    @classmethod
    def info(cls, message):
        cls._ensure_logger()
        cls._logger.info(message)

    @classmethod
    def debug(cls, message):
        cls._ensure_logger()
        cls._logger.debug(message)

    @classmethod
    def warning(cls, message):
        cls._ensure_logger()
        cls._logger.warning(message)

    @classmethod
    def error(cls, message):
        cls._ensure_logger()
        cls._logger.error(message)

    @classmethod
    def success(cls, message):
        cls._ensure_logger()
        cls._logger.success(message)

    # ----- State section -----
    @classmethod
    def get_state(cls):
        cls._ensure_logger()
        return {
            'maps': cls._maps,
            'algorithms': cls._algorithms,
            'envs': cls._envs,
            'logger_config': cls._logger_config,
            'run_episode': cls._run_episode_funcs,
        }

    @classmethod
    def recreate_from_state(cls, state):
        cls._maps = state.get('maps', {})
        cls._algorithms = state.get('algorithms', {})
        cls._envs = state.get('envs', {})
        cls._run_episode_funcs = state.get("run_episode", {})
        logger_config = state.get('logger_config', {})
        if logger_config:
            cls.setup_logger(level=logger_config.get('level', 'INFO'), sink=logger_config.get('sink', sys.stderr))
        cls.debug('Registry is recreated from state')

    # ----- Algorithms section -----
    @classmethod
    def register_algorithm(cls, name, make_func, config_make_func=None, preprocessing_make_func=None):
        if name in cls._algorithms:
            cls._logger.warning(f'Registering existing algorithm with name {name}')
        cls._algorithms[name] = make_func, config_make_func, preprocessing_make_func
        cls.debug(f'Registered algorithm with name {name}')

    @classmethod
    def create_algorithm(cls, algo_name, **kwargs):
        algo_make_func, config_make_func, _ = cls._algorithms[algo_name]
        cls.debug(f'Creating {algo_name} algorithm')
        if config_make_func:
            config = cls.create_algorithm_config(algo_name, **kwargs)
            return algo_make_func(config)
        return algo_make_func()

    @classmethod
    def create_algorithm_config(cls, algo_name, **kwargs):
        _, config_make_func, _ = cls._algorithms[algo_name]
        if config_make_func is None:
            cls.debug('Using default config - AlgoBase')
            return AlgoBase(**kwargs)
        cls.debug(f'Creating {algo_name} config')
        return config_make_func(**kwargs)

    @classmethod
    def create_algorithm_preprocessing(cls, env_to_wrap, algo_name, **kwargs):
        _, config_make_func, preprocessing_make_func = cls._algorithms[algo_name]
        if config_make_func is not None:
            algo_cfg = cls.create_algorithm_config(algo_name, **kwargs)
            cls.debug(f'Creating {algo_name} preprocessing')
            return preprocessing_make_func(env_to_wrap, algo_cfg)
        return preprocessing_make_func(env_to_wrap)

    # ----- Environments section -----
    @classmethod
    def register_env(cls, name, make_func, config_make_func=None):
        if name in cls._envs:
            cls._logger.warning(f'Registering existing environment with name {name}')
        cls._envs[name] = make_func, config_make_func
        cls.debug(f'Registered environment with name {name}')

    @classmethod
    def create_env(cls, env_name, **kwargs):
        env_make_func, config_make_func = cls._envs[env_name]

        if config_make_func:
            cls.debug(f'Creating {env_name} env using config')
            config = config_make_func(**kwargs)
            return env_make_func(config)
        cls.debug(f'Creating {env_name} env')
        return env_make_func()

    # ----- Maps section -----
    @classmethod
    def get_maps(cls):
        if cls._maps is None:
            cls._initialize_maps()
        return cls._maps

    @classmethod
    def register_maps(cls, maps):
        for map_name, map_grid in maps.items():
            if map_name in cls.get_maps():
                cls._logger.warning(f'Registering existing map with name {map_name}')
            cls._maps[map_name] = map_grid
            cls.debug(f'Registered map with name {map_name}')

    @classmethod
    def _initialize_maps(cls):
        maps_folder_path = Path(__file__).parent / "maps"
        cls._maps = {}

        for yaml_file_path in maps_folder_path.glob("*.yaml"):
            with open(yaml_file_path, "r") as f:
                try:
                    grids_content = yaml.safe_load(f)
                    if grids_content:  # Check if the YAML file is not empty
                        cls._maps.update(grids_content)
                except yaml.YAMLError as exc:
                    cls.error(f'Error loading YAML file {yaml_file_path}: {exc}')
        cls.debug(f'Registered {len(cls._maps)} maps')
    
    # ----- Run episode section -----
    @classmethod
    def register_run_func(cls, name, run_func):
        if name in cls._run_episode_funcs:
            cls._logger.warning(f'Registering existing run_episode function with name {name}')
        cls._run_episode_funcs[name] = run_func
        cls.debug(f'Registered run_episode function with name {name}')
    
    @classmethod
    def run_episode(cls, env, algo, run_func_name='default'):
        run_episode_func = cls._run_episode_funcs[run_func_name]
        return run_episode_func(env, algo)