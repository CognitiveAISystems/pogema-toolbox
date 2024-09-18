# noinspection PyUnresolvedReferences
from pogema_toolbox import fix_num_threads_issue

import json
from pathlib import Path

import time

import numpy as np

from pogema_toolbox.config_variant_generator import generate_variants
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry

from pogema_toolbox.views.view_multi_plot import process_multi_plot_view, MultiPlotView
from pogema_toolbox.views.view_plot import process_plot_view, PlotView
from pogema_toolbox.views.view_tabular import process_table_view, TabularView

from concurrent.futures import ProcessPoolExecutor


def sequential_backend(algo_config, env_configs, full_algo_name, registry_state=None):
    """
    Runs the algorithm sequentially on multiple environments.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.
        registry_state:

    Returns:
        List: Results of running the algorithm on the environments.
    """
    registry = ToolboxRegistry
    if registry_state is not None:
        registry.recreate_from_state(registry_state)

    results = []
    algo_name = algo_config['name']
    algo = registry.create_algorithm(algo_name, **algo_config)
    algo_cfg = registry.create_algorithm_config(algo_name, **algo_config)
    for idx, env_config in enumerate(env_configs):
        ToolboxRegistry.info(f'Running: {full_algo_name} [{idx + 1}/{len(env_configs)}]')
        env = registry.create_env(env_config['name'], **env_config)
        if algo_cfg.preprocessing:
            ToolboxRegistry.debug('Adding preprocessing')
            env = registry.create_algorithm_preprocessing(env, algo_name, **algo_config)
        results.append(registry.run_episode(env, algo, algo_cfg.run_episode_func))

        if env_config.get('with_animation', None):
            from pathlib import Path

            directory = Path(f'renders/{full_algo_name}/')
            name = env.pick_name(env.grid_config)

            directory.mkdir(parents=True, exist_ok=True)
            ToolboxRegistry.debug(f'Saving animation to "{directory / name}"')
            env.save_animation(name=directory / name)
    return results


def split_on_chunks(size, num_chunks):
    """
    Splits the given size into equal chunks.

    Args:
        size: The size to be split.
        num_chunks: Number of chunks to split into.

    Yields:
        Tuple[int, int]: Start and end indexes of each chunk.
    """
    offset = int(1.0 * size / num_chunks + 0.5)
    for i in range(0, num_chunks - 1):
        yield i * offset, i * offset + offset
    yield num_chunks * offset - offset, size


def get_env_config_cost(raw_config):
    gc = Environment(**raw_config)
    return gc.num_agents * gc.max_episode_steps


def get_balanced_buckets_indexes(env_configs, num_buckets):
    """
    Distributes environment indexes into balanced buckets based on their costs.

    Args:
        env_configs: List of environment configurations.
        num_buckets: Number of buckets to distribute the indexes into.

    Returns:
        List[List[int]]: Balanced buckets containing environment indexes.
    """
    buckets = [[] for _ in range(num_buckets)]
    bucket_costs = [0 for _ in range(num_buckets)]
    env_costs = [get_env_config_cost(ec) for ec in env_configs]
    indexes = np.argsort(env_costs)[::-1]

    for idx in indexes:
        min_bucket_idx = np.argmin(bucket_costs)
        buckets[min_bucket_idx].append(idx)
        bucket_costs[min_bucket_idx] += env_costs[idx]

    # remove empty buckets
    buckets = [bucket for bucket in buckets if len(bucket) > 0]

    return buckets


def get_num_of_available_cpus():
    """
    Returns the number of available CPUs.

    Returns:
        int: Number of available CPUs.
    """
    import multiprocessing
    return multiprocessing.cpu_count()


def dask_backend(algo_config, env_configs, full_algo_name):
    """
    Runs the algorithm using Dask for distributed computing.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.

    Returns:
        List: Results of running the algorithm on the environments.
    """
    import dask.distributed as dd
    initialized_algo_config = ToolboxRegistry.create_algorithm_config(algo_config['name'], **algo_config)

    num_process = min(initialized_algo_config.num_process, get_num_of_available_cpus())
    cluster = dd.LocalCluster(n_workers=num_process, threads_per_worker=1, nthreads=1, )
    client = dd.Client(cluster, timeout="120s")  # Connect the client to the cluster

    ToolboxRegistry.get_maps()
    registry_state = ToolboxRegistry.get_state()

    futures = []
    for left, right in split_on_chunks(len(env_configs), initialized_algo_config.num_process):
        future = client.submit(sequential_backend, algo_config, env_configs[left:right], full_algo_name, registry_state, pure=False)
        futures.append(future)

    results = client.gather(futures)  # Gather the results from the distributed tasks
    client.close()  # Close the Dask client and cluster
    cluster.close()
    results = np.concatenate(results).tolist()
    return results


def balanced_multiprocess_backend(algo_config, env_configs, full_algo_name):
    """
    Runs the algorithm in a balanced manner using multiple processes.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.

    Returns:
        List: Results of running the algorithm on the environments.
    """
    initialized_algo_config = ToolboxRegistry.create_algorithm_config(algo_config['name'], **algo_config)

    num_process = min(initialized_algo_config.num_process, get_num_of_available_cpus())
    balanced_buckets = get_balanced_buckets_indexes(env_configs, num_process)

    # Getting maps to initialize registry (if not) and  avoid multiple loading
    ToolboxRegistry.get_maps()
    registry_state = ToolboxRegistry.get_state()

    with ProcessPoolExecutor(num_process) as executor:
        future2stuff = []
        for bucket in balanced_buckets:
            # select config from env_configs by their id from bucket
            bucket_configs = [env_configs[idx] for idx in bucket]
            future2stuff.append(
                executor.submit(sequential_backend, algo_config, bucket_configs, full_algo_name, registry_state))

        # Reorder the results according to the original order of env_configs
        ordered_results = [None for _ in range(len(env_configs))]
        for idx, future in enumerate(future2stuff):
            bucket = balanced_buckets[idx]
            bucket_results = future.result()
            for i, env_idx in enumerate(bucket):
                ordered_results[env_idx] = bucket_results[i]

    return ordered_results


def multiprocess_backend(algo_config, env_configs, full_algo_name):
    """
    Runs the algorithm in parallel using multiple processes.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.

    Returns:
        List: Results of running the algorithm on the environments.
    """
    initialized_algo_config = ToolboxRegistry.create_algorithm_config(algo_config['name'], **algo_config)

    ToolboxRegistry.get_maps()
    registry_state = ToolboxRegistry.get_state()

    results = []
    num_process = min(initialized_algo_config.num_process, get_num_of_available_cpus())
    with ProcessPoolExecutor(num_process) as executor:
        future2stuff = []
        for left, right in split_on_chunks(len(env_configs), initialized_algo_config.num_process):
            future2stuff.append(
                executor.submit(sequential_backend, algo_config, env_configs[left:right], full_algo_name, registry_state))
        for future in future2stuff:
            results += future.result()
    return results


def balanced_dask_backend(algo_config, env_configs, full_algo_name):
    """
    Runs the algorithm in a balanced manner using Dask for distributed computing.

    Args:
        algo_config: Configuration for the algorithm.
        env_configs: List of environment configurations.
        full_algo_name: Full name of the algorithm.

    Returns:
        List: Results of running the algorithm on the environments.
    """
    ToolboxRegistry.debug('Running experiment with balanced task backend')
    import dask.distributed as dd

    initialized_algo_config = ToolboxRegistry.create_algorithm_config(algo_config['name'], **algo_config)

    num_process = min(initialized_algo_config.num_process, get_num_of_available_cpus())
    balanced_buckets = get_balanced_buckets_indexes(env_configs, num_process)

    cluster = dd.LocalCluster(n_workers=num_process, threads_per_worker=1, nthreads=1)
    client = dd.Client(cluster, timeout="120s")  # Connect the client to the cluster

    futures = []

    ToolboxRegistry.get_maps()
    registry_state = ToolboxRegistry.get_state()

    for bucket in balanced_buckets:
        bucket_configs = [env_configs[idx] for idx in bucket]
        future = client.submit(sequential_backend, algo_config, bucket_configs, full_algo_name, registry_state,
                               pure=False)
        futures.append(future)

    results = client.gather(futures)
    client.close()
    cluster.close()

    # Reorder the results according to the original order of env_configs
    ordered_results = [None for _ in range(len(env_configs))]
    for idx, bucket in enumerate(balanced_buckets):
        bucket_results = results[idx]
        for i, env_idx in enumerate(bucket):
            ordered_results[env_idx] = bucket_results[i]

    return ordered_results


def join_metrics_and_configs(metrics, evaluation_configs, env_grid_search, algo_config, algo_name):
    """
    Joins metrics, evaluation configurations, environment grid search, and algorithm name into a result dictionary.

    Args:
        metrics: List of metrics.
        evaluation_configs: List of evaluation configurations.
        env_grid_search: List of environment grid search configurations.
        algo_config: Configuration for the algorithm.
        algo_name: Name of the algorithm.

    Returns:
        List[dict]: List of result dictionaries.
    """
    env_grid_search = [{key[-1]: value for key, value in x.items()} for x in env_grid_search]
    results = []
    for idx, metric in enumerate(metrics):
        results.append({'metrics': metrics[idx], 'env_grid_search': env_grid_search[idx], 'algorithm': algo_name})
    return results


def run_views(results, evaluation_config, eval_dir=None):
    """
    Runs the views specified in the evaluation configuration on the results.

    Args:
        results: List of result dictionaries.
        evaluation_config: Configuration for the evaluation.
        eval_dir: Directory to save the views (optional).

    Returns:
        List: Results of running the views.
    """
    view_results = []
    if 'results_views' not in evaluation_config:
        ToolboxRegistry.info("No result views provided in config")
        return
    for key, view in evaluation_config['results_views'].items():
        save_path = Path(eval_dir if eval_dir else '.') / f'{key}.pdf'
        # create directory if not exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if view['type'] == 'tabular':
            view_results.append(process_table_view(results, TabularView(**view)))
        elif view['type'] == 'plot':
            view_results.append(process_plot_view(results, PlotView(**view), save_path))
        elif view['type'] == 'multi-plot':
            view_results.append(process_multi_plot_view(results, MultiPlotView(**view), save_path))
    return view_results


def evaluation(evaluation_config, eval_dir=None):
    """
    Runs the evaluation based on the evaluation configuration.

    Args:
        evaluation_config: Configuration for the evaluation.
        eval_dir: Directory to save the evaluation results (optional).

    Returns:
        List: Results of the evaluation.
    """
    env_grid_search, environment_configs = zip(*generate_variants(evaluation_config['environment']))

    results = []
    for key, algo_cfg in evaluation_config['algorithms'].items():
        p_algo_cfg = ToolboxRegistry.create_algorithm_config(algo_cfg['name'], **algo_cfg)
        ToolboxRegistry.info(f'Starting: {key}, {algo_cfg}')
        start_time = time.monotonic()
        if p_algo_cfg.parallel_backend == 'sequential':
            metrics = sequential_backend(algo_cfg, environment_configs, key)
        elif p_algo_cfg.parallel_backend == 'multiprocessing':
            metrics = multiprocess_backend(algo_cfg, environment_configs, key)
        elif p_algo_cfg.parallel_backend == 'dask':
            metrics = dask_backend(algo_cfg, environment_configs, key)
        elif p_algo_cfg.parallel_backend == 'balanced_multiprocessing':
            metrics = balanced_multiprocess_backend(algo_cfg, environment_configs, key)
        elif p_algo_cfg.parallel_backend == 'balanced_dask':
            metrics = balanced_dask_backend(algo_cfg, environment_configs, key)
        else:
            raise ValueError(f'Unknown parallel backend: {p_algo_cfg.parallel_backend}')
        algo_results = join_metrics_and_configs(metrics, environment_configs, env_grid_search, algo_cfg, key)
        if eval_dir:
            save_path = Path(eval_dir) / f'{key}.json'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(algo_results, f)
        results += algo_results
        ToolboxRegistry.success(f'Finished: {key}, runtime: {time.monotonic() - start_time}')

    run_views(results, evaluation_config, eval_dir=eval_dir)
    return results
