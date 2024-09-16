from pogema_toolbox.results_holder import ResultsHolder

def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()

    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)

        if all(terminated) or all(truncated):
            break
    return results_holder.get_final()
