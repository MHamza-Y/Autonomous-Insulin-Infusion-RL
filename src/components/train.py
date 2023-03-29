from ray import air, tune


def train_env(algo, config, log_dir, iterations, stop_reward_mean,
              name, checkpoint_frequency):
    """
    Wrapper for the ray.tune.Tuner.fit function used for training an agent on the given environment
    :param checkpoint_frequency:
    :param algo: The algorithm to use for training
    :param config: The algorithm configs
    :param log_dir: Where to save logs and the checkpoints
    :param iterations: Number of iterations to train for
    :param stop_reward_mean: If this mean reward is achieved before all the iterations are done, the training is stopped
    :param name: name of the training to group the logs
    :return: the results object returned by the ray.tune.Tuner.fit
    """
    tuner = tune.Tuner(
        algo,
        param_space=config,
        run_config=air.RunConfig(
            name=name,
            local_dir=log_dir,
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
            ),
            stop={"training_iteration": iterations, "episode_reward_mean": stop_reward_mean},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=checkpoint_frequency
            )
        ),

    )
    results = tuner.fit()

    return results
