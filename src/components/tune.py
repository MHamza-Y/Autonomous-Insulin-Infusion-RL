from ray import air, tune


def tune_hyper_param(algo, config, log_dir, iterations,
                     name,max_concurrent_trials=1):
    """
    Wrapper for the ray.tune.Tuner.fit function used for tuning the hyper-parameters
    :param algo: The algorithm to use for training
    :param config: The algorithm configs
    :param log_dir: Where to save logs and the checkpoints
    :param iterations: Number of iterations to tune for
    :param name: name of the training to group the logs
    :param max_concurrent_trials: maximum number of trial tuned in parallel
    :return: the results of the hyper-parameter tuning
    """
    tuner = tune.Tuner(
        algo,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            max_concurrent_trials=max_concurrent_trials
        ),
        run_config=air.RunConfig(
            name=name,
            local_dir=log_dir,

            stop={"training_iteration": iterations},
        ),

    )

    results = tuner.fit()
    return results
