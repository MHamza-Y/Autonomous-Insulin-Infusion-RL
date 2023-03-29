import warnings

import pandas as pd

from src.base_rl.evaluation import EvaluationActor
from src.rllib_utills.policies_wrapper import Policy


def get_evenly_divided_values(value_to_be_distributed, times):
    """
    Evenly divide a number into chunks
    :param value_to_be_distributed: The value that has to be divided into chunks
    :param times: Number of chunks to create
    :return: The list containing the chunk values e.g. if value_to_be_distributed is 10 and times is 2, [5,5] is returned
    """
    return [value_to_be_distributed // times + int(x < value_to_be_distributed % times) for x in range(times)]


def eval_env(ray, epochs, env_creator, env_kwargs, checkpoint_path, render=False, workers=1):
    """
    Evaluates a given environment in parallel manner and returns the merged results from all the episodes and workers
    :param ray: ray instance for context, can be initialized by calling ray.init() in the parent script
    :param epochs: number of episodes to evaluate
    :param env_creator: the function that returns an instance of the environment
    :param env_kwargs: the parameters to pass to the environment
    :param checkpoint_path: the path to the checkpoint containing the policy
    :param render: whether to render the evaluation
    :param workers: the number of parallel workers
    :return: the dataframe containing trajectories from all the evaluated episodes
    """
    if render:
        warnings.warn('render=True, numbers of workers changed to 1')
        workers = 1
    divided_epochs = get_evenly_divided_values(epochs, workers)
    tmp_env = env_creator(env_kwargs)
    policy_kwargs = dict(checkpoint_path=checkpoint_path, obs_space=tmp_env.observation_space)

    evaluators = [EvaluationActor.remote(env_creator=env_creator, env_kwargs=env_kwargs, policy_class=Policy,
                                         policy_kwargs=policy_kwargs) for _ in range(workers)]

    evaluators_futures = [evaluator.evaluate.remote(epochs=divided_epochs[i], render=False) for i, evaluator in
                          enumerate(evaluators)]

    evaluator_results = ray.get(evaluators_futures)
    evaluator_results = sum(evaluator_results, [])
    result_dfs = [result.get_dataframe(ep_id=i) for i, result in enumerate(evaluator_results)]

    results_df = pd.concat(result_dfs, axis=0)
    return results_df
