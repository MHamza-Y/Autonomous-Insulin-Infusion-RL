import ray

from src.base_rl.trajectory import Trajectory


@ray.remote
class EvaluationActor:
    """
    Wrapper class to create ray actor from the Evaluator class for parallel evaluations
    """

    def __init__(self, **kwargs):
        self.evaluator = Evaluator(**kwargs)

    def evaluate(self, **kwargs):
        return self.evaluator.evaluate(**kwargs)


class Evaluator:
    """
    Class to evaluate the environment using given policy and get the trajectories
    """

    def __init__(self, env_creator, env_kwargs, policy_class, policy_kwargs):
        """

        :param env_creator: function to which returns the instance of the environment class
        :param env_kwargs: parameters passed to the env_creator
        :param policy_class: this policy class implements __call__ method which returns best action for the given state
        :param policy_kwargs: the parameters used to instantiate the policy_class
        """
        self.env_creator = env_creator
        self.env_kwargs = env_kwargs
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

    def evaluate(self, epochs, render=False, sample_action=False):
        """
        This method performs the evaluations on the given environment using the provided policy
        :param sample_action: whether to sample action or chose the mean value
        :param epochs: number of episodes to evaluate
        :param render: whether to render the environment
        :return: list of trajectories for each episode, where each trajectory contains Trajectory type objects
        for each time step (see :class:`Trajectory`)
        """
        env = self.env_creator(self.env_kwargs)
        policy = self.policy_class(**self.policy_kwargs)
        trajectories = []
        for i in range(epochs):
            trajectory = Trajectory()
            obs = env.reset()
            done = False
            while not done:
                action = policy(obs, explore=sample_action)
                next_obs, reward, done, info = env.step(action)
                if render:
                    env.render()
                trajectory.add_sample(obs=obs, next_obs=next_obs, reward=reward, action=action)
                obs = next_obs
            trajectories.append(trajectory)

        return trajectories
