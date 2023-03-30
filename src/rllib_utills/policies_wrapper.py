from os.path import join

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.policy import Policy as RLLibPolicy
from ray.rllib.utils.spaces import space_utils

from src.rllib_utills.configs import get_checkpoint_configs


class Policy:
    """ Easy to use RlLib policy wrapper
    Also this class allow the user to compute actions on the saved policy without restoring the whole experiment
    """

    def __init__(self, checkpoint_path, obs_space, action_space):
        """
        loads the policy and the pre-processor for the observations
        :param checkpoint_path:
        :param obs_space:
        """
        configs = get_checkpoint_configs(checkpoint_path)
        policy_path = join(checkpoint_path, 'policies', 'default_policy')
        self.policy = RLLibPolicy.from_checkpoint(policy_path)
        self.use_lstm = configs["model"]["use_lstm"]
        self.lstm_use_prev_actions = configs["model"]["lstm_use_prev_action"]
        if self.use_lstm:
            self.state = self.policy.get_initial_state()

        self.preprocessor = get_preprocessor(obs_space)(obs_space)
        self.action_preprocessor = get_preprocessor(action_space)(action_space)
        self.prev_action = action_space.sample()

    def __call__(self, obs, explore=False):
        """
        Computes the action for a given observed state
        :param obs: the observed state
        :param explore: if turned on an action is sampled from the policy instead of returning the best action
        :return: the action returned by the policy
        """
        obs = self.preprocessor.transform(obs)

        if self.use_lstm:
            if self.lstm_use_prev_actions:
                #self.prev_action = self.action_preprocessor.transform(self.prev_action)
                action, self.state, _ = self.policy.compute_single_action(obs=obs, state=self.state, explore=explore,
                                                                          prev_action=self.prev_action)
            else:
                action, self.state, _ = self.policy.compute_single_action(obs=obs, state=self.state, explore=explore)

        else:
            action, _, _ = self.policy.compute_single_action(obs=obs, explore=explore)

        action = space_utils.unsquash_action(action, self.policy.action_space_struct)
        self.prev_action = action
        return action
