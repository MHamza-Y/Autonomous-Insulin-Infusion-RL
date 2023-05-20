from ray.rllib.policy.policy import Policy
from ray.rllib.utils.policy import local_policy_inference


class RLLibPolicy:
    """ Easy to use RlLib policy wrapper
    Also this class allow the user to compute actions on the saved policy without restoring the whole experiment
    """

    def __init__(self, checkpoint_path):
        """
        loads the policy and the pre-processor for the observations
        :param checkpoint_path:
        :param obs_space:
        """

        policies = Policy.from_checkpoint(
            checkpoint=checkpoint_path,
            policy_ids=['default_policy'],
        )
        self.policy = policies['default_policy']

    def __call__(self, obs, explore=False):
        """
        Computes the action for a given observed state
        :param obs: the observed state
        :param explore: if turned on an action is sampled from the policy instead of returning the best action
        :return: the action returned by the policy
        """
        policy_outputs = local_policy_inference(self.policy, "env_1", "agent_1", obs, info={"explore": explore})
        action, _, _ = policy_outputs[0]
        return action
