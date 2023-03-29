import numpy as np
from simglucose.controller.base import Controller, Action

from src.rllib_utills.policies_wrapper import Policy


class SimGlucoseController(Controller):

    def __init__(self, checkpoint_path, obs_space, sample_action=False):
        self.trained_policy = Policy(checkpoint_path=checkpoint_path, obs_space=obs_space)
        self.init_state = 0
        self.sample_action = sample_action
        super().__init__(self.init_state)

    def policy(self, observation, reward, done, **info):
        obs = np.array([observation[0]])
        action = self.trained_policy(obs, explore=self.sample_action)
        if np.array(action).size > 1:
            basal = action[1][0] if action[0] else 0
        else:
            basal = action
        return Action(basal=basal, bolus=0)

    def reset(self):
        pass
