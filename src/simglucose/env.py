import random

import numpy as np
import pandas as pd
import pkg_resources
from gym import spaces
from ray.tune.registry import register_env
from simglucose.controller.base import Action
from simglucose.envs import T1DSimEnv

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')
patient_params = pd.read_csv(PATIENT_PARA_FILE)
patient_names = patient_params['Name'].values

OBS_MIN = np.array([0, 0])
OBS_MAX = np.array([420, 10])


def calculate_base_observation(bg_value, insulin_value, meal):
    meal_indicator = int(meal > 0.0)
    #obs = (meal_indicator, np.array([bg_value[0], insulin_value], dtype=np.float32))
    obs = np.array([bg_value[0], insulin_value], dtype=np.float32)
    return obs


class SimglucoseEnv(T1DSimEnv):
    def __init__(self, patient_name=None, patient_type=None, **kwargs):
        if patient_name is None:
            if patient_type:
                filtered_patients = [p for p in patient_names if patient_type in p]
                patient_name = random.choice(filtered_patients)
            else:
                patient_name = random.choice(patient_names)

        self.i = 0
        super(SimglucoseEnv, self).__init__(patient_name, **kwargs)

    def _step(self, action):

        self.i += 1
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            bg_value, reward, done, info = self.env.step(act)
        else:
            bg_value, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)

        print(self.i)
        print(bg_value)
        print(action)
        meal = info['meal']
        print(meal)
        obs_dict = calculate_base_observation(bg_value, action, meal)
        return obs_dict, reward, done, info

    def _reset(self):
        self.env, _, _, _ = self._create_env_from_random_state(self.custom_scenario)
        obs, _, _, _ = self.env.reset()
        return calculate_base_observation(obs, 0, 0)

    def step(self, action):
        print(action)
        print(self.i)
        return self._step(action)

    def reset(self, **kwargs):
        return self._reset()

    def seed(self, seed=None):
        return self._seed(seed)

    def render(self, mode="human", **kwargs):
        self._render(mode)

    @property
    def action_space(self):
        ub = 10
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):

        # return spaces.Tuple(
        #     (
        #         spaces.Discrete(2),
        #         spaces.Box(low=OBS_MIN, high=OBS_MAX, shape=(OBS_MAX.size,), dtype=np.float32)
        #
        #     )
        # )
        return spaces.Box(low=OBS_MIN, high=OBS_MAX, shape=(OBS_MAX.size,), dtype=np.float32)


class SimglucoseDiscEnv(SimglucoseEnv):
    def __init__(self, **kwargs):
        self.ACTIONS = [0, 0.01, 0.02, 0.04, 0.08, 0.1,
                        0.2]  # [0, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4]  # [0, 0.01, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 10]
        super(SimglucoseDiscEnv, self).__init__(**kwargs)

    def step(self, action):
        basal_val = self.ACTIONS[action]
        observation, reward, done, info = self._step(basal_val)
        return observation, reward, done, info

    @property
    def action_space(self):
        num_actions = len(self.ACTIONS)
        return spaces.Discrete(num_actions)


def env_creator(env_config):
    """
    Function that returns an instantiated instance of the simglucose environment
    :param env_config: The parameter used by RlLib to pass extra initialization parameters to the environment
    :return: the platform environment object
    """
    return SimglucoseEnv(**env_config)


def register_simglucose_env(env_name):
    """
    Calling this function registers the simglucose environment for RlLibs usage
    :param env_name: this name is used by the RlLib to find the registered environment
    """
    register_env(env_name, env_creator)
