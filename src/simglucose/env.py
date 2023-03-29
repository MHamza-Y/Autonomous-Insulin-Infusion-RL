import numpy as np
import pandas as pd
from ray.tune.registry import register_env
import random
import pkg_resources
from simglucose.envs import T1DSimEnv

# from simglucose.envs.simglucose_gym_env import T1DSimEnv

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

patient_params = pd.read_csv(PATIENT_PARA_FILE)
patient_names = patient_params['Name'].values

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class SimglucoseEnv(T1DSimEnv):
    def __init__(self, patient_name=None, patient_type=None, **kwargs):
        if patient_name is None:
            if patient_type:
                filtered_patients = [p for p in patient_names if patient_type in p]
                patient_name = random.choice(filtered_patients)
            else:
                patient_name = random.choice(patient_names)

        super(SimglucoseEnv, self).__init__(patient_name, **kwargs)

    def step(self, action):

        basal_val = 0
        if action[0]:
            basal_val = action[1][0]
        observation, reward, done, info = self._step(basal_val)
        return np.asarray(observation), reward, done, info

    def reset(self, **kwargs):
        return np.asarray(self._reset())

    def seed(self, seed=None):
        return self._seed(seed)

    def render(self, mode="human", **kwargs):
        self._render(mode)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        num_actions = 2
        return spaces.Tuple((spaces.Discrete(num_actions), spaces.Box(low=0, high=ub, shape=(1,))))


class SimglucoseDiscEnv(T1DSimEnv):
    def __init__(self, patient_name=None, patient_type=None, **kwargs):
        if patient_name is None:
            if patient_type:
                filtered_patients = [p for p in patient_names if patient_type in p]
                patient_name = random.choice(filtered_patients)
            else:
                patient_name = random.choice(patient_names)
        self.ACTIONS = [0, 0.01, 0.1, 1, 10]
        self.i = 0
        super(SimglucoseDiscEnv, self).__init__(patient_name, **kwargs)

    def step(self, action):
        basal_val = self.ACTIONS[action]
        observation, reward, done, info = self._step(basal_val)
        print(self.i)
        self.i += 1
        print(observation)
        return np.asarray(observation), reward, done, info

    def reset(self, **kwargs):
        return np.asarray(self._reset())

    def seed(self, seed=None):
        return self._seed(seed)

    def render(self, mode="human", **kwargs):
        self._render(mode)

    @property
    def action_space(self):
        num_actions = len(self.ACTIONS)
        return spaces.Discrete(num_actions)


class SimglucoseNPEnv(T1DSimEnv):
    def __init__(self, patient_name=None, patient_type=None, **kwargs):
        if patient_name is None:
            if patient_type:
                filtered_patients = [p for p in patient_names if patient_type in p]
                patient_name = random.choice(filtered_patients)
            else:
                patient_name = random.choice(patient_names)

        super(SimglucoseNPEnv, self).__init__(patient_name, **kwargs)

    def step(self, action):
        observation, reward, done, info = self._step(action)
        return np.asarray(observation), reward, done, info

    def reset(self, **kwargs):
        return np.asarray(self._reset())

    def seed(self, seed=None):
        return self._seed(seed)

    def render(self, mode="human", **kwargs):
        self._render(mode)


def env_creator(env_config):
    """
    Function that returns an instantiated instance of the simglucose environment
    :param env_config: The parameter used by RlLib to pass extra initialization parameters to the environment
    :return: the platform environment object
    """
    return SimglucoseDiscEnv(**env_config)


def register_simglucose_env(env_name):
    """
    Calling this function registers the simglucose environment for RlLibs usage
    :param env_name: this name is used by the RlLib to find the registered environment
    """
    register_env(env_name, env_creator)
