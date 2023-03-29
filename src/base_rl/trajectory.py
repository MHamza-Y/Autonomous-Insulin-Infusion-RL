import pandas as pd
import numpy as np


class Trajectory:
    """
    The class to neatly save a single trajectory while providing interface to add samples and get in dataframe format
    """
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.rewards = []
        self.actions = []

    def add_sample(self, obs, next_obs, reward, action):
        """
        This method is called at each time step to store the sample
         tuple containing (observation, next observation, reward, action)
        :param obs: The observed state of the environment
        :param next_obs: The observed state of the environment on performing action
        :param reward: The reward received on performing action
        :param action: The action performed on the environment
        """
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_dataframe(self, ep_id=None):
        """
        Returns the whole trajectory in the dataframe format
        :param ep_id: a column named ep_id with its value is added to the dataframe if this parameter is specified
        :return: the dataframe containing the trajectory
        """
        time_steps = range(len(self.observations))
        df = pd.DataFrame.from_dict(self.__dict__)
        df["ep_id"] = ep_id
        df["time_step"] = time_steps
        return df
