import os
import pickle


def get_checkpoint_configs(checkpoint):
    """
    gets the training configs from the saved RlLib training checkpoint
    :param checkpoint: the path to the saved training checkpoint
    :return: the configs object
    """
    run_base_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(run_base_dir, 'params.pkl')
    with open(config_path, 'rb') as f:
        configs = pickle.load(f)
    return configs
