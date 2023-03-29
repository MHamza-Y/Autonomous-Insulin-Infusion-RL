import os

import dill


def pickle_obj(obj, save_path):
    """
    Saves an object as a pickle
    :param obj: the object to save
    :param save_path: the save path
    :return:
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        dill.dump(obj, f)


def load_obj(save_path):
    """
    loads a pickled object
    :param save_path: the path where the pickled object is saved
    :return: the loaded object
    """
    with open(save_path, 'rb') as f:
        return dill.load(f)
