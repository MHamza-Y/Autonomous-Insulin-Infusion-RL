import gym
from ray.rllib.models.preprocessors import _legacy_patch_shapes, OneHotPreprocessor, ATARI_OBS_SHAPE, logger, \
    GenericPixelPreprocessor, ATARI_RAM_OBS_SHAPE, AtariRamPreprocessor, TupleFlatteningPreprocessor, \
    DictFlatteningPreprocessor, RepeatedValuesPreprocessor, NoPreprocessor
from ray.rllib.utils.spaces.repeated import Repeated


def _get_preprocessor(space: gym.Space) -> type:
    """ This is modification of the ray.rllib.models.preprocessors.get_preprocessor
    Returns NoPreprocessor instead of OneHotPreprocessor for categorical object
    """

    _legacy_patch_shapes(space)
    obs_shape = space.shape

    if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        preprocessor = NoPreprocessor
    elif obs_shape == ATARI_OBS_SHAPE:
        logger.debug(
            "Defaulting to RLlib's GenericPixelPreprocessor because input "
            "space has the atari-typical shape {}. Turn this behaviour off by setting "
            "`preprocessor_pref=None` or "
            "`preprocessor_pref='deepmind'` or disabling the preprocessing API "
            "altogether with `_disable_preprocessor_api=True`.".format(ATARI_OBS_SHAPE)
        )
        preprocessor = GenericPixelPreprocessor
    elif obs_shape == ATARI_RAM_OBS_SHAPE:
        logger.debug(
            "Defaulting to RLlib's AtariRamPreprocessor because input "
            "space has the atari-typical shape {}. Turn this behaviour off by setting "
            "`preprocessor_pref=None` or "
            "`preprocessor_pref='deepmind' or disabling the preprocessing API "
            "altogether with `_disable_preprocessor_api=True`."
            "`.".format(ATARI_OBS_SHAPE)
        )
        preprocessor = AtariRamPreprocessor
    elif isinstance(space, gym.spaces.Tuple):
        preprocessor = TupleFlatteningPreprocessor
    elif isinstance(space, gym.spaces.Dict):
        preprocessor = DictFlatteningPreprocessor
    elif isinstance(space, Repeated):
        preprocessor = RepeatedValuesPreprocessor
    else:
        preprocessor = NoPreprocessor

    return preprocessor