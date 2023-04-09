import numpy as np
from simglucose.analysis.risk import risk_index

REF_BG_POINT = 130
LOW_BG_LIMIT = 70
HIGH_BG_LIMIT = 350
HIGH_BG_SAFE_LIMIT = 180
LOW_BG_SAFE_LIMIT = 90
MAX_HIGH_SAFE_INTERVAL = HIGH_BG_SAFE_LIMIT - REF_BG_POINT
MAX_LOW_SAFE_INTERVAL = REF_BG_POINT - LOW_BG_SAFE_LIMIT
MAX_HIGH_TO_REF_DIFF = HIGH_BG_LIMIT - REF_BG_POINT
MAX_LOW_TO_REF_DIFF = REF_BG_POINT - LOW_BG_LIMIT
MAX_LOW_SAFE_TO_LOW = LOW_BG_SAFE_LIMIT - 40


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 260:
        return -10
    elif BG_last_hour[-1] > 180:
        return -2
    elif BG_last_hour[-1] > 140:
        return -1
    elif BG_last_hour[-1] < 100:
        return -1
    elif BG_last_hour[-1] < 90:
        return -2
    elif BG_last_hour[-1] < 70:
        return -10
    else:
        return 2


def shaped_reward_around_normal_bg(BG_last_hour):
    current_bg = BG_last_hour[-1]

    if current_bg > REF_BG_POINT:
        diff = current_bg - REF_BG_POINT
        reward = 1 - ((diff / MAX_HIGH_TO_REF_DIFF) ** 0.06)
    else:
        diff = REF_BG_POINT - current_bg
        reward = 1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.2)

    if current_bg <= LOW_BG_LIMIT:
        reward += -10
    elif current_bg >= HIGH_BG_LIMIT:
        reward += -20

    return reward


def shaped_negative_reward_around_normal_bg(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if REF_BG_POINT < current_bg <= HIGH_BG_SAFE_LIMIT:
        diff = current_bg - REF_BG_POINT
        reward = 100 * (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.2))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = 100 * (1 - ((diff / MAX_LOW_SAFE_INTERVAL) ** 0.2))
    elif HIGH_BG_LIMIT > current_bg > HIGH_BG_SAFE_LIMIT:
        diff = HIGH_BG_LIMIT - current_bg
        reward = -(1 - (diff / (HIGH_BG_LIMIT - HIGH_BG_SAFE_LIMIT)))
    elif LOW_BG_LIMIT < current_bg <= LOW_BG_SAFE_LIMIT:
        diff = current_bg - LOW_BG_LIMIT
        reward = -(1 - (diff / (LOW_BG_SAFE_LIMIT - LOW_BG_LIMIT)))
    else:
        reward = -1  # change to -10000

    return reward


def no_negativity(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if REF_BG_POINT < current_bg <= HIGH_BG_SAFE_LIMIT:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.2))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / MAX_LOW_SAFE_INTERVAL) ** 0.2))
    else:
        reward = 3e-3

    return reward


def no_negativityV2(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if REF_BG_POINT < current_bg:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_TO_REF_DIFF) ** 0.06))
    elif REF_BG_POINT >= current_bg:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.1))
    else:
        reward = 3e-4

    return reward


def partial_negativity(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if HIGH_BG_SAFE_LIMIT <= current_bg:
        diff = HIGH_BG_LIMIT - current_bg
        reward = -5 * (1 - (diff / (HIGH_BG_LIMIT - HIGH_BG_SAFE_LIMIT)))
    elif REF_BG_POINT < current_bg:
        diff = current_bg - REF_BG_POINT
        reward = (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.3))
    elif REF_BG_POINT >= current_bg:
        diff = REF_BG_POINT - current_bg
        reward = (1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.1))
    else:
        reward = -3e-4

    return reward


def partial_negativityV2(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if HIGH_BG_SAFE_LIMIT <= current_bg:
        diff = HIGH_BG_LIMIT - current_bg
        reward = -1.0 * (1.0 - (diff / (HIGH_BG_LIMIT - HIGH_BG_SAFE_LIMIT)))
    elif REF_BG_POINT < current_bg:
        diff = current_bg - REF_BG_POINT
        reward = (1.0 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.3))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = (1.0 - ((diff / (REF_BG_POINT - LOW_BG_SAFE_LIMIT)) ** 0.1))
    elif LOW_BG_SAFE_LIMIT >= current_bg:
        diff = current_bg - LOW_BG_LIMIT
        reward = -(1.0 - (diff / MAX_LOW_SAFE_TO_LOW))
    else:
        reward = -3e-4

    return reward


def risk_diff(BG_last_hour):
    hist_len = 20
    actual_len = len(BG_last_hour)
    if actual_len < 2:
        return 0
    elif actual_len >= 2 < hist_len * 2:
        hist_len = actual_len // 2

    _, _, risk_current = risk_index(BG_last_hour[-1:-hist_len], hist_len)
    _, _, risk_prev = risk_index(BG_last_hour[-hist_len: - (hist_len * 2)], hist_len)
    return risk_prev - risk_current


def smooth_reward(BG_last_hour):
    slope = .5
    reward = 1 - np.tanh(np.abs((BG_last_hour[-1] - REF_BG_POINT) / slope) * .1) ** 2
    if (BG_last_hour[-1] < LOW_BG_LIMIT) or (BG_last_hour[-1] > HIGH_BG_SAFE_LIMIT):
        reward = -1.
    return reward


# def tan_reward(BG_last_hour):
#     current_bg = BG_last_hour[-1]
#     if current_bg < LOW_BG_SAFE_LIMIT:
#         z = (current_bg - LOW_BG_SAFE_LIMIT) / (REF_BG_POINT - LOW_BG_SAFE_LIMIT) * 2
#         reward = np.tanh(z)
#     elif current_bg <= REF_BG_POINT:
#         diff = REF_BG_POINT - current_bg
#         reward = (1 - ((diff / MAX_LOW_SAFE_INTERVAL) ** 0.1))
#     elif REF_BG_POINT < current_bg < HIGH_BG_SAFE_LIMIT:
#         diff = current_bg - REF_BG_POINT
#         reward = (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.1))
#     else:
#         z = (HIGH_BG_SAFE_LIMIT - current_bg) / (HIGH_BG_SAFE_LIMIT - REF_BG_POINT) * 2
#         reward = np.tanh(z)
#
#     return reward

def tan_reward(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if current_bg < REF_BG_POINT:
        z = (current_bg - REF_BG_POINT) / (REF_BG_POINT - LOW_BG_SAFE_LIMIT) * 2
        reward = np.tanh(z)
    else:
        z = (REF_BG_POINT - current_bg) / (HIGH_BG_SAFE_LIMIT - REF_BG_POINT) * 2
        reward = np.tanh(z)

    reward += 0.7
    reward = reward if reward >= 0 else 0.1 * reward
    return reward


def uniform_reward_regions(BG_last_hour):
    current_bg = BG_last_hour[-1]
    if LOW_BG_SAFE_LIMIT < current_bg < HIGH_BG_SAFE_LIMIT:
        reward = 0.1
    else:
        reward = -0.1

    return reward


def uniform_reward_with_risk(BG_last_hour):
    current_bg = BG_last_hour[-1]
    _, _, risk_idx = risk_index(BG_last_hour, 50)
    risk_idx_norm = (risk_idx / 30)

    if LOW_BG_SAFE_LIMIT < current_bg < HIGH_BG_SAFE_LIMIT:
        reward = 0.1
        reward *= (1 - risk_idx_norm)
    else:
        reward = -0.1
        reward *= risk_idx_norm

    return reward
