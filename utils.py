import hashlib
from typing import List

import numpy as np


def hash_to_int(s: str, seed: str):
    return int(hashlib.md5((s + seed).encode("utf-8")).hexdigest(), 16)


def split_keys(keys: List[str], split_ratio: List[int], seed=""):
    """
    Stable method to split keys. The location of a given key is deterministic
    wrt splits and seed (wow).

    :param keys:
    List of string keys to split
    :param split_ratio:
    E.g., [70, 30] for 70:30 split
    :param seed:
    :return:
    """
    thresholds = np.cumsum(split_ratio)
    lists = list()
    for _ in split_ratio:
        lists.append(list())
    for key in keys:
        v = hash_to_int(key, seed) % thresholds[-1]
        for i, t in enumerate(thresholds):
            if v < t:
                lists[i].append(key)
                break
    return lists
