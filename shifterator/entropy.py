"""
entropy.py
"""
import numpy as np
from math import log

def get_relative_freqs(type2freq):
    """
    Calculates the relative frequency (proportion) of each type in a system

    Parameters
    ----------
    type2freq: dict
        keys are types of a system and values are frequencies of those types
    """
    n = sum(type2freq.values())
    type2p = {t:s/n for t,s in type2freq.items()}
    return type2p

def get_entropy_scores(type2p_1, type2p_2, base=2, alpha=1):
    type2score_1 = dict()
    type2score_2 = dict()
    types = set(type2p_1.keys()).union(set(type2p_2.keys()))
    for t in types:
        if t in type2p_1 and t in type2p_2:
            s1,s2 = get_entropy_type_scores(type2p_1[t], type2p_2[t], base, alpha)
        elif t in type2p_1:
            s1,s2 = get_entropy_type_scores(type2p_1[t], 0, base, alpha)
        else:
            s1,s2 = get_entropy_type_scores(0, type2p_2[t], base, alpha)

        type2score_1[t] = s1
        type2score_2[t] = s2

    return type2score_1,type2score_2

def get_entropy_type_scores(p_1, p_2, base, alpha):
    if alpha == 1:
        if p_1 > 0 and p_2 > 0:
            score_1 = -1 * log(p_1, base)
            score_2 = -1 * log(p_2, base)
        elif p_1 > 0:
            score_1 = -1 * log(p_1, base)
            score_2 = 0
        elif p_2 > 0:
            score_1 = 0
            score_2 = -1 *log(p_2, base)
    elif alpha > 1:
        score_1 = p_1**(alpha - 1) / (alpha - 1)
        score_2 = p_2**(alpha - 1) / (alpha - 1)
    else:
        if p_1 > 0 and p_2 > 0:
            score_1 = p_1**(alpha - 1) / (alpha - 1)
            score_2 = p_2**(alpha - 1) / (alpha - 1)
        elif p_1 > 0:
            score_1 = p_1**(alpha - 1) / (alpha - 1)
            score_2 = 0
        elif p_2 > 0:
            score_1 = 0
            score_2 = p_2**(alpha - 1) / (alpha - 1)


    return score_1,score_2


def get_jsd_scores(type2p_1, type2p_2, weight_1=0.5, weight_2=0.5, base=2, alpha=1):
    """
    Calculates the contribution of the types in two systems to the Jensen-Shannon
    divergence (JSD) between those systems

    Parameters
    ----------
    type2p_1, type2p_2: dict
        keys are types of a system and values are relative frequencies of those types
    weight_1, weight_2: float
        relative weights of type2p_1 and type2p_2 when constructing their mixed
        distribution. Should sum to 1
    base: int
        the base for the logarithm when computing entropy
    alpha: float
        the parameter for the generalized Tsallis entropy. Setting `alpha=`
        recovers the Shannon entropy. Higher `alpha` emphasizes more common
        types, lower `alpha` emphasizes less common types
        For details: https://en.wikipedia.org/wiki/Tsallis_entropy
    """
    type2m = dict()
    type2score_1 = dict()
    type2score_2 = dict()
    types = set(type2p_1.keys()).union(set(type2p_2.keys()))
    for t in types:
        if t in type2p_1 and t in type2p_2:
            p_1 = type2p_1[t]
            p_2 = type2p_2[t]
            m = weight_1 * p_1 + weight_2 * p_2
            s1,s2 = get_jsd_type_scores(p_1, p_2, m, weight_1, weight_2, base, alpha)
        elif t in type2p_1:
            p_1 = type2p_1[t]
            m = weight_1 * p_1
            s1,s2 = get_jsd_type_scores(p_1, 0, m, weight_1, weight_2, base, alpha)
        else:
            p_2 = type2p_2[t]
            m = weight_2 * p_2
            s1,s2 = get_jsd_type_scores(0, p_2, m, weight_1, weight_2, base, alpha)

        type2m[t] = m
        type2score_1[t] = s1
        type2score_2[t] = s2

    return type2m,type2score_1,type2score_2

def get_jsd_type_scores(p_1, p_2, m, weight_1, weight_2, base, alpha):
    """
    Calculates the JSD weighted average scores for a particular type in a system

    Parameters
    ----------
    p_1, p_2, m: float
        the probability of the type appearing in system 1, system 2, and the
        their mixture M
    weight_1, weight_2: float
        relative weights of type2p_1 and type2p_2 when constructing their mixed
        distribution. Should sum to 1
    base: int
        the base for the logarithm when computing entropy for the JSD
    alpha: float
    """
    if alpha == 1:
        if p_1 > 0 and p_2 > 0:
            score_1 = weight_1 * (log(m, base) - log(p_1, base))
            score_2 = weight_2 * (log(p_2, base) - log(m, base))
        elif p_1 > 0:
            score_1 = weight_1 * (log(m, base) - log(p_1, base))
            score_2 = weight_2 * (-log(m, base))
        elif p_2 > 0:
            score_1 = weight_1 * log(m, base)
            score_2 = weight_2 * (log(p_2, base) - log(m, base))
    elif alpha > 1:
        score_1 = weight_1 * (m**(alpha - 1) - p_1**(alpha - 1)) / (alpha - 1)
        score_2 = weight_2 * (m**(alpha - 1) - p_2**(alpha - 1)) / (alpha - 1)
    else:
        if p_1 > 0 and p_2 > 0:
            score_1 = weight_1 * (m**(alpha - 1) - p_1**(alpha - 1)) / (alpha - 1)
            score_2 = weight_2 * (m**(alpha - 1) - p_2**(alpha - 1)) / (alpha - 1)
        elif p_1 > 0:
            score_1 = weight_1 * (m**(alpha - 1) - p_1**(alpha - 1)) / (alpha - 1)
            score_2 = 0
        elif p_2 > 0:
            score_1 = 0
            score_2 = weight_2 * (m**(alpha - 1) - p_2**(alpha - 1)) / (alpha - 1)

    return score_1,score_2
