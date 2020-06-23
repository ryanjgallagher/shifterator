import math
from math import log


def get_relative_freqs(type2freq):
    """
    Calculates the relative frequency (proportion) of each type in a system

    Parameters
    ----------
    type2freq: dict
        Keys are types of a system and values are frequencies of those types

    Returns
    -------
    type2p: dict
        Keys are types and values are relative (normalized) frequencies
    """
    n = sum(type2freq.values())
    type2p = {t: s / n for t, s in type2freq.items()}
    return type2p


def get_entropy_scores(type2p_1, type2p_2, base=2, alpha=1):
    """
    Calculates the generalized Tsallis entropy scores for the difference in
    entropies of two systems

    Parameters
    ----------
    type2p_1, type2p_2: dict
        Keys are types of a system and values are relative frequencies of those types
    base: int, optional
        Base of the logarithm for calculating entropy
    alpha: float, optional
        The parameter for the generalized Tsallis entropy. Setting `alpha=1`
        recovers the Shannon entropy

    Returns
    -------
    type2score_1,type2score_2: dict
        Keys are types and values are the weights of each type for its contribution
        to the difference in entropies
    """
    type2score_1 = dict()
    type2score_2 = dict()
    types = set(type2p_1.keys()).union(set(type2p_2.keys()))
    for t in types:
        if t in type2p_1 and t in type2p_2:
            s1, s2 = get_entropy_type_scores(type2p_1[t], type2p_2[t], base, alpha)
        elif t in type2p_1:
            s1, s2 = get_entropy_type_scores(type2p_1[t], 0, base, alpha)
        else:
            s1, s2 = get_entropy_type_scores(0, type2p_2[t], base, alpha)

        type2score_1[t] = s1
        type2score_2[t] = s2

    return type2score_1, type2score_2


def get_entropy_type_scores(p_1, p_2, base, alpha):
    """
    Calculates the scores for a particular type in a system when calculating the
    difference in Tsallis entropy between two systems

    Parameters
    ----------
    p_1, p_2, float
        The probability of the type appearing in system 1 or system 2
    base: int
        The base for the logarithm when computing entropy
    alpha: float
        The parameter for the generalized Tsallis entropy. Setting `alpha=1`
        recovers the Shannon entropy.

    Returns
    -------
    score_1, score_2: float
        The weights of the type's contribution
    """
    if alpha == 1:
        if p_1 > 0 and p_2 > 0:
            score_1 = -1 * log(p_1, base)
            score_2 = -1 * log(p_2, base)
        elif p_1 > 0:
            score_1 = -1 * log(p_1, base)
            score_2 = 0
        elif p_2 > 0:
            score_1 = 0
            score_2 = -1 * log(p_2, base)
    elif alpha > 1:
        score_1 = p_1 ** (alpha - 1) / (alpha - 1)
        score_2 = p_2 ** (alpha - 1) / (alpha - 1)
    else:
        if p_1 > 0 and p_2 > 0:
            score_1 = p_1 ** (alpha - 1) / (alpha - 1)
            score_2 = p_2 ** (alpha - 1) / (alpha - 1)
        elif p_1 > 0:
            score_1 = p_1 ** (alpha - 1) / (alpha - 1)
            score_2 = 0
        elif p_2 > 0:
            score_1 = 0
            score_2 = p_2 ** (alpha - 1) / (alpha - 1)

    return score_1, score_2


def get_jsd_scores(type2p_1, type2p_2, weight_1=0.5, weight_2=0.5, base=2, alpha=1):
    """
    Calculates the contribution of the types in two systems to the Jensen-Shannon
    divergence (JSD) between those systems

    Parameters
    ----------
    type2p_1, type2p_2: dict
        Keys are types of a system and values are relative frequencies of those types
    weight_1, weight_2: float
        Relative weights of type2p_1 and type2p_2 when constructing their mixed
        distribution. Should sum to 1
    base: int
        The base for the logarithm when computing entropy
    alpha: float
        The parameter for the generalized Tsallis entropy. Setting `alpha=1`
        recovers the Shannon entropy.

    Returns
    -------
        Keys are types and values are the weights of each type for its contribution
        to the JSD
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
            s1, s2 = get_jsd_type_scores(p_1, p_2, m, weight_1, weight_2, base, alpha)
        elif t in type2p_1:
            p_1 = type2p_1[t]
            m = weight_1 * p_1
            s1, s2 = get_jsd_type_scores(p_1, 0, m, weight_1, weight_2, base, alpha)
        else:
            p_2 = type2p_2[t]
            m = weight_2 * p_2
            s1, s2 = get_jsd_type_scores(0, p_2, m, weight_1, weight_2, base, alpha)

        type2m[t] = m
        type2score_1[t] = s1
        type2score_2[t] = s2

    return type2m, type2score_1, type2score_2


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
        the base for the logarithm when computing entropy
    alpha: float
        the parameter for the generalized Tsallis entropy. Setting `alpha=1`
        recovers the Shannon entropy.

    Returns
    -------
    score_1, score_2: float
        The weights of the type's contribution
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
        score_1 = weight_1 * (m ** (alpha - 1) - p_1 ** (alpha - 1)) / (alpha - 1)
        score_2 = weight_2 * (m ** (alpha - 1) - p_2 ** (alpha - 1)) / (alpha - 1)
    else:
        if p_1 > 0 and p_2 > 0:
            score_1 = weight_1 * (m ** (alpha - 1) - p_1 ** (alpha - 1)) / (alpha - 1)
            score_2 = weight_2 * (m ** (alpha - 1) - p_2 ** (alpha - 1)) / (alpha - 1)
        elif p_1 > 0:
            score_1 = weight_1 * (m ** (alpha - 1) - p_1 ** (alpha - 1)) / (alpha - 1)
            score_2 = 0
        elif p_2 > 0:
            score_1 = 0
            score_2 = weight_2 * (m ** (alpha - 1) - p_2 ** (alpha - 1)) / (alpha - 1)

    return score_1, score_2


def get_type_surprisals(type2p, base=2, alpha=1):
    """
    Calculates the surprisal of each type in a system, i.e. log(1/p_i). Does not
    catch types with 0 relative frequency, assumed to be handled upstream

    Parameters
    ----------
    type2p: dict
        keys are types of a system and values are relative freqs of those types
    base: int
        the base of the logarithm
    alpha: float
        currently not implemented, but left for later updates
    """
    return {t: -math.log(p, base) for t, p in type2p.items()}


def tsallis_entropy(prob, count, alpha=1, base=math.e):
    """
    References
    ----------
        - https://arxiv.org/abs/1611.03596
        - https://en.wikipedia.org/wiki/Tsallis_entropy
    """
    if prob == 0:
        entropy = 0
    elif alpha == 1:
        entropy = -prob * math.log(prob, base)
    elif alpha >= 0:
        entropy = (prob ** alpha - 1 / count) / (1 - alpha)
    else:
        raise ValueError(f"Expected 0 <= alpha, received alpha = {alpha}!")
    return entropy


def get_tsallis_jsd_scores(
    type2freq_1, type2freq_2, base=math.e, alpha=1, weight_1=0.5, weight_2=0.5
):
    type2p = get_relative_freqs(type2freq_1)
    type2q = get_relative_freqs(type2freq_2)
    type2m = get_mixed_distribution(type2p, type2q, p=weight_1, q=weight_2)

    p_count = len(type2p)
    q_count = len(type2q)
    m_count = len(type2m)

    type2p = {
        t: tsallis_entropy(p, p_count, alpha=alpha, base=base)
        for t, p in type2p.items()
    }
    type2q = {
        t: tsallis_entropy(q, q_count, alpha=alpha, base=base)
        for t, q in type2q.items()
    }
    type2m = {
        t: tsallis_entropy(m, m_count, alpha=alpha, base=base)
        for t, m in type2m.items()
    }

    type2score_1 = {
        t: 0.5 * (type2m[t] - type2p[t]) if t in type2p else 0 for t in type2m
    }
    type2score_2 = {
        t: 0.5 * (type2q[t] - type2m[t]) if t in type2q else 0 for t in type2m
    }
    return type2p, type2q, type2m, type2score_1, type2score_2


def get_mixed_distribution(type2p, type2q, p=0.5, q=0.5):
    """
    Calculates the additive mixed distribution of two other distributions

    Parameters
    ----------
    type2p, type2q: dict
        keys are types of a system and values are relative freqs of those types
    p, q: float
        relative weights of each distribution in the mixed distribution. Should
        sum to 1.
    """
    types = set(type2p.keys()).union(set(type2q.keys()))
    return {t: p * type2p.get(t, 0) + q * type2q.get(t, 0) for t in types}
