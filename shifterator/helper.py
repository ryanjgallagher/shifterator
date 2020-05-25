"""
Author: Ryan J. Gallagher, Network Science Institute, Northeastern University

TODO:
 - Allow different order entropies to be specified using alpha
"""
import math
import os


# ------------------------------------------------------------------------------
# -------------------------- Relative Frequency Funcs --------------------------
# ------------------------------------------------------------------------------
def get_relative_freqs(type2freq):
    """
    Calculates the relative frequency (proportion) of each type in a system

    Parameters
    ----------
    type2freq: dict
        keys are types of a system and values are frequencies of those types
    """
    n = sum(type2freq.values())
    type2p = {t: s / n for t, s in type2freq.items()}
    return type2p


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


# ------------------------------------------------------------------------------
# -------------------------------- Score Funcs ---------------------------------
# ------------------------------------------------------------------------------
def filter_by_scores(type2freq, type2score, stop_lens):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    type2freq: dict
        keys are types, values are frequencies of those types
    type2score: dict
        keys are types, values are scores associated with those types
    stop_lens: iteratble of 2-tuples
        denotes intervals that should be excluded when calculating shift scores

    Returns
    -------
    type2freq_new,type2score_new: dict,dict
        Frequency and score dicts filtered of words whose score fall within stop
        window
    """
    type2freq_new = dict()
    type2score_new = dict()
    stop_words = set()
    for lower_stop, upper_stop in stop_lens:
        for t in type2score:
            if (
                (type2score[t] < lower_stop) or (type2score[t] > upper_stop)
            ) and t not in stop_words:
                try:
                    type2freq_new[t] = type2freq[t]
                except KeyError:
                    pass
                type2score_new[t] = type2score[t]
            else:
                stop_words.add(t)

    return type2freq_new, type2score_new, stop_words


def get_score_dictionary(scores):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    scores: dict or str
        if dict, then returns the dict automatically. If str, then it is either
        the name of a shifterator dictionary to load, or file path of dictionary
        to load. File should be two columns of types and scores on each line,
        separated by delimiter
            Options: 'labMT_english'

    Returns
    -------
    type2score: dict
        Keys are types and values are scores of those types
    """
    if type(scores) is dict:
        return scores.copy()
    # Check if dictionary name is in shifterator data
    score_dicts = os.listdir("data")
    if scores in score_dicts:
        dict_file = "data/" + scores
    elif scores + ".csv" in score_dicts:
        dict_file = "data/" + scores + ".csv"
    else:  # Assume file path
        dict_file = scores
    # Load score dictionary
    type2score = {}
    with open(dict_file, "r") as f:
        for line in f:
            t, score = line.strip().split("\t")
            type2score[t] = score

    return type2score


def get_missing_scores(type2score_1, type2score_2):
    """
    Get missing scores between systems by setting the score in one system with
    the score in the other system

    Parameters
    ----------
    type2score_1, type2score_2: dict
        keys are types and values are scores

    Returns
    ------
    type2score_1, type2score_2: dict
        keys are types and values are scores, updated to have scores across all
        types between the two score dictionaries

    missing_types: set
        Keys that were present in only one of the provided dicts.
    """
    missing_types = set()
    types = set(type2score_1.keys()).union(set(type2score_2.keys()))
    for t in types:
        if t not in type2score_1:
            type2score_1[t] = type2score_2[t]
            missing_types.add(t)
        elif t not in type2score_2:
            type2score_2[t] = type2score_1[t]
            missing_types.add(t)
    return type2score_1, type2score_2, missing_types


# ------------------------------------------------------------------------------
# -------------------------------- Entropy Funcs -------------------------------
# ------------------------------------------------------------------------------
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


def get_type_logs(type2p, base=2, alpha=1, force_zero=False):
    """
    Calculates the logarithm of each type in a system, i.e. log(p_i)

    Parameters
    ----------
    type2p: dict
        keys are types of a system and values are relative freqs of those types
    base: int
        the base of the logarithm
    alpha: float
        currently not implemented, but left for later updates
    force_zero: boolean
        if True, force any type with 0 probability to have log(p_i) = log(0) = 0
        This is mathematically invalid, but a useful workaround for calculating
        the JSD, where even though we calculate log(p) here individually, it is
        recombined with other scores to get p * log(p), which should be 0 if
        both p and log(p) are zero.
    """
    type2log = dict()
    for t, p in type2p.items():
        try:
            type2log[t] = math.log(p, base)
        except ValueError:
            if force_zero:
                type2log[t] = 0
            else:
                raise
    return type2log


def get_surprisal_scores(system_1, system_2, base=2, alpha=1):
    """
    Gets all surprisal scores necessary for calculating probability divergence
    measures like the KLD or JSD

    Parameters
    ----------
    system_1, system_2: dict
        keys are types of a system and values are frequencies of those types
    base: int
        the base for the logarithm when computing entropy for the JSD
    alpha: float
        currently not implemented, but left for later updates
    """
    # Normalize reference and comparison frequencies
    type2p_1 = get_relative_freqs(system_1)
    type2p_2 = get_relative_freqs(system_2)
    # Get surprisal of each type
    type2surprisal_1 = get_type_surprisals(type2p_1, base=base, alpha=alpha)
    type2surprisal_2 = get_type_surprisals(type2p_2, base=base, alpha=alpha)
    return type2p_1, type2p_2, type2surprisal_1, type2surprisal_2


def get_jsd_scores(
    type2freq_1, type2freq_2, base=2, alpha=1, weight_1=0.5, weight_2=0.5
):
    """
    Calculates the contribution of the types in two systems to the Jensen-Shannon
    divergence (JSD) between those systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        keys are types of a system and values are frequencies of those types
    base: int
        the base for the logarithm when computing entropy for the JSD
    weight_1, weight_2: float
        relative weights of system_1 and system_2 when constructing their mixed
        distribution. Should sum to 1
    alpha: float
        currently not implemented, but left for later updates
    """
    # Normalize reference and comparison frequencies
    type2p = get_relative_freqs(type2freq_1)
    type2q = get_relative_freqs(type2freq_2)
    # Get mixed distribution
    type2m = get_mixed_distribution(type2p, type2q, p=weight_1, q=weight_2)
    # Get surprisal of each type
    # Forcing zero should be OK, by formula anything that has a 0 should be 0
    #   in the end when multiplied against its 0 frequency, i.e. 0 * log 0 = 0
    type2log_p = get_type_logs(type2p, base=base, alpha=alpha, force_zero=True)
    type2log_q = get_type_logs(type2q, base=base, alpha=alpha, force_zero=True)
    type2log_m = get_type_logs(type2m, base=base, alpha=alpha)
    # Get scores (handle missing types)
    type2score_1 = {
        t: 0.5 * (type2log_m[t] - type2log_p[t]) if t in type2log_p else 0
        for t in type2log_m
    }
    type2score_2 = {
        t: 0.5 * (type2log_q[t] - type2log_m[t]) if t in type2log_q else 0
        for t in type2log_m
    }
    return type2p, type2q, type2m, type2score_1, type2score_2


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
    type2freq_1, type2freq_2, base=2, alpha=1, weight_1=0.5, weight_2=0.5
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
