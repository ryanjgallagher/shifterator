"""
helper.py
"""
import os
import pkgutil
import collections

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
    for lower_stop,upper_stop in stop_lens:
        for t in type2score:
            if ((type2score[t] < lower_stop) or (type2score[t] > upper_stop))\
            and t not in stop_words:
                try:
                    type2freq_new[t] = type2freq[t]
                except KeyError:
                    pass
                type2score_new[t] = type2score[t]
            else:
                stop_words.add(t)

    return (type2freq_new, type2score_new, stop_words)

def get_score_dictionary(scores, encoding='utf-8'):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    scores: dict or str
        if dict, then returns the dict automatically. If str, then it is either
        the name of a lexicon included in Shifterator

    Returns
    -------
    type2score, dict
        dictionary where keys are types and values are scores of those types
    """
    if isinstance(scores, collections.Mapping):
        return scores.copy(),None

    # Else, load scores from predefined score file in shifterator
    try:
        lexicon = scores.split('_')[0]
        score_f = 'lexicons/{}/{}.tsv'.format(lexicon, scores)
        all_scores = pkgutil.get_data(__name__, score_f).decode(encoding)
        if 'labMT' in lexicon:
            lexicon_ref = 5
        elif 'SocialSent' in lexicon:
            lexicon_ref = 0
        elif 'NRC' in lexicon:
            lexicon_ref = 0.5
    except FileNotFoundError:
        raise FileNotFoundError('Lexicon does not exit in Shifterator: {}'.format(scores))
    # Parse scores from all_scores, which is just a long str
    # Score files are line delimited with two tab-spaced columns: type and score
    type_scores = all_scores.split('\n')
    type2score = dict()
    for t_s in type_scores:
        if len(t_s) == 0:
            continue
        t,s = t_s.split('\t')
        type2score[t] = float(s)

    return type2score,lexicon_ref

def get_missing_scores(type2score_1, type2score_2):
    """
    Get missing scores between systems by setting the score in one system with
    the score in the other system

    Parameters
    ----------
    type2score_1, type2score_2: dict
        keys are types and values are scores

    Output
    ------
    type2score_1, type2score_2: dict
        keys are types and values are scores, updated to have scores across all
        types between the two score dictionaries
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
    return (type2score_1, type2score_2, missing_types)
