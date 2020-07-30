import collections
import pkgutil


def preprocess_words_scores(type2freq_1, type2score_1, type2freq_2, type2score_2,
                            stop_lens, stop_words, handle_missing_scores):
    """
    Filters stop words according to a list of words or stop lens on the scores

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        Keys are types, values are frequencies of those types
    type2score_1, type2freq_2: dict
        Keys are types, values are scores associated with those types
    stop_lens: iteratble of 2-tuples
        Denotes intervals that should be excluded from word shifts
    stop_words: iterable
        Denotes words that should be excluded from word shifts
    handle_missing_scores_scores
        If 'error', throws an error whenever a word has a score in one score
        dictionary but not the other. If 'exclude', excludes any word that is
        missing a score in one score dictionary from all word shift
        calculations, regardless if it may have a score in the other dictionary.
        If 'adopt' and the score is missing in one dictionary, then uses the
        score from the other dictionary if it is available
    """
    ts_1 = set(type2freq_1.keys()).union(set(type2score_1.keys()))
    ts_2 = set(type2freq_2.keys()).union(set(type2score_2.keys()))
    ts = ts_1.union(ts_2)

    type2freq_1_new = dict()
    type2score_1_new = dict()
    type2freq_2_new = dict()
    type2score_2_new = dict()
    adopted_score_types = set()
    no_score_types = set()
    filtered_types = set()
    for t in ts:
        # Exclude words specified by stop words
        if t in stop_words:
            filtered_types.add(t)
            continue

        # Handle words with missing scores before excluding based on stop lens
        if t in type2score_1:
            s_1 = type2score_1[t]
        else:
            s_1 = None
        if t in type2score_2:
            s_2 = type2score_2[t]
        else:
            s_2 = None
        # Word does not have score in either dictioary
        if t not in type2score_1 and t not in type2score_2:
            no_score_types.add(t)
            continue
        # Word has score in dict2 but not dict1
        elif t not in type2score_1 and t in type2score_2:
            if handle_missing_scores == "adopt":
                s_1 = type2score_2[t]
                s_2 = type2score_2[t]
                adopted_score_types.add(t)
            elif handle_missing_scores == "error":
                raise KeyError(
                    "Word has freq but no score in type2score_1: {}".format(t)
                )
            elif handle_missing_scores == "exclude":
                no_score_types.add(t)
                continue
            else:
                raise ValueError(
                    "handle_missing_scores has not been provided a valid argument"
                )
        # Word has score in dict1 but not dict2
        elif t in type2score_1 and t not in type2score_2:
            if handle_missing_scores == "adopt":
                s_1 = type2score_1[t]
                s_2 = type2score_1[t]
                adopted_score_types.add(t)
            elif handle_missing_scores == "error":
                raise KeyError(
                    "Word has freq but no score in type2score_2: {}".format(t)
                )
            elif handle_missing_scores == "exclude":
                filtered_types.add(t)
                continue
            else:
                raise ValueError(
                    "handle_missing_scores has not been provided a valid argument"
                )
        # Word has score in dict1 and dict2
        else:
            s_1 = type2score_1[t]
            s_2 = type2score_2[t]

        # Exclude words based on stop lens
        filter_word = False
        for lower, upper in stop_lens:
            # Word is in stop lens
            if (lower <= s_1 and s_1 <= upper) and (lower <= s_2 and s_2 <= upper):
                filter_word = True
            # One score is in stop lens but the other is not
            elif (lower <= s_1 and s_1 <= upper) or (lower <= s_2 and s_2 <= upper):
                raise ValueError(
                    "{}: stop_lens cannot be applied consistently.".format(t)\
                    + " One word score falls within the stop lens while the"\
                    + " other does not."
                )
        if filter_word:
            filtered_types.add(t)
            continue

        # Set words and freqs for words that pass all checks
        type2score_1_new[t] = s_1
        if t in type2freq_1:
            type2freq_1_new[t] = type2freq_1[t]
        else:
            type2freq_1_new[t] = 0

        type2score_2_new[t] = s_2
        if t in type2freq_2:
            type2freq_2_new[t] = type2freq_2[t]
        else:
            type2freq_2_new[t] = 0

    # Update types to only be those that made it through all filters
    final_types = ts.difference(filtered_types).difference(no_score_types)

    return (
        type2freq_1_new,
        type2freq_2_new,
        type2score_1_new,
        type2score_2_new,
        final_types,
        filtered_types,
        no_score_types,
        adopted_score_types
    )

def get_score_dictionary(scores):
    """
    Loads a dictionary of type scores

    Parameters
    ----------
    scores: dict or str
        If dict, then returns the dict automatically. If str, then the name of a
        lexicon included in Shifterator

    Returns
    -------
    type2score: dict
        Keys are types and values are scores of those types
    """
    if isinstance(scores, collections.Mapping):
        return scores.copy(), None

    # Else, load scores from predefined score file in shifterator
    try:
        lexicon = scores.split("_")[0]
        score_f = "lexicons/{}/{}.tsv".format(lexicon, scores)
        all_scores = pkgutil.get_data(__name__, score_f).decode("utf-8")
        if "labMT" in lexicon:
            lexicon_ref = 5
        elif "SocialSent" in lexicon:
            lexicon_ref = 0
        elif "NRC" in lexicon:
            lexicon_ref = 0.5
    except FileNotFoundError:
        raise FileNotFoundError(
            "Lexicon does not exist in Shifterator: {}".format(scores)
        )
    # Parse scores from all_scores, which is just a long str
    # Score files are line delimited with two tab-spaced columns: type and score
    type_scores = all_scores.split("\n")
    type2score = dict()
    for t_s in type_scores:
        if len(t_s) == 0:
            continue
        t, s = t_s.split("\t")
        type2score[t] = float(s)

    return type2score, lexicon_ref
