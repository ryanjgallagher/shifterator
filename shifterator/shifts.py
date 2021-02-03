from . import entropy
from .shifterator import Shift


class WeightedAvgShift(Shift):
    """
    Shift object for calculating weighted scores of two systems of types,
    and the shift between them

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        Keys are types of a system and values are frequencies of those types
    type2score_1, type2score_2: dict or str, optional
        If dict, types are keys and values are scores associated with each
        type. If str, the name of a score lexicon included in Shifterator.
        If None and other type2score is None, defaults to uniform scores
        across types. Otherwise defaults to the other type2score dict
    reference_value: str or float, optional
        The reference score to use to partition scores into two different
        regimes. If 'average', uses the average score according to type2freq_1
        and type2score_1. If None and a lexicon is selected for type2score,
        uses the respective middle point in that lexicon's scale. Otherwise
        if None, uses zero as the reference point
    handle_missing_scores: str, optional
        If 'error', throws an error whenever a word has a score in one score
        dictionary but not the other. If 'exclude', excludes any word that is
        missing a score in one score dictionary from all word shift
        calculations, regardless if it may have a score in the other dictionary.
        If 'adopt' and the score is missing in one dictionary, then uses the
        score from the other dictionary if it is available
    stop_lens: iterable of 2-tuples, optional
        Denotes intervals of scores that should be excluded from word shifts
        calculations. Types with scores in this range will be excluded from word
        shift calculations
    stop_words: set, optional
        Denotes words that should be excluded from word shifts calculations
    normalization: str, optional
        If 'variation', normalizes shift scores so that the sum of
        their absolute values sums to 1. If 'trajectory', normalizes
        them so that the sum of shift scores is 1 or -1. The trajectory
        normalization cannot be applied if the total shift score is 0, so
        scores are left unnormalized if the total is 0 and 'trajectory' is
        specified
    """

    def __init__(
        self,
        type2freq_1,
        type2freq_2,
        type2score_1=None,
        type2score_2=None,
        reference_value=None,
        handle_missing_scores="error",
        stop_lens=None,
        stop_words=set(),
        normalization="variation",
    ):
        super().__init__(
            type2freq_1=type2freq_1,
            type2freq_2=type2freq_2,
            type2score_1=type2score_1,
            type2score_2=type2score_2,
            reference_value=reference_value,
            handle_missing_scores=handle_missing_scores,
            stop_lens=stop_lens,
            stop_words=stop_words,
            normalization=normalization,
        )


class ProportionShift(Shift):
    """
    Shift object for calculating differences in proportions of types across two
    systems

    Parameters
    __________
    type2freq_1, type2freq_2: dict
        Keys are types of a system and values are frequencies of those types
    """

    def __init__(self, type2freq_1, type2freq_2):
        # Set relative frequency to 0 for types that don't appear
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        types = set(type2freq_1.keys()).union(type2freq_2.keys())
        for t in types:
            if t not in type2freq_1:
                type2freq_1[t] = 0
            elif t not in type2freq_2:
                type2freq_2[t] = 0
        # Initialize shift object
        super().__init__(
            type2freq_1=type2freq_1,
            type2freq_2=type2freq_2,
            type2score_1=None,
            type2score_2=None,
            reference_value=0,
            handle_missing_scores="error",
            stop_lens=None,
            stop_words=None,
            normalization="variation",
        )

    def get_shift_graph(
        self,
        top_n=50,
        show_plot=True,
        detailed=False,
        text_size_inset=True,
        cumulative_inset=True,
        title=None,
        filename=None,
        **kwargs
    ):
        if title is None:
            title = ""
        ax = super().get_shift_graph(
            top_n=top_n,
            text_size_inset=text_size_inset,
            cumulative_inset=cumulative_inset,
            detailed=detailed,
            show_plot=show_plot,
            filename=filename,
            show_total=False,
            title=title,
            **kwargs
        )
        return ax


class EntropyShift(Shift):
    """
    Shift object for calculating the shift in entropy between two systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        Keys are types of a system and values are frequencies of those types
    base: float, optional
        Base of the logarithm for calculating entropy
    alpha: float, optional
        The parameter for the generalized Tsallis entropy. Setting `alpha=1`
        recovers the Shannon entropy. Higher `alpha` emphasizes more common
        types, lower `alpha` emphasizes less common types
        For details: https://en.wikipedia.org/wiki/Tsallis_entropy
    reference_value: str or float, optional
        The reference score to use to partition scores into two different
        regimes. If 'average', uses the average score according to type2freq_1
        and type2score_1. Otherwise, uses zero as the reference point
    normalization: str, optional
        If 'variation', normalizes shift scores so that the sum of
        their absolute values sums to 1. If 'trajectory', normalizes
        them so that the sum of shift scores is 1 or -1. The trajectory
        normalization cannot be applied if the total shift score is 0, so
        scores are left unnormalized if the total is 0 and 'trajectory' is
        specified
    """

    def __init__(
        self,
        type2freq_1,
        type2freq_2,
        base=2,
        alpha=1,
        reference_value=0,
        normalization="variation",
    ):
        # Get relative frequencies
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1 = entropy.get_relative_freqs(type2freq_1)
        type2p_2 = entropy.get_relative_freqs(type2freq_2)
        # Get entropy scores
        type2s_1, type2s_2 = entropy.get_entropy_scores(type2p_1, type2p_2, base, alpha)

        # Initialize shift
        super().__init__(
            type2freq_1=type2freq_1,
            type2freq_2=type2freq_2,
            type2score_1=type2s_1,
            type2score_2=type2s_2,
            handle_missing_scores="error",
            stop_lens=None,
            stop_words=None,
            reference_value=reference_value,
            normalization=normalization,
        )
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2
        self.alpha = alpha

    def get_shift_graph(
        self,
        top_n=50,
        show_plot=True,
        detailed=False,
        text_size_inset=True,
        cumulative_inset=True,
        filename=None,
        **kwargs
    ):
        ax = super().get_shift_graph(
            top_n=top_n,
            text_size_inset=text_size_inset,
            cumulative_inset=cumulative_inset,
            detailed=detailed,
            show_plot=show_plot,
            filename=filename,
            **kwargs
        )
        return ax


class KLDivergenceShift(Shift):
    """
    Shift object for calculating the Kullback-Leibler divergence (KLD) between
    two systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        Keys are types of a system and values are frequencies of those types.
        The KLD will be computed with respect type2freq_1, i.e. D(T2 || T1).
        For the KLD to be well defined, all types must have nonzero frequencies
        in both type2freq_1 and type2_freq2
    base: float, optional
        Base of the logarithm for calculating entropy
    stop_lens: iterable of 2-tuples, optional
        Denotes intervals that should be excluded when calculating shift
        scores
    normalization: str, optional
        If 'variation', normalizes shift scores so that the sum of
        their absolute values sums to 1. If 'trajectory', normalizes
        them so that the sum of shift scores is 1 or -1. The trajectory
        normalization cannot be applied if the total shift score is 0, so
        scores are left unnormalized if the total is 0 and 'trajectory' is
        specified
    """

    def __init__(
        self,
        type2freq_1,
        type2freq_2,
        base=2,
        reference_value=0,
        normalization="variation",
    ):
        # Check that KLD is well defined
        types_1 = set(type2freq_1.keys())
        types_2 = set(type2freq_2.keys())
        if len(types_1.symmetric_difference(types_2)) > 0:
            err = (
                "There are types that appear in either type2freq_1 or "
                + "type2freq_2 but not the other: the KL divergence is not "
                + "well defined"
            )
            raise ValueError(err)

        # Get relative frequencies
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1 = entropy.get_relative_freqs(type2freq_1)
        type2p_2 = entropy.get_relative_freqs(type2freq_2)
        # Get surprisal scores
        type2s_1 = {t: p * -1 * entropy.log(p, base) for t, p in type2p_1.items()}
        type2s_2 = {t: p * -1 * entropy.log(p, base) for t, p in type2p_2.items()}

        # Initialize shift
        super().__init__(
            type2freq_1=type2p_2,
            type2freq_2=type2p_2,
            type2score_1=type2s_1,
            type2score_2=type2s_2,
            handle_missing_scores="error",
            stop_lens=None,
            stop_words=None,
            reference_value=reference_value,
            normalization=normalization,
        )
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2

    def get_shift_graph(
        self,
        top_n=50,
        show_plot=True,
        detailed=False,
        text_size_inset=True,
        cumulative_inset=True,
        title=None,
        filename=None,
        **kwargs
    ):
        if title is None:
            title = ""
        ax = super().get_shift_graph(
            top_n=top_n,
            text_size_inset=text_size_inset,
            cumulative_inset=cumulative_inset,
            detailed=detailed,
            show_plot=show_plot,
            title=title,
            filename=filename,
            **kwargs
        )
        return ax


class JSDivergenceShift(Shift):
    """
    Shift object for calculating the Jensen-Shannon divergence (JSD) between two
    systems

    Parameters
    ----------
    type2freq_1, type2freq_2: dict
        Keys are types of a system and values are frequencies of those types
    weight_1, weight_2: float
        Relative weights of type2freq_1 and type2frq_2 when constructing their
        mixed distribution. Should sum to 1
    base: float, optional
        Base of the logarithm for calculating entropy
    alpha: float, optional
        The parameter for the generalized Tsallis entropy. Setting `alpha=1`
        recovers the Shannon entropy. Higher `alpha` emphasizes more common
        types, lower `alpha` emphasizes less common types
        For details: https://en.wikipedia.org/wiki/Tsallis_entropy
    reference_value: str or float, optional
        The reference score to use to partition scores into two different
        regimes. Defaults to zero as the reference point
    normalization: str, optional
        If 'variation', normalizes shift scores so that the sum of
        their absolute values sums to 1. If 'trajectory', normalizes
        them so that the sum of shift scores is 1 or -1. The trajectory
        normalization cannot be applied if the total shift score is 0, so
        scores are left unnormalized if the total is 0 and 'trajectory' is
        specified
    """

    def __init__(
        self,
        type2freq_1,
        type2freq_2,
        base=2,
        weight_1=0.5,
        weight_2=0.5,
        alpha=1,
        reference_value=0,
        normalization="variation",
    ):
        # Check weights
        if weight_1 + weight_2 != 1:
            raise ValueError("weight_1 and weight_2 do not sum to 1")

        # Get relative frequencies
        type2freq_1 = type2freq_1.copy()
        type2freq_2 = type2freq_2.copy()
        type2p_1 = entropy.get_relative_freqs(type2freq_1)
        type2p_2 = entropy.get_relative_freqs(type2freq_2)
        # Get shift scores
        type2m, type2s_1, type2s_2 = entropy.get_jsd_scores(
            type2p_1,
            type2p_2,
            weight_1=weight_1,
            weight_2=weight_2,
            base=base,
            alpha=alpha,
        )

        # Initialize shift object
        super().__init__(
            type2freq_1=type2freq_1,
            type2freq_2=type2freq_2,
            type2score_1=type2s_1,
            type2score_2=type2s_2,
            reference_value=reference_value,
            handle_missing_scores="error",
            normalization=normalization,
            stop_lens=None,
            stop_words=None,
        )
        self.type2p_1 = type2p_1
        self.type2p_2 = type2p_2
        self.type2m = type2m
        self.alpha = alpha

    def get_shift_graph(
        self,
        top_n=50,
        show_plot=True,
        detailed=False,
        text_size_inset=True,
        cumulative_inset=True,
        title=None,
        filename=None,
        **kwargs
    ):
        if self.alpha == 1 and self.reference_value == 0:
            all_pos_contributions = True
        else:
            all_pos_contributions = False
        if title is None:
            title = ""
        ax = super().get_shift_graph(
            top_n=top_n,
            text_size_inset=text_size_inset,
            cumulative_inset=cumulative_inset,
            detailed=detailed,
            show_plot=show_plot,
            filename=filename,
            title=title,
            all_pos_contributions=all_pos_contributions,
            **kwargs
        )
        return ax
