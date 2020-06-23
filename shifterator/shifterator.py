"""
<<<<<<< HEAD
TODO:
 - Make "system names" a parameter of the shift class, so that "Reference" and
   "Comparison" can be passed via the relative shift classes
 - Add funcs to shift class that allow for easy updating of type2freq dicts
 - Make it easy to remove / reset the filter. This will involve having to hold
   onto stop words, their freqs, and their values (discarded as of now)
 - Make it so you can specify words as stop words instead of just a filter window
 - Properly handle types without scores
 - Clean up class docstrings to fit standards of where things should be described
   (whether it's in init or under class, and listing what funcs are available)
=======
shifterator.py

TODO:
- Add funcs to shift class that allow for easy updating of type2freq dicts
- Make it easy to remove / reset the filter. This will involve having to hold
  onto stop words, their freqs, and their values (discarded as of now)
- Make it so you can specify words as stop words instead of just a filter window
>>>>>>> upstream/master
"""

import warnings

import matplotlib.pyplot as plt

from . import helper, plotting


class Shift:
    """
    Shift object for calculating weighted scores of two systems of types,
    and the shift between them
<<<<<<< HEAD
    """

    def __init__(
        self,
        system_1,
        system_2,
        type2score_1=None,
        type2score_2=None,
        reference_value=None,
        stop_lens=None,
    ):
        """
        Parameters
        ----------
        system_1, system_2: dict
            keys are types of a system and values are frequencies
            of those types
        type2score_1, type2score_2: dict or str, optional
            if dict, types are keys and values are "scores" associated with each
            type (e.g., sentiment). If str, either the name of a score dict or
            file path to a score dict, where types and scores are given on each
            line, separated by commas. If None and other type2score is None,
            defaults to uniform scores across types. Otherwise defaults to the
            other type2score dict
        reference_value: float, optional
            the reference score from which to calculate the deviation. If None,
            defaults to the weighted score of system_1
        stop_lens: iterable of 2-tuples, optional
            denotes intervals that should be excluded when calculating shift
            scores
        """
=======

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
    stop_lens: iterable of 2-tuples, optional
        Denotes intervals of scores that should be excluded when calculating
        shift scores, and types with scores in this range will be excluded
        from shift calculations
    normalization: str, optional
        If 'variation', normalizes shift scores so that the sum of
        their absolute values sums to 1. If 'trajectory', normalizes
        them so that the sum of shift scores is 1 or -1. The trajectory
        normalization cannot be applied if the total shift score is 0, so
        scores are left unnormalized if the total is 0 and 'trajectory' is
        specified
    """
    def __init__(self, type2freq_1, type2freq_2, type2score_1=None,
                 type2score_2=None, reference_value=None, stop_lens=None,
                 normalization='variation'):
>>>>>>> upstream/master
        # Set type2freq dictionaries
        self.type2freq_1 = type2freq_1.copy()
        self.type2freq_2 = type2freq_2.copy()
        # Set type2score dictionaries
        if type2score_1 is not None and type2score_2 is not None:
<<<<<<< HEAD
            self.type2score_1 = helper.get_score_dictionary(type2score_1)
            self.type2score_2 = helper.get_score_dictionary(type2score_2)
            self.show_score_diffs = True
        elif type2score_1 is not None:
            self.type2score_1 = helper.get_score_dictionary(type2score_1)
            self.type2score_2 = self.type2score_1
            self.show_score_diffs = False
        elif type2score_2 is not None:
            self.type2score_2 = helper.get_score_dictionary(type2score_2)
=======
            self.type2score_1,lex_ref = get_score_dictionary(type2score_1)
            self.type2score_2,_ = get_score_dictionary(type2score_2)
            if type2score_1 != type2score_2:
                self.show_score_diffs = True
            else:
                self.show_score_diffs = False
        elif type2score_1 is not None:
            self.type2score_1,lex_ref = get_score_dictionary(type2score_1)
            self.type2score_2 = self.type2score_1
            self.show_score_diffs = False
        elif type2score_2 is not None:
            self.type2score_2,lex_ref = get_score_dictionary(type2score_2)
>>>>>>> upstream/master
            self.type2score_1 = self.type2score_2
            self.show_score_diffs = False
        else:
            self.type2score_1 = {t: 1 for t in self.type2freq_1}
            self.type2score_2 = {t: 1 for t in self.type2freq_2}
            self.show_score_diffs = False
        # Filter type dictionaries by stop_lens
        self.stop_lens = stop_lens
        if stop_lens is not None:
            self.type2freq_1, self.type2score_1, sw_1 = helper.filter_by_scores(
                self.type2freq_1, self.type2score_1, stop_lens
            )
            self.type2freq_2, self.type2score_2, sw_2 = helper.filter_by_scores(
                self.type2freq_2, self.type2score_2, stop_lens
            )
            self.stop_words = sw_1.union(sw_2)
        # Get common vocabulary
<<<<<<< HEAD
        self.types = self.get_types(
            self.type2freq_1, self.type2score_1, self.type2freq_2, self.type2score_2
        )
        # Assume missing scores in each vocabulary
        # TODO: add options
        missing_scores_info = helper.get_missing_scores(
            self.type2score_1, self.type2score_2
        )
=======
        self.types = get_types(self.type2freq_1, self.type2score_1,
                               self.type2freq_2, self.type2score_2)
        # Assume missing scores in each vocabulary (TODO: add options)
        missing_scores_info = get_missing_scores(self.type2score_1, self.type2score_2)
>>>>>>> upstream/master
        self.type2score_1 = missing_scores_info[0]
        self.type2score_2 = missing_scores_info[1]
        self.missing_score_types = missing_scores_info[2]

        # Set reference value
        if reference_value is not None:
            if reference_value == 'average':
                self.reference_value = self.get_weighted_score(self.type2freq_1,
                                                               self.type2score_1)
            else:
                self.reference_value = reference_value
        else:
<<<<<<< HEAD
            self.reference_value = self.get_weighted_score(
                self.type2freq_1, self.type2score_1
            )
        # Set default score shift values
        self.diff = None
        self.type2p_diff = None
        self.type2s_diff = None
        self.type2p_avg = None
        self.type2s_ref_diff = None
        self.type2shift_score = None

    def get_types(self, type2freq_1, type2score_1, type2freq_2, type2score_2):
        """
        Parameters
        ----------
        type2freq_1, type2freq_2: dict
            keys are types and values are frequencies
        type2score_1, type2score_2: dict
            keys are types and values are scores

        Returns
        -------
        types: set
            Common "vocabulary" between the types of both systems and
            the types in the dictionaries
        """
        # Get observed types that are also in score dicts
        types_1 = set(type2freq_1.keys()).intersection(set(type2score_1.keys()))
        types_2 = set(type2freq_2.keys()).intersection(set(type2score_2.keys()))
        types = types_1.union(types_2)
        return types
=======
            if lex_ref is not None:
                self.reference_value = lex_ref
            else:
                self.reference_value = 0

        # Get shift scores
        self.normalization = normalization
        self.get_shift_scores(details=False)
>>>>>>> upstream/master

    def get_weighted_score(self, type2freq, type2score):
        """
        Calculate an average score according to a set of frequencies and scores

        Parameters
        ----------
        type2freq: dict
            Keys are types and values are frequencies
        type2score: dict
            Keys are types and values are scores

        Returns
        -------
        s_avg: float
            Average weighted score of system
        """
        # Check we have a vocabulary to work with
        types = set(type2freq.keys()).intersection(set(type2score.keys()))
        if len(types) == 0:
            return
        # Get weighted score and total frequency
<<<<<<< HEAD
        f_total = sum([freq for t, freq in type2freq.items() if t in types])
        s_weighted = sum(
            [type2score[t] * freq for t, freq in type2freq.items() if t in types]
        )
        s_avg = s_weighted / f_total
        return s_avg

    def get_shift_scores(
        self,
        type2freq_1=None,
        type2score_1=None,
        type2freq_2=None,
        type2score_2=None,
        reference_value=None,
        normalize=True,
        details=False,
    ):
=======
        f_total = sum([freq for t,freq in type2freq.items() if t in types])
        s_weighted = sum([type2score[t] * freq for t,freq in type2freq.items()
                          if t in types])
        s_avg = s_weighted / f_total
        return s_avg

    def get_shift_scores(self, details=False):
>>>>>>> upstream/master
        """
        Calculates the type shift scores between the two systems

        Parameters
        ----------
<<<<<<< HEAD
        type2freq_1, type2freq_2: dict
            keys are types and values are frequencies. If None, defaults to the
            system_1 and system_2 type2freq dicts respectively
        type2score_1, type2score_2: dict
            keys are types and values are scores. If None, defaults to the
            system_1 and system_2 type2score dicts respectively
        reference_value: float
            the reference score from which to calculate the deviation. If None,
            defaults to the weighted score given by type2freq_1 and type2score_1
        normalize: bool
            if True normalizes shift scores so they sum to 1 or -1
        details: bool
            Toggles the return of shift score components.
=======
        details: boolean
            If true, returns each of the major components of each type's shift
            score, along with the overall shift scores. Otherwise, only returns
            the overall shift scores
>>>>>>> upstream/master

        Returns
        -------
        type2p_diff: dict
            If details is True, returns dict where keys are types and values are
            the difference in relatively frequency, i.e. p_i,2 - p_i,1 for type i
        type2s_diff: dict,
            If details is True, returns dict where keys are types and values are
            the relative differences in score, i.e. s_i,2 - s_i,1 for type i
        type2p_avg: dict,
            If details is True, returns dict where keys are types and values are
            the average relative frequencies, i.e. 0.5*(p_i,1+p_i,2) for type i
        type2s_ref_diff: dict
            If details is True, returns dict where keys are types and values are
            relative deviation from reference score, i.e. 0.5*(s_i,2+s_i,1)-s_ref
            for type i
        type2shift_score: dict
            Keys are types and values are shift scores. The overall shift scores
            are normalized according to the `normalization` parameter of the
            Shift object
        """
<<<<<<< HEAD
        # Check input of type2freq and type2score dicts
        if type2freq_1 is None:
            type2freq_1 = self.type2freq_1
        if type2score_1 is None:
            type2score_1 = self.type2score_1
        if type2freq_2 is None:
            type2freq_2 = self.type2freq_2
        if type2score_2 is None:
            type2score_2 = self.type2score_2
        if reference_value is None:
            s_avg_ref = self.reference_value
        else:
            s_avg_ref = reference_value

        # Get type vocabulary
        types = self.get_types(type2freq_1, type2score_1, type2freq_2, type2score_2)

        # Get total frequencies
        total_freq_1 = sum([freq for t, freq in type2freq_1.items() if t in types])
        total_freq_2 = sum([freq for t, freq in type2freq_2.items() if t in types])
        # Get relative frequency of types in both systems
        type2p_1 = {
            t: type2freq_1[t] / total_freq_1 if t in type2freq_1 else 0 for t in types
        }
        type2p_2 = {
            t: type2freq_2[t] / total_freq_2 if t in type2freq_2 else 0 for t in types
        }
=======
        s_avg_ref = self.reference_value

        # Get total frequencies
        total_freq_1 = sum([freq for t,freq in self.type2freq_1.items() if t in self.types])
        total_freq_2 = sum([freq for t,freq in self.type2freq_2.items() if t in self.types])
        # Get relative frequency of types in both systems
        type2p_1 = {t:self.type2freq_1[t] / total_freq_1 if t in self.type2freq_1 else 0
                    for t in self.types}
        type2p_2 = {t:self.type2freq_2[t] / total_freq_2 if t in self.type2freq_2 else 0
                    for t in self.types}
>>>>>>> upstream/master

        # Calculate shift components
        type2p_avg = dict()
        type2p_diff = dict()
        type2s_diff = dict()
        type2s_ref_diff = dict()
        type2shift_score = dict()
<<<<<<< HEAD
        for t in types:
            type2p_avg[t] = 0.5 * (type2p_1[t] + type2p_2[t])
            type2p_diff[t] = type2p_2[t] - type2p_1[t]
            type2s_diff[t] = type2score_2[t] - type2score_1[t]
            type2s_ref_diff[t] = 0.5 * (type2score_2[t] + type2score_1[t]) - s_avg_ref
            type2shift_score[t] = (
                type2p_diff[t] * type2s_ref_diff[t] + type2s_diff[t] * type2p_avg[t]
            )
=======
        for t in self.types:
            type2p_avg[t] = 0.5 * (type2p_1[t] + type2p_2[t])
            type2p_diff[t] = type2p_2[t] - type2p_1[t]
            type2s_diff[t] = self.type2score_2[t] - self.type2score_1[t]
            type2s_ref_diff[t] = 0.5 * (self.type2score_2[t] + self.type2score_1[t])\
                                 - s_avg_ref
            type2shift_score[t] = type2p_diff[t] * type2s_ref_diff[t]\
                                 + type2s_diff[t] * type2p_avg[t]
>>>>>>> upstream/master

        # Normalize the total shift scores
        total_diff = sum(type2shift_score.values())
        self.diff = total_diff
<<<<<<< HEAD
        if normalize and self.diff != 0:
            type2shift_score = {
                t: shift_score / abs(total_diff)
                for t, shift_score in type2shift_score.items()
            }
=======
        if self.normalization == 'variation':
            abs_sum = sum(abs(s) for s in type2shift_score.values())
            self.norm = abs_sum
        elif self.normalization == 'trajectory' and total_diff != 0:
            self.norm = abs(total_diff)
        else:
            self.norm = 1
        type2shift_score = {t : shift_score / self.norm for t,shift_score
                            in type2shift_score.items()}
>>>>>>> upstream/master

        # Set results in shift object
        self.type2p_diff = type2p_diff
        self.type2s_diff = type2s_diff
        self.type2p_avg = type2p_avg
        self.type2s_ref_diff = type2s_ref_diff
        self.type2shift_score = type2shift_score
        # Return shift scores
        if details:
            return (
                type2p_diff,
                type2s_diff,
                type2p_avg,
                type2s_ref_diff,
                type2shift_score,
            )
        else:
            return type2shift_score

<<<<<<< HEAD
    def get_shift_component_sums(
        self,
        type2freq_1=None,
        type2score_1=None,
        type2freq_2=None,
        type2score_2=None,
        reference_value=None,
        normalize=True,
    ):
=======
    def get_shift_component_sums(self):
>>>>>>> upstream/master
        """
        Calculates the cumulative contribution of each component of the different
        kinds of shift scores.

        Returns
        -------
        Dictionary with six keys, one for each of the different component
        contributions: pos_s_pos_p, pos_s_neg_p, neg_s_pos_p, neg_s_neg_p,
        pos_s, neg_s. Values are the total contribution from that component
        across all types
        """
        # Get shift scores
        if self.type2shift_score is None:
<<<<<<< HEAD
            shift_scores = self.get_shift_scores(
                type2freq_1,
                type2score_1,
                type2freq_2,
                type2score_2,
                reference_value,
                normalize,
                details=True,
            )
=======
            shift_scores = self.get_shift_scores(details=True)
>>>>>>> upstream/master
        else:
            shift_scores = [
                (
                    t,
                    self.type2p_diff[t],
                    self.type2s_diff[t],
                    self.type2p_avg[t],
                    self.type2s_ref_diff[t],
                    self.type2shift_score[t],
                )
                for t in self.type2s_diff
            ]

        # Sum up components of shift score
        pos_s_pos_p = 0
        pos_s_neg_p = 0
        neg_s_pos_p = 0
        neg_s_neg_p = 0
        pos_s = 0
        neg_s = 0
        for t, p_diff, s_diff, p_avg, s_ref_diff, _ in shift_scores:
            # Get contribution of p_diff*s_ref_diff term
            if s_ref_diff > 0:
                if p_diff > 0:
                    pos_s_pos_p += p_diff * s_ref_diff
                else:
                    pos_s_neg_p += p_diff * s_ref_diff
            else:
                if p_diff > 0:
                    neg_s_pos_p += p_diff * s_ref_diff
                else:
                    neg_s_neg_p += p_diff * s_ref_diff
            # Get contribution of s_diff term
            if s_diff > 0:
                pos_s += p_avg * s_diff
            else:
                neg_s += p_avg * s_diff

<<<<<<< HEAD
        return {
            "pos_s_pos_p": pos_s_pos_p,
            "pos_s_neg_p": pos_s_neg_p,
            "neg_s_pos_p": neg_s_pos_p,
            "neg_s_neg_p": neg_s_neg_p,
            "pos_s": pos_s,
            "neg_s": neg_s,
        }

    def get_shift_graph(
        self,
        top_n=50,
        normalize=True,
        text_size_inset=True,
        cumulative_inset=True,
        show_plot=True,
        filename=None,
        **kwargs
    ):
=======
        return {'pos_s_pos_p': pos_s_pos_p, 'pos_s_neg_p': pos_s_neg_p,
                'neg_s_pos_p': neg_s_pos_p, 'neg_s_neg_p': neg_s_neg_p,
                'pos_s': pos_s, 'neg_s': neg_s}

    def get_shift_graph(self, top_n=50, text_size_inset=True,
                        cumulative_inset=True, show_plot=True, filename=None,
                        **kwargs):
>>>>>>> upstream/master
        """
        Plot the shift graph between two systems of types

        Parameters
        ----------
        top_n: int, optional
            Display the top_n types as sorted by their absolute contribution to
            the difference between systems
<<<<<<< HEAD
        normalize: bool
            Toggles rescaling of bar heights.
        cumulative_inset, text_size_inset: bool
            Whether to show insets showing the cumulative contribution to the
            shift by ranked types, and the relative sizes of each system
        show_plot: bool
            Whether to show plot on finish
        filename: str
            If not None, name of the file for saving the word shift graph
=======
        cumulative_inset: bool, optional
            Whether to show an inset showing the cumulative contributions to the
            shift by ranked types
        text_size_inset: bool, optional
            Whether to show an inset showing the relative sizes of each system
        show_plot: bool, optional
            Whether to show plot when it is done being rendered
        filename: str, optional
            If not None, name of the file for saving the shift graph
>>>>>>> upstream/master

        Returns
        -------
        ax
            Matplotlib ax of shift graph. Displays shift graph if show_plot=True
        """
        # Set plotting parameters
        kwargs = plotting.get_plot_params(kwargs, self.show_score_diffs)

        # Get type score components
<<<<<<< HEAD
        if self.type2shift_score is None:
            self.get_shift_scores(details=False)
        type_scores = [
            (
                t,
                self.type2p_diff[t],
                self.type2s_diff[t],
                self.type2p_avg[t],
                self.type2s_ref_diff[t],
                self.type2shift_score[t],
            )
            for t in self.type2s_diff
        ]
=======
        type_scores = [(t, self.type2p_diff[t], self.type2s_diff[t],
                        self.type2p_avg[t], self.type2s_ref_diff[t],
                        self.type2shift_score[t]) for t in self.type2s_diff]
>>>>>>> upstream/master
        # Reverse sorting to get highest scores, then reverse top n for plotting
        type_scores = sorted(type_scores, key=lambda x: abs(x[-1]), reverse=True)[
            :top_n
        ]
        type_scores.reverse()

        # Get bar heights and colors
<<<<<<< HEAD
        if normalize and self.diff != 0:
            norm = abs(self.diff)
        else:
            norm = 1
        bar_dims = plotting.get_bar_dims(type_scores, norm, kwargs)
        bar_colors = plotting.get_bar_colors(type_scores, kwargs)
=======
        bar_dims = get_bar_dims(type_scores, self.norm, kwargs)
        bar_colors = get_bar_colors(type_scores, kwargs)
>>>>>>> upstream/master

        # Initialize plot
        f, ax = plt.subplots(figsize=(kwargs["width"], kwargs["height"]))
        ax.margins(kwargs["y_margin"])
        # Plot type contributions
        ax = plotting.plot_contributions(ax, top_n, bar_dims, bar_colors, kwargs)
        # Plot total sum contributions
        total_comp_sums = self.get_shift_component_sums()
        bar_order = plotting.get_bar_order(kwargs)
        ax, comp_bar_heights, bar_order = plotting.plot_total_contribution_sums(
            ax, total_comp_sums, bar_order, top_n, bar_dims, kwargs
        )
        # Get labels for bars
        type_labels = [t for (t, _, _, _, _, _) in type_scores]
        # Add indicator if type borrwed a score
        m_sym = kwargs["missing_symbol"]
        type_labels = [
            t + m_sym if t in self.missing_score_types else t for t in type_labels
        ]
        # Get labels for total contribution bars
        bar_labels = [kwargs["symbols"][b] for b in bar_order]
        labels = type_labels + bar_labels
        # Set font type
<<<<<<< HEAD
        if kwargs["serif"]:
            plotting.set_serif()
        if kwargs["detailed"]:
            ax = plotting.set_bar_labels(
                f,
                ax,
                top_n,
                labels,
                bar_dims["label_heights"],
                comp_bar_heights,
                kwargs,
            )
=======
        if kwargs['serif']:
            set_serif()
        # Set labels
        if kwargs['detailed']:
            ax = set_bar_labels(f, ax, top_n, labels, bar_dims['label_heights'],
                                comp_bar_heights, kwargs)
>>>>>>> upstream/master
        else:
            ax = plotting.set_bar_labels(
                f,
                ax,
                top_n,
                labels,
                bar_dims["total_heights"],
                comp_bar_heights,
                kwargs,
            )

        # Add center dividing line
<<<<<<< HEAD
        y_min, y_max = ax.get_ylim()
        ax.plot([0, 0], [1, y_max], "-", color="black", linewidth=1.0, zorder=20)
        # Add dividing line between words and component bars
        x_min, x_max = ax.get_xlim()
        ax.plot(
            [x_min, x_max], [top_n + 1, top_n + 1], "-", color="black", linewidth=0.7
        )
        if kwargs["show_total"]:
            ax.plot(
                [x_min, x_max],
                [top_n + 2.75, top_n + 2.75],
                "-",
                color="black",
                linewidth=0.5,
            )
=======
        ax.axvline(0, ls='-', color='black', lw=1.0, zorder=20)

        # Add dividing line between types and component bars
        ax.axhline(top_n+1, ls='-', color='black', lw=0.7, zorder=20)
        if kwargs['show_total']:
            ax.axhline(top_n+2.75, ls='-', color='black', lw=0.5, zorder=20)
>>>>>>> upstream/master

        # Set insets
        if cumulative_inset:
<<<<<<< HEAD
            f = plotting.get_cumulative_inset(f, self.type2shift_score, top_n, kwargs)
        if text_size_inset:
            f = plotting.get_text_size_inset(
                f, self.type2freq_1, self.type2freq_2, kwargs
            )
        # Set guidance arrows (for relative plot)
        # if guidance:
        #    ax = get_guidance_annotations(ax, top_n, annotation_text=None)

        # Make x-tick labels bigger, flip y-axis ticks and label every 5th one
        ax = plotting.set_ticks(ax, top_n, kwargs)
        # Set axis labels and title
        ax.set_xlabel(kwargs["xlabel"], fontsize=kwargs["xlabel_fontsize"])
        ax.set_ylabel(kwargs["ylabel"], fontsize=kwargs["ylabel_fontsize"])
        if kwargs["all_pos_contributions"] and "title" not in kwargs:
            kwargs["title"] = ""
        elif "title" not in kwargs:
            s_avg_1 = self.get_weighted_score(self.type2freq_1, self.type2score_1)
            s_avg_2 = self.get_weighted_score(self.type2freq_2, self.type2score_2)
            title = (
                r"$\Phi_{\Omega^{(2)}}$: $s_{avg}^{(1)}=$"
                + "{0:.2f}".format(s_avg_1)
                + "\n"
                + r"$\Phi_{\Omega^{(1)}}$: $s_{avg}^{(2)}=$"
                + "{0:.2f}".format(s_avg_2)
            )
            kwargs["title"] = title
        ax.set_title(kwargs["title"], fontsize=kwargs["title_fontsize"])
=======
            f = get_cumulative_inset(f, self.type2shift_score, top_n,
                                     self.normalization, kwargs)
        if text_size_inset:
            f = get_text_size_inset(f, self.type2freq_1, self.type2freq_2, kwargs)

        # Make x-tick labels bigger, flip y-axis ticks and label every 5th one
        ax = set_ticks(ax, top_n, kwargs)

        # Set axis spines
        ax = set_spines(ax, kwargs)

        # Set axis labels and title
        ax.set_xlabel(kwargs['xlabel'], fontsize=kwargs['xlabel_fontsize'])
        ax.set_ylabel(kwargs['ylabel'], fontsize=kwargs['ylabel_fontsize'])
        if 'title' not in kwargs:
            s_avg_1 = self.get_weighted_score(self.type2freq_1,self.type2score_1)
            s_avg_2 = self.get_weighted_score(self.type2freq_2,self.type2score_2)
            title = '{}: '.format(kwargs['system_names'][0])\
                    + r'$\Phi_{avg}=$'+'{0:.2f}'.format(s_avg_1)+'\n'\
                    +'{}: '.format(kwargs['system_names'][1])\
                    + r'$\Phi_{avg}=$'+'{0:.2f}'.format(s_avg_2)
            kwargs['title'] = title
        ax.set_title(kwargs['title'], fontsize=kwargs['title_fontsize'])
>>>>>>> upstream/master

        # Show and return plot
        if kwargs["tight"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=kwargs["dpi"])
        if show_plot:
            plt.show()
        return ax
