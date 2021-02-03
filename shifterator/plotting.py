"""
TODO:
 - Add params for explicitly setting fonts
"""
import numpy as np
from matplotlib import rcParams


def get_plot_params(plot_params, show_score_diffs, diff):
    defaults = {
        "all_pos_contributions": False,
        "alpha_fade": 0.35,
        "bar_linewidth": 0.25,
        "bar_type_space_scaling": 0.015,
        "bar_width": 0.8,
        "cumulative_xlabel": None,
        "cumulative_xticklabels": None,
        "cumulative_xticks": None,
        "cumulative_ylabel": None,
        "detailed": True,
        "dpi": 200,
        "every_nth_ytick": 5,
        "height": 15,
        "invisible_spines": [],
        "label_fontsize": 13,
        "missing_symbol": "*",
        "pos_cumulative_inset": [0.19, 0.12, 0.175, 0.175],
        "pos_text_size_inset": [0.81, 0.12, 0.08, 0.08],
        "remove_xticks": False,
        "remove_yticks": False,
        "score_colors": {
            "all_pos_neg": "#9E75B7",
            "all_pos_pos": "#FECC5D",
            "neg_s": "#9E75B7",
            "neg_s_neg_p": "#C4CAFC",
            "neg_s_pos_p": "#2F7CCE",
            "neg_total": "#9E75B7",
            "pos_s": "#FECC5D",
            "pos_s_neg_p": "#FDFFD2",
            "pos_s_pos_p": "#FFFF80",
            "pos_total": "#FECC5D",
            "total": "#707070",
        },
        "serif": False,
        "show_score_diffs": show_score_diffs,
        "show_total": True,
        "system_names": ["Text 1", "Text 2"],
        "tick_format": "{:.1f}",
        "tight": True,
        "title_fontsize": 18,
        "width": 7,
        "width_scaling": 1.2,
        "xlabel": r"Score shift $\delta \Phi_{\tau}$ (%)",
        "xlabel_fontsize": 20,
        "xtick_fontsize": 14,
        "y_margin": 0.005,
        "ylabel": r"Rank",
        "ylabel_fontsize": 20,
        "ytick_fontsize": 14,
    }
    defaults["symbols"] = {
        "all_pos_neg": defaults["system_names"][0],
        "all_pos_pos": defaults["system_names"][1],
        "neg_s": u"\u25BD",
        "neg_s_neg_p": u"-\u2193",
        "neg_s_pos_p": u"-\u2191",
        "neg_total": "",
        "pos_s": u"\u25B3",
        "pos_s_neg_p": u"+\u2193",
        "pos_s_pos_p": u"+\u2191",
        "pos_total": "",
        "total": r"$\Sigma$",
    }
    defaults.update(plot_params)
    return defaults


def set_serif():
    rcParams["font.family"] = "serif"
    rcParams["mathtext.fontset"] = "dejavuserif"


def get_bar_dims(type_scores, norm, plot_params):
    """
    Gets the height and location of every bar needed to plot each type's
    contribution.

    Parameters
    ----------
    type_scores: list of tuples
        List of tuples of the form (type,p_diff,s_diff,p_avg,s_ref_diff,shift_score)
        for every type scored in the two systems. This is the detailed output
        of a Shift object's `get_shift_scores`
    norm: float
        The factor by which to normalize all the component scores
    plot_params: dict
        Dictionary of plotting parameters. Here, `all_pos_contributions` is used

    Returns
    -------
    Dictionary with nine keys: `p_solid_heights`, `s_solid_bases`,
    `s_solid_heights`, `p_fade_heights`, `p_fade_bases`, `s_fade_bases`,
    `s_fade_heights`, `total_heights`, `label_heights`. Values are lists are the
    corresponding bar dimensions for each word

    'p' stands for the component with p_diff
    's' stands for the component with s_diff.
    'solid' indicates the part of the contribution that is not alpha faded
    'base' stands for where the bottom of the bar is
    'height' stands for the height relative to the base
        Note, `p_solid_base` would always be 0, which is why it is not included
    `total_heights` is the overall contribution for simple (not detailed) shift
        graphs (base is always 0).
    `label_heights` is the label position after making up for counteracting components
    """
    # 'p' for p_diff component, 's' for s_diff component
    # 'solid' for part of comp that is not alpha faded, 'faded' otherwise
    # 'base' for where bottom of bar is, 'height' for height from that base
    #     note, 'p_solid_base' is always 0
    # 'total' for total contribution for simple word shift graphs (base always 0)
    dims = {
        "p_solid_heights": [],
        "s_solid_bases": [],
        "s_solid_heights": [],
        "p_fade_heights": [],
        "p_fade_bases": [],
        "s_fade_bases": [],
        "s_fade_heights": [],
        "total_heights": [],
        "label_heights": [],
    }
    for (_, p_diff, s_diff, p_avg, s_ref_diff, _) in type_scores:
        c_p = 100 * p_diff * s_ref_diff / norm
        c_s = 100 * p_avg * s_diff / norm
        # This is for JSD to make bars face different directions based on p
        # p_diff is p_2 - p_1, so point to right if p_1 > p_2
        if not plot_params["all_pos_contributions"] or p_diff > 0:
            dims["total_heights"].append(c_p + c_s)
        else:
            dims["total_heights"].append(-1 * (c_p + c_s))
        # Determine if direction of comp bars are congruent
        if np.sign(s_ref_diff * p_diff) * np.sign(s_diff) == 1:
            dims["p_solid_heights"].append(c_p)
            dims["s_solid_bases"].append(c_p)
            dims["s_solid_heights"].append(c_s)
            dims["label_heights"].append(c_p + c_s)
            for d in [
                "p_fade_bases",
                "p_fade_heights",
                "s_fade_bases",
                "s_fade_heights",
            ]:
                dims[d].append(0)
        else:
            if abs(c_p) > abs(c_s):
                dims["p_solid_heights"].append(c_p + c_s)
                dims["p_fade_bases"].append(c_p + c_s)
                dims["p_fade_heights"].append(-1 * c_s)
                dims["s_fade_heights"].append(c_s)
                dims["label_heights"].append(c_p)
                for d in ["s_solid_bases", "s_solid_heights", "s_fade_bases"]:
                    dims[d].append(0)
            else:
                dims["s_solid_heights"].append(c_s + c_p)
                dims["p_fade_heights"].append(c_p)
                dims["s_fade_bases"].append(c_s + c_p)
                dims["s_fade_heights"].append(-1 * c_p)
                dims["label_heights"].append(c_s)
                for d in ["p_solid_heights", "s_solid_bases", "p_fade_bases"]:
                    dims[d].append(0)
    return dims


def get_bar_colors(type_scores, plot_params):
    """
    Returns the component colors of each type's contribution bars.

    Parameters
    ----------
    type_scores: list of tuples
        List of tuples of the form (type,p_diff,s_diff,p_avg,s_ref_diff,shift_score)
        for every type scored in the two systems. This is the detailed output
        of a Shift object's `get_shift_scores`
    plot_params: dict
        Dictionary of plotting parameters. Here, `all_pos_contributions` and
        `score_colors` are used

    Returns
    -------
    Dictionary with three keys: `p`, `s`, and `total`. Values are lists of the
    colors to assign to the p_diff and s_diff components respectively. If just
    the overall contributions are being shown in a simple (not detailed) shift
    graph, then the `total` colors are used
    """
    score_colors = plot_params["score_colors"]
    bar_colors = {"p": [], "s": [], "total": []}
    for (_, p_diff, s_diff, p_avg, s_ref_diff, _) in type_scores:
        c_total = p_diff * s_ref_diff + p_avg * s_diff
        # Get total contribution colors
        if not plot_params["all_pos_contributions"]:
            if c_total > 0:
                bar_colors["total"].append(score_colors["pos_total"])
            else:
                bar_colors["total"].append(score_colors["neg_total"])
        else:
            if p_diff > 0:
                bar_colors["total"].append(score_colors["all_pos_pos"])
            else:
                bar_colors["total"].append(score_colors["all_pos_neg"])
        # Get p_diff * s_ref_diff comp colors
        if s_ref_diff > 0:
            if p_diff > 0:
                bar_colors["p"].append(score_colors["pos_s_pos_p"])
            else:
                bar_colors["p"].append(score_colors["pos_s_neg_p"])
        else:
            if p_diff > 0:
                bar_colors["p"].append(score_colors["neg_s_pos_p"])
            else:
                bar_colors["p"].append(score_colors["neg_s_neg_p"])
        # Get s_diff comp colors
        if s_diff > 0:
            bar_colors["s"].append(score_colors["pos_s"])
        else:
            bar_colors["s"].append(score_colors["neg_s"])

    return bar_colors


def plot_contributions(ax, top_n, bar_dims, bar_colors, plot_params):
    """
    Plots all of the type contributions as horizontal bars

    Parameters
    ----------
    ax: Matplotlib ax
        Current ax of the shift graph
    top_n: int
        The number of types being plotted on the shift graph
    bar_dims: dict
        Dictionary where keys are names of different types of bar dimensions and
        values are lists of those dimensions for each word type. See `get_bar_dims`
        for details
    bar_colors: dict
        Dictionary where keys are names of different types of bar colors and
        values are lists of those colors for each word type. See `get_bar_colors`
        for details
    plot_params: dict
        Dictionary of plotting parameters. Here, `alpha_fade`, `bar_width`,
        `detailed`, and `bar_linewidth` are used
    """
    # Set plotting params
    bar_count = min(top_n, len(bar_dims["total_heights"]))
    ys = range(top_n - bar_count + 1, top_n + 1)
    alpha = plot_params["alpha_fade"]
    width = plot_params["bar_width"]
    linewidth = plot_params["bar_linewidth"]
    edgecolor = ["black"] * bar_count  # hack b/c matplotlib has a bug
    if plot_params["detailed"]:
        # Plot the p_diff and s_diff solid contributions
        ax.barh(
            ys,
            bar_dims["p_solid_heights"],
            width,
            align="center",
            zorder=10,
            color=bar_colors["p"],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.barh(
            ys,
            bar_dims["s_solid_heights"],
            width,
            left=bar_dims["s_solid_bases"],
            align="center",
            zorder=10,
            color=bar_colors["s"],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        # Plot the p_diff and s_diff faded counteractions
        ax.barh(
            ys,
            bar_dims["p_fade_heights"],
            width,
            left=bar_dims["p_fade_bases"],
            align="center",
            zorder=10,
            color=bar_colors["p"],
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.barh(
            ys,
            bar_dims["s_fade_heights"],
            width,
            left=bar_dims["s_fade_bases"],
            align="center",
            zorder=10,
            color=bar_colors["s"],
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
    else:
        # Plot the total contributions
        ax.barh(
            ys,
            bar_dims["total_heights"],
            width,
            align="center",
            zorder=10,
            color=bar_colors["total"],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    return ax


def get_bar_order(plot_params):
    """
    Gets which cumulative bars to show at the top of the graph given what level
    of detail is being specified

    Parameters
    ----------
    plot_params: dict
        Dictionary of plotting parameters. Here, `all_pos_contributions`,
        `detailed`, `show_score_diffs`, and `show_total` are used

    Returns
    -------
    List of strs indicating which cumulative bars to show
    """
    if plot_params["detailed"]:
        if plot_params["show_score_diffs"]:
            bar_order = [
                "neg_s",
                "pos_s",
                "neg_s_neg_p",
                "neg_s_pos_p",
                "pos_s_neg_p",
                "pos_s_pos_p",
            ]
        else:
            bar_order = ["neg_s_neg_p", "neg_s_pos_p", "pos_s_neg_p", "pos_s_pos_p"]
    else:
        if not plot_params["all_pos_contributions"]:
            bar_order = ["neg_total", "pos_total"]
        else:
            bar_order = ["all_pos_pos", "all_pos_neg"]

    if plot_params["show_total"]:
        bar_order = ["total"] + bar_order

    return bar_order


def plot_total_contribution_sums(
    ax, total_comp_sums, bar_order, top_n, bar_dims, plot_params
):
    """
    Plots the cumulative contribution bars at the top of the shift graph

    Parameters
    ----------
    ax: Matplotlib ax
        Current ax of the shift graph
    total_comp_sums: dict
        Dictionary with six keys, one for each of the different component
        contributions, where values are floats indicating the total contribution.
        See `get_shift_component_sums` for details
    bar_order: list of strs
        List of the names of which bars to show at the top of the shift graph.
        See `get_bar_order` for more detail
    top_n: int
        The number of types being plotted on the shift graph
    bar_dims: dict
        Dictionary where keys are names of different types of bar dimensions and
        values are lists of those dimensions for each word type. See `get_bar_dims`
        for details
    plot_params: dict
        Dictionary of plotting parameters. Here, `all_pos_contributions`,
        `show_total`, `score_colors`, `bar_width`, and `bar_linewidth` are used
    """
    # Get contribution bars
    comp_bar_heights = []
    for b in bar_order:
        if b == "total":
            h = 0
        elif b == "neg_total":
            h = (
                total_comp_sums["neg_s"]
                + total_comp_sums["neg_s_pos_p"]
                + total_comp_sums["pos_s_neg_p"]
            )
        elif b == "pos_total":
            h = (
                total_comp_sums["pos_s"]
                + total_comp_sums["neg_s_neg_p"]
                + total_comp_sums["pos_s_pos_p"]
            )
        elif b == "all_pos_pos":
            a = np.array(bar_dims["total_heights"])
            h = np.sum(a[a > 0])
        elif b == "all_pos_neg":
            a = np.array(bar_dims["total_heights"])
            h = np.sum(a[a < 0])
        else:
            h = total_comp_sums[b]

        comp_bar_heights.append(h)

    if "total" in bar_order:
        total_index = bar_order.index("total")
        total = sum(comp_bar_heights)
        comp_bar_heights[total_index] = total
    # Rescale bars
    if not plot_params["all_pos_contributions"]:
        max_bar_height = np.max(np.abs(bar_dims["label_heights"]))
    else:
        max_bar_height = np.max(np.abs(bar_dims["total_heights"]))
    comp_scaling = max_bar_height / np.max(np.abs(comp_bar_heights))
    comp_bar_heights = [comp_scaling * h for h in comp_bar_heights]

    # Get bar ys
    if plot_params["show_total"]:
        min_y = top_n + 3.5
        ys = [top_n + 2]
    else:
        min_y = top_n + 2
        ys = []
    for n_h in range(int(len(comp_bar_heights) / 2)):
        y = min_y + (1.5 * n_h)
        ys += [y, y]
    # Get other plotting params
    comp_colors = [plot_params["score_colors"][b] for b in bar_order]
    width = plot_params["bar_width"]
    linewidth = plot_params["bar_linewidth"]
    edgecolor = ["black"] * len(comp_bar_heights)
    # Plot total contribution bars
    ax.barh(
        ys,
        comp_bar_heights,
        width,
        align="center",
        color=comp_colors,
        linewidth=linewidth,
        edgecolor=edgecolor,
    )

    return ax, comp_bar_heights, bar_order


def get_bar_type_space(ax, plot_params):
    """
    Gets the amount of space to place in between the ends of bars and labels
    """
    # Estimate bar_type_space as a fraction of largest xlim
    x_width = 2 * abs(max(ax.get_xlim(), key=lambda x: abs(x)))
    bar_type_space = plot_params["bar_type_space_scaling"] * x_width
    return bar_type_space


def set_bar_labels(
    ax, top_n, type_labels, full_bar_heights, comp_bar_heights, plot_params
):
    """
    Sets the labels on the end of each type's contribution bar

    Parameters
    ----------
    ax: Matplotlib ax
        Current ax of the shift graph
    top_n: int
        The number of types being plotted on the shift graph
    type_labels: list of strs
        Sorted list of labels to plot on the shift graph
    full_bar_heights: list of floats
        List of heights of where to place the type contribution labels
    comp_bar_heights: list of floats
        List of heights of where to place the cumulative contribution labels
    plot_params: dict
        Dictionary of plotting parameters. Here, `show_total`, `label_fontsize`,
        and `bar_type_space_scaling`
    """
    # Put together all bar heights
    n = len(full_bar_heights)
    all_bar_ends = full_bar_heights + comp_bar_heights
    # Estimate bar_type_space as a fraction of largest xlim
    bar_type_space = get_bar_type_space(ax, plot_params)
    # Get heights of all bars
    if plot_params["show_total"]:
        min_y = top_n + 3.5
        top_heights = [top_n + 2]
    else:
        min_y = top_n + 2
        top_heights = []
    for n_h in range(int(len(comp_bar_heights) / 2)):
        y = min_y + (1.5 * n_h)
        top_heights += [y, y]
    bar_heights = list(range(top_n - n + 1, top_n + 1)) + top_heights
    # Set all bar labels
    text_objs = []
    fontsize = plot_params["label_fontsize"]
    for bar_n, width in enumerate(all_bar_ends):
        height = bar_heights[bar_n]
        if width < 0:
            ha = "right"
            space = -1 * bar_type_space
        else:
            ha = "left"
            space = bar_type_space
        t = ax.text(
            width + space,
            height,
            type_labels[bar_n],
            ha=ha,
            va="center",
            fontsize=fontsize,
            zorder=5,
        )
        text_objs.append(t)
    # Adjust axes for labels
    ax = adjust_axes_for_labels(
        ax, full_bar_heights, comp_bar_heights, text_objs, bar_type_space, plot_params
    )
    return ax


def adjust_axes_for_labels(
    ax, bar_ends, comp_bars, text_objs, bar_type_space, plot_params
):
    """
    Attempts to readjusts the axes to account for newly plotted labels

    Parameters
    ----------
    ax: Matplotlib ax
        Current ax of the shift graph
    bar_ends: list of floats
        List of heights of where to place the type contribution labels
    comp_bars: list of floats
        List of heights of where to place the cumulative contribution labels
    text_objs: list of Matplotlib text objects
        List of text after being plotted on the ax
    bar_type_space: float
        How much space to put between bar ends and labels
    plot_parms: dict
        Dictionary of plotting parameters. Here, `width_scaling` is used
    """
    # Get the max length
    lengths = []
    for bar_n, bar_end in enumerate(bar_ends):
        bar_length = bar_end
        bbox = text_objs[bar_n].get_window_extent(
            renderer=ax.figure.canvas.get_renderer()
        )
        bbox = ax.transData.inverted().transform(bbox)
        text_length = abs(bbox[0][0] - bbox[1][0])
        if bar_length > 0:
            lengths.append(bar_length + text_length + bar_type_space)
        else:
            lengths.append(bar_length - text_length - bar_type_space)
    # Add the top component bars to the lengths to check
    comp_bars = [abs(b) for b in comp_bars]
    lengths += comp_bars
    # Get max length
    width_scaling = plot_params["width_scaling"]
    max_length = width_scaling * abs(
        sorted(lengths, key=lambda x: abs(x), reverse=True)[0]
    )
    # Symmetrize the axis around that max length
    ax.set_xlim((-1 * max_length, max_length))

    return ax


def set_ticks(ax, top_n, plot_params):
    """
    Sets ticks and tick labels of the shift graph

    Parameters
    ----------
    ax: Matplotlib ax
        Current ax of the shift graph
    top_n: int
        The number of types being plotted on the shift graph
    plot_parms: dict
        Dictionary of plotting parameters. Here, `all_pos_contributions`,
        `tick_format`, `xtick_fontsize`, `ytick_fontsize`, `remove_xticks`,
        and `remove_yticks` are used
    """
    tick_format = plot_params["tick_format"]

    # Make xticks larger
    if not plot_params["all_pos_contributions"]:
        x_ticks = [tick_format.format(t) for t in ax.get_xticks()]
    else:
        x_ticks = [tick_format.format(abs(t)) for t in ax.get_xticks()]
    ax.set_xticklabels(x_ticks, fontsize=plot_params["xtick_fontsize"])
    # Flip y-axis tick labels and make sure every 5th tick is labeled
    y_ticks = list(range(1, top_n, plot_params["every_nth_ytick"])) + [top_n]
    y_tick_label_pos = (list(range(top_n, 1, -plot_params["every_nth_ytick"])) + ["1"])
    y_tick_labels = [str(n) for n in y_tick_label_pos]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=plot_params["ytick_fontsize"])

    # Remove all x or y axis ticks
    if plot_params["remove_xticks"]:
        remove_xaxis_ticks(ax)
    if plot_params["remove_yticks"]:
        remove_yaxis_ticks(ax)

    return ax


def set_spines(ax, plot_params):
    """
    Sets spines of the shift graph to be invisible if chosen by the user

    Parameters
    ----------
    ax: Matplotlib ax
        Current ax of the shift graph
    plot_parms: dict
        Dictionary of plotting parameters. Here `invisible_spines` is used
    """
    spines = plot_params["invisible_spines"]
    if spines:
        for spine in spines:
            if spine in {"left", "right", "top", "bottom"}:
                ax.spines[spine].set_visible(False)
            else:
                print("invalid spine argument")
    return ax


def remove_yaxis_ticks(ax, major=True, minor=True):
    """
    Removes all y-axis ticks on the shift graph
    """
    if major:
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    if minor:
        for tic in ax.yaxis.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)


def remove_xaxis_ticks(ax, major=True, minor=True):
    """
    Removes all x-axis ticks on the shift graph
    """
    if major:
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
    if minor:
        for tic in ax.xaxis.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)


def get_cumulative_inset(f, type2shift_score, top_n, normalization, plot_params):
    """
    Plots the cumulative contribution inset on the shift graph

    Parameters
    ----------
    f: Matpotlib figure
        Current figure of the shift graph
    type2shift_score: dict
        Keys are types and values are their total shift score
    top_n: int
        The number of types being plotted on the shift graph
    normalization: str
        The type of normalization being used on the shift scores, either
        'variation' (sum of abs values of scores) or 'trajectory' (sum of scores)
    plot_params: dict
        Dictionary of plotting parameters. Here, `pos_cumulative_inset`,
        `cumulative_xlabel`, `cumulative_ylabel`, `cumulative_xticks`,
        `cumulative_xticklabels`, `cumulative_yticks`, `cumulative_yticklabels`
        are used
    """
    # Get plotting params
    inset_pos = plot_params["pos_cumulative_inset"]
    # Get cumulative scores
    if normalization == "variation":
        scores = sorted(
            [100 * np.abs(s) for s in type2shift_score.values()],
            key=lambda x: abs(x),
            reverse=True,
        )
        if plot_params["cumulative_xlabel"] is None:
            plot_params["cumulative_xlabel"] = "$\sum | \delta \Phi_{\\tau} |$"
    else:
        scores = sorted(
            [100 * s for s in type2shift_score.values()],
            key=lambda x: abs(x),
            reverse=True,
        )
        if plot_params["cumulative_xlabel"] is None:
            plot_params["cumulative_xlabel"] = "$\sum \delta \Phi_{\\tau}$"
    cum_scores = np.cumsum(scores)
    # Plot cumulative difference
    left, bottom, width, height = inset_pos
    in_ax = f.add_axes([left, bottom, width, height])
    in_ax.semilogy(
        cum_scores,
        range(1, len(cum_scores) + 1),
        "-",
        color="black",
        linewidth=0.5,
        markersize=1.2,
    )
    # Remove extra space around line plot
    in_ax.set_xlim((min(cum_scores), max(cum_scores)))
    in_ax.set_ylim((1, len(cum_scores) + 1))
    in_ax.margins(x=0, y=0)
    # Reverse the y-axis
    y_min, y_max = in_ax.get_ylim()
    in_ax.set_ylim((y_max, y_min))
    # Set xticks
    # TODO: these defaults are unappealing if score goes way past 100 or -100
    total_score = cum_scores[-1]
    if np.sign(total_score) == 1:
        if plot_params["cumulative_xticks"] is None:
            plot_params["cumulative_xticks"] = [0, 25, 50, 75, 100]
        if plot_params["cumulative_xticklabels"] is None:
            plot_params["cumulative_xticklabels"] = ["0", "", "50", "", "100"]
    else:
        if plot_params["cumulative_xticks"] is None:
            plot_params["cumulative_xticks"] = [-100, -75, -50, -25, 0]
        if plot_params["cumulative_xticklabels"] is None:
            plot_params["cumulative_xticklabels"] = ["-100", "", "-50", "", "0"]
    in_ax.set_xticks(plot_params["cumulative_xticks"])
    in_ax.set_xticklabels(plot_params["cumulative_xticklabels"], fontsize=11)
    # Make tick labels smaller
    for tick in in_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    # Plot top_n line
    x_min, x_max = in_ax.get_xlim()
    in_ax.hlines(top_n, x_min, x_max, linestyle="-", color="black", linewidth=0.5)
    # Set labels
    in_ax.set_xlabel(plot_params["cumulative_xlabel"], fontsize=12)
    in_ax.set_ylabel(plot_params["cumulative_ylabel"], fontsize=12)
    # Make background transparent
    in_ax.patch.set_alpha(0)

    return f


def get_text_size_inset(f, type2freq_1, type2freq_2, plot_params):
    """
    Plots the relative text size inset on the shift graph

    Parameters
    ----------
    f: Matpotlib figure
        Current figure of the shift graph
    type2freq_1, type2freq_2: dict
        Keys are types, values are their frequencies
    plot_params: dict
        Dictionary of plotting parameters. Here, pos_text_size_inset` and
        `pos_text_size_inset` are used
    """
    # Get plotting params
    system_names = plot_params["system_names"]
    inset_pos = plot_params["pos_text_size_inset"]
    # Get size of each text
    n1 = sum(type2freq_1.values())
    n2 = sum(type2freq_2.values())
    # Normalize text sizes
    n = max(n1, n2)
    n1 = n1 / n
    n2 = n2 / n
    # Plot text size inset
    left, bottom, width, height = inset_pos
    in_ax = f.add_axes([left, bottom, width, height])
    in_ax.barh(
        [0.6, 0.4],
        [n1, n2],
        0.1,
        color="#707070",
        linewidth=0.5,
        edgecolor=["black"] * 2,
        tick_label=system_names,
    )
    # Rescale to make the bars appear to be more thin
    in_ax.set_ylim((0, 1))
    # Set title and label properties
    in_ax.text(0.5, 0.75, "Text Size:", horizontalalignment="center", fontsize=14)
    for tick in in_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    in_ax.tick_params(axis="y", length=0)
    # Turn off axes and make transparent
    for side in ["left", "right", "top", "bottom"]:
        in_ax.spines[side].set_visible(False)
    in_ax.get_xaxis().set_visible(False)
    in_ax.set_alpha(0)

    return f
