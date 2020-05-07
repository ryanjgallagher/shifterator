# -*- coding: utf-8 -*-
"""
plotting.py

Author: Ryan J. Gallagher, Network Science Institute, Northeastern University

Requires: Python 3

TODO:
- Make func for missing type labels and add param for setting borrowing symbol
- Add params for explicitly setting fonts
- Add doc strings
"""
import numpy as np
from matplotlib import rcParams

def get_plot_params(plot_params, show_score_diffs):
    if 'detailed' not in plot_params:
        plot_params['detailed'] = True
    if 'show_total' not in plot_params:
        plot_params['show_total'] = True
    if 'show_score_diffs' not in plot_params:
        plot_params['show_score_diffs'] = show_score_diffs
    if 'all_pos_contributions' not in plot_params:
        plot_params['all_pos_contributions'] = False
    if 'width' not in plot_params:
        plot_params['width'] = 7
    if 'height' not in plot_params:
        plot_params['height'] = 15
    if 'bar_width' not in plot_params:
        plot_params['bar_width'] = 0.8
    if 'bar_linewidth' not in plot_params:
        plot_params['bar_linewidth'] = 0.25
    if 'score_colors' not in plot_params:
        plot_params['score_colors'] = {'pos_s_pos_p': '#FFFF80',
                                       'pos_s_neg_p': '#FDFFD2',
                                       'neg_s_pos_p': '#2F7CCE',
                                       'neg_s_neg_p': '#C4CAFC',
                                       'pos_s': '#FECC5D',
                                       'neg_s': '#9E75B7',
                                       'pos_total': '#FFFF80',
                                       'neg_total': '#C4CAFC',
                                       'all_pos_pos': '#FFFF80',
                                       'all_pos_neg': '#C4CAFC',
                                       'total': '#707070'}
    if 'alpha_fade' not in plot_params:
        plot_params['alpha_fade'] = 0.35
    if 'symbols' not in plot_params:
        plot_params['symbols'] = {'pos_s_pos_p': u'+\u2191',
                                  'pos_s_neg_p': u'+\u2193',
                                  'neg_s_pos_p': u'-\u2191',
                                  'neg_s_neg_p': u'-\u2193',
                                  'pos_s': u'\u25B3',
                                  'neg_s': u'\u25BD',
                                  'pos_total': '',
                                  'neg_total': '',
                                  'all_pos_pos': 'Sys. 1',
                                  'all_pos_neg': 'Sys. 2',
                                  'total': r'$\Sigma$'}
    if 'missing_symbol' not in plot_params:
        plot_params['missing_symbol'] = '*'
    if 'width_scaling' not in plot_params:
        plot_params['width_scaling'] = 1.2
    if 'bar_type_space_scaling' not in plot_params:
        plot_params['bar_type_space_scaling'] = 0.015
    if 'pos_cumulative_inset' not in plot_params:
        plot_params['pos_cumulative_inset'] = [0.19, 0.12, 0.175, 0.175]
    if 'pos_text_size_inset' not in plot_params:
        plot_params['pos_text_size_inset'] = [0.81, 0.12, 0.08, 0.08]
    if 'xlabel' not in plot_params:
        plot_params['xlabel'] = r'Per type average score shift $\delta s_{avg,r}$ (%)'
    if 'ylabel' not in plot_params:
        plot_params['ylabel'] = r'Type rank $r$'
    if 'xlabel_fontsize' not in plot_params:
        plot_params['xlabel_fontsize'] = 20
    if 'ylabel_fontsize' not in plot_params:
        plot_params['ylabel_fontsize'] = 20
    if 'title_fontsize' not in plot_params:
        plot_params['title_fontsize'] = 18
    if 'label_fontsize' not in plot_params:
        plot_params['label_fontsize'] = 13
    if 'xtick_fontsize' not in plot_params:
        plot_params['xtick_fontsize'] = 14
    if 'ytick_fontsize' not in plot_params:
        plot_params['ytick_fontsize'] = 14
    if 'system_names' not in plot_params:
        plot_params['system_names']=['Sys. 1', 'Sys. 2']
    if 'serif' not in plot_params:
        plot_params['serif'] = False
    if 'tight' not in plot_params:
        plot_params['tight'] = True
    if 'dpi' not in plot_params:
        plot_params['dpi'] = 200
    if 'y_margin' not in plot_params:
        plot_params['y_margin'] = 0.005

    return plot_params

def set_serif():
    rcParams['font.family'] = 'serif'
    rcParams['mathtext.fontset'] = 'dejavuserif'

def get_bar_dims(type_scores, norm, plot_params):
    """

    """
    # 'p' for p_diff component, 's' for s_diff component
    # 'solid' for part of comp that is not alpha faded, 'faded' otherwise
    # 'base' for where bottom of bar is, 'height' for height from that base
    #     note, 'p_solid_base' is always 0
    # 'total' for total contribution for simple word shift graphs (base always 0)
    dims = {'p_solid_heights':[], 's_solid_bases':[], 's_solid_heights':[],
            'p_fade_heights':[], 'p_fade_bases':[], 's_fade_bases':[],
            's_fade_heights':[], 'total_heights':[], 'label_heights':[]}

    for (_,p_diff,s_diff,p_avg,s_ref_diff,_) in type_scores:
        c_p = 100 * p_diff * s_ref_diff / norm
        c_s = 100 * p_avg * s_diff / norm
        # This is for JSD to make bars face different directions based on p
        # p_diff is p_2 - p_1, so point to right if p_1 > p_2
        if not plot_params['all_pos_contributions'] or p_diff < 0:
            dims['total_heights'].append(c_p + c_s)
        else:
            dims['total_heights'].append(-1 * (c_p + c_s))
        # Determine if direction of comp bars are congruent
        if np.sign(s_ref_diff * p_diff) * np.sign(s_diff) == 1:
            dims['p_solid_heights'].append(c_p)
            dims['s_solid_bases'].append(c_p)
            dims['s_solid_heights'].append(c_s)
            dims['label_heights'].append(c_p + c_s)
            for d in ['p_fade_bases', 'p_fade_heights', 's_fade_bases', 's_fade_heights']:
                dims[d] .append(0)
        else:
            if abs(c_p) > abs(c_s):
                dims['p_solid_heights'].append(c_p + c_s)
                dims['p_fade_bases'].append(c_p + c_s)
                dims['p_fade_heights'].append(-1 * c_s)
                dims['s_fade_heights'].append(c_s)
                dims['label_heights'].append(c_p)
                for d in ['s_solid_bases', 's_solid_heights', 's_fade_bases']:
                    dims[d].append(0)
            else:
                dims['s_solid_heights'].append(c_s + c_p)
                dims['p_fade_heights'].append(c_p)
                dims['s_fade_bases'].append(c_s + c_p)
                dims['s_fade_heights'].append(-1 * c_p)
                dims['label_heights'].append(c_s)
                for d in ['p_solid_heights', 's_solid_bases', 'p_fade_bases']:
                    dims[d].append(0)

    return dims

def get_bar_colors(type_scores, plot_params):
    """

    """
    score_colors = plot_params['score_colors']
    bar_colors = {'p':[], 's':[], 'total':[]}
    for (_,p_diff,s_diff,p_avg,s_ref_diff,_) in type_scores:
        c_total = p_diff * s_ref_diff + p_avg * s_diff
        # Get total contribution colors
        if not plot_params['all_pos_contributions']:
            if c_total > 0:
                bar_colors['total'].append(score_colors['pos_total'])
            else:
                bar_colors['total'].append(score_colors['neg_total'])
        else:
            if p_diff < 0:
                bar_colors['total'].append(score_colors['all_pos_pos'])
            else:
                bar_colors['total'].append(score_colors['all_pos_neg'])
        # Get p_diff * s_ref_diff comp colors
        if s_ref_diff > 0:
            if p_diff > 0:
                bar_colors['p'].append(score_colors['pos_s_pos_p'])
            else:
                bar_colors['p'].append(score_colors['pos_s_neg_p'])
        else:
            if p_diff > 0:
                bar_colors['p'].append(score_colors['neg_s_pos_p'])
            else:
                bar_colors['p'].append(score_colors['neg_s_neg_p'])
        # Get s_diff comp colors
        if s_diff > 0:
            bar_colors['s'].append(score_colors['pos_s'])
        else:
            bar_colors['s'].append(score_colors['neg_s'])

    return bar_colors

def plot_contributions(ax, top_n, bar_dims, bar_colors, plot_params):
    """
    """
    # Set plotting params
    ys = range(1, top_n + 1)
    alpha = plot_params['alpha_fade']
    width = plot_params['bar_width']
    linewidth = plot_params['bar_linewidth']
    edgecolor = ['black'] * top_n # hack b/c matplotlib has a bug
    if plot_params['detailed']:
        # Plot the p_diff and s_diff solid contributions
        ax.barh(ys, bar_dims['p_solid_heights'], width, align='center', zorder=10,
                color=bar_colors['p'], edgecolor=edgecolor, linewidth=linewidth)
        ax.barh(ys, bar_dims['s_solid_heights'], width, left=bar_dims['s_solid_bases'],
                align='center', zorder=10, color=bar_colors['s'], edgecolor=edgecolor,
                linewidth=linewidth)
        # Plot the p_diff and s_diff faded counteractions
        ax.barh(ys, bar_dims['p_fade_heights'], width, left=bar_dims['p_fade_bases'],
                align='center', zorder=10, color=bar_colors['p'], edgecolor=edgecolor,
                alpha=alpha, linewidth=linewidth)
        ax.barh(ys, bar_dims['s_fade_heights'], width, left=bar_dims['s_fade_bases'],
                align='center', zorder=10, color=bar_colors['s'], edgecolor=edgecolor,
                alpha=alpha, linewidth=linewidth)
    else:
        # Plot the total contributions
        ax.barh(ys, bar_dims['total_heights'], width, align='center', zorder=10,
                color=bar_colors['total'], edgecolor=edgecolor, linewidth=linewidth)

    return ax

def get_bar_order(plot_params):
    if plot_params['detailed']:
        if plot_params['show_score_diffs']:
            bar_order = ['neg_s', 'pos_s', 'neg_s_neg_p', 'neg_s_pos_p',
                         'pos_s_neg_p', 'pos_s_pos_p']
        else:
            bar_order = ['neg_s_neg_p', 'neg_s_pos_p', 'pos_s_neg_p', 'pos_s_pos_p']
    else:
        if not plot_params['all_pos_contributions']:
            bar_order = ['neg_total', 'pos_total']
        else:
            bar_order = ['all_pos_pos', 'all_pos_neg']

    if plot_params['show_total']:
        bar_order = ['total'] + bar_order

    return bar_order

def plot_total_contribution_sums(ax, total_comp_sums, bar_order, top_n, bar_dims,
                                 plot_params):
    # Get contribution bars
    comp_bar_heights = []
    for b in bar_order:
        if b == 'total':
            h = 0
        elif b == 'neg_total':
            h = total_comp_sums['neg_s'] + total_comp_sums['neg_s_pos_p'] +\
                total_comp_sums['pos_s_neg_p']
        elif b == 'pos_total':
            h = total_comp_sums['pos_s'] + total_comp_sums['neg_s_neg_p'] +\
                total_comp_sums['pos_s_pos_p']
        elif b == 'all_pos_pos':
            a = np.array(bar_dims['total_heights'])
            h = np.sum(a[a > 0])
        elif b == 'all_pos_neg':
            a = np.array(bar_dims['total_heights'])
            h = np.sum(a[a < 0])
        else:
            h = total_comp_sums[b]

        comp_bar_heights.append(h)

    if 'total' in bar_order:
        total_index = bar_order.index('total')
        total = sum(comp_bar_heights)
        comp_bar_heights[total_index] = total
    # Rescacle bars
    if not plot_params['all_pos_contributions']:
        max_bar_height = np.max(np.abs(bar_dims['label_heights']))
    else:
        max_bar_height = np.max(np.abs(bar_dims['total_heights']))
    comp_scaling = max_bar_height / np.max(np.abs(comp_bar_heights))
    comp_bar_heights = [comp_scaling * h for h in comp_bar_heights]

    # Get bar ys
    if plot_params['show_total']:
        min_y = top_n + 3.5
        ys = [top_n + 2]
    else:
        min_y = top_n + 2
        ys = []
    for n_h in range(int(len(comp_bar_heights)/2)):
        y = min_y + (1.5 * n_h)
        ys += [y, y]
    # Get other plotting params
    comp_colors = [plot_params['score_colors'][b] for b in bar_order]
    width = plot_params['bar_width']
    linewidth = plot_params['bar_linewidth']
    edgecolor = ['black'] * len(comp_bar_heights)
    # Plot total contribution bars
    ax.barh(ys, comp_bar_heights, width, align='center', color=comp_colors,
            linewidth=linewidth, edgecolor=edgecolor)

    return ax, comp_bar_heights, bar_order

def get_bar_type_space(ax, plot_params):
    # Estimate bar_type_space as a fraction of largest xlim
    x_width = 2 * abs(max(ax.get_xlim(), key=lambda x: abs(x)))
    bar_type_space = plot_params['bar_type_space_scaling'] * x_width
    return bar_type_space

def set_bar_labels(f, ax, top_n, type_labels, full_bar_heights, comp_bar_heights,
                   plot_params):
    # Put together all bar heights
    n = len(full_bar_heights)
    all_bar_ends = full_bar_heights + comp_bar_heights
    # Estimate bar_type_space as a fraction of largest xlim
    bar_type_space = get_bar_type_space(ax, plot_params)
    # Get heights of all bars
    if plot_params['show_total']:
        min_y = top_n + 3.5
        top_heights = [top_n + 2]
    else:
        min_y = top_n + 2
        top_heights = []
    for n_h in range(int(len(comp_bar_heights)/2)):
        y = min_y + (1.5 * n_h)
        top_heights += [y, y]
    bar_heights = list(range(1, n + 1)) + top_heights
    # Set all bar labels
    text_objs = []
    fontsize = plot_params['label_fontsize']
    for bar_n,width in enumerate(all_bar_ends):
        height = bar_heights[bar_n]
        if width < 0:
            ha='right'
            space = -1 * bar_type_space
        else:
            ha='left'
            space = bar_type_space
        t = ax.text(width + space, height, type_labels[bar_n],
                    ha=ha, va='center', fontsize=fontsize, zorder=5)
        text_objs.append(t)
    # Adjust axes for labels
    ax = adjust_axes_for_labels(f, ax, full_bar_heights, comp_bar_heights,
                                text_objs, bar_type_space, plot_params)
    return ax

def adjust_axes_for_labels(f, ax, bar_ends, comp_bars, text_objs, bar_type_space,
                           plot_params):
    # Get the max length
    lengths = []
    for bar_n,bar_end in enumerate(bar_ends):
        bar_length = bar_end
        bbox = text_objs[bar_n].get_window_extent(renderer=f.canvas.get_renderer())
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
    width_scaling = plot_params['width_scaling']
    max_length = width_scaling*abs(sorted(lengths, key=lambda x: abs(x),
                                          reverse=True)[0])
    # Symmetrize the axis around that max length
    ax.set_xlim((-1 * max_length, max_length))

    return ax

def set_ticks(ax, top_n, plot_params):
    # Make xticks larger
    if not plot_params['all_pos_contributions']:
        x_ticks = ['{:.1f}'.format(t) for t in ax.get_xticks()]
    else:
        x_ticks = ['{:.1f}'.format(abs(t)) for t in ax.get_xticks()]
    ax.set_xticklabels(x_ticks, fontsize=plot_params['xtick_fontsize'])
    # Flip y-axis tick labels and make sure every 5th tick is labeled
    y_ticks = list(range(1,top_n,5))+[top_n]
    y_tick_labels = [str(n) for n in (list(range(top_n,1,-5))+['1'])]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=plot_params['ytick_fontsize'])

    return ax

def get_cumulative_inset(f, type2shift_score, top_n, plot_params):
    # Get plotting params
    inset_pos = plot_params['pos_cumulative_inset']
    # Get cumulative scores
    scores = sorted([100 * s for s in type2shift_score.values()],
                     key=lambda x:abs(x), reverse=True)
    cum_scores = np.cumsum(scores)
    # Plot cumulative difference
    left, bottom, width, height = inset_pos
    in_ax = f.add_axes([left, bottom, width, height])
    in_ax.semilogy(cum_scores, range(1, len(cum_scores) + 1), '-o', color='black',
                   linewidth=0.5, markersize=1.2)
    # Remove extra space around line plot
    in_ax.set_xlim((min(cum_scores),max(cum_scores)))
    in_ax.set_ylim((1, len(cum_scores) + 1))
    in_ax.margins(x=0, y=0)
    # Reverse the y-axis
    y_min,y_max = in_ax.get_ylim()
    in_ax.set_ylim((y_max, y_min))
    # Set x-axis limits
    total_score = cum_scores[-1]
    x_min,x_max = in_ax.get_xlim()
    if np.sign(total_score) == -1:
        in_ax.set_xlim((x_min, 0))
    else:
        in_ax.set_xlim((0, x_max))
    # Plot top_n line
    x_min,x_max = in_ax.get_xlim()
    in_ax.plot([x_min,x_max], [top_n,top_n], '-', color='black', linewidth=0.5)
    # Make tick labels smaller
    for ticks in [in_ax.xaxis.get_major_ticks(), in_ax.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(12)
    # Set labels
    in_ax.set_xlabel('$\sum^r \delta \Phi_{\\tau}(T^{(1)}, T^{(2)})$', fontsize=12)
    # Make background transparent
    in_ax.patch.set_alpha(0)

    return f

def get_text_size_inset(f, type2freq_1, type2freq_2, plot_params):
    # Get plotting params
    system_names = plot_params['system_names']
    inset_pos = plot_params['pos_text_size_inset']
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
    in_ax.barh([0.6, 0.4], [n1, n2], 0.1, color='#707070', linewidth=0.5,
               edgecolor=['black']*2, tick_label=system_names)
    # Rescale to make the bars appear to be more thin
    in_ax.set_ylim((0, 1))
    # Set title and label properties
    in_ax.text(0.5, 0.75, 'Text Size:', horizontalalignment='center', fontsize=14)
    for tick in in_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    in_ax.tick_params(axis='y', length=0)
    # Turn off axes and make transparent
    for side in ['left', 'right', 'top', 'bottom']:
        in_ax.spines[side].set_visible(False)
    in_ax.get_xaxis().set_visible(False)
    in_ax.set_alpha(0)

    return f

def get_guidance_annotations(ax, top_n, score_diff=None, annotation_text=None):
    """
    Note: this annotation only make sense for relative shifts
    """
    x_min,x_max = ax.get_xlim()
    y = np.floor(top_n / 2)+0.5 # depends on width=0.8 for bars
    ax.arrow(0, y, -0.985*x_max, 0, color='#D0D0D0', zorder=1, head_width=0.4,
             head_length=0.4, length_includes_head=True, head_starts_at_zero=False)
    ax.arrow(0, y, 0.985*x_max, 0, color='#D0D0D0', zorder=1, head_width=0.4,
             head_length=0.4, length_includes_head=True, head_starts_at_zero=False)
    left_text = 'Contributes to $T^{(comp)}$\nBeing Less Positive'
    right_text = 'Contributes to $T^{(comp)}$\nBeing More Positive'
    ax.text(-0.9*x_max, y, left_text, ha='left', va='bottom', fontsize=7,
            color='#808080', zorder=10)
    ax.text(0.9*x_max, y, right_text, ha='right', va='bottom', fontsize=7,
            color='#808080', zorder=10)
    return ax
