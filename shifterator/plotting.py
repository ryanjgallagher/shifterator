def get_plotting_params(plotting_params):
    if 'width' not in plotting_params:
        plotting_params['width'] = 7
    if 'height' not in plotting_params:
        plotting_params['height'] = 15
    if 'bar_width' not in plotting_params:
        plotting_params['bar_width'] = 0.8
    if 'bar_linewidth' not in plotting_params:
        plotting_params['bar_linewidth'] = 0.25
    if 'score_colors' not in plotting_params:
        plotting_params['score_colors'] = ('#ffff80', '#FDFFD2', '#2f7cce',
                                           '#C4CAFC', '#9E75B7', '#FECC5D')
    if 'alpha_fade' not in plotting_params:
        plotting_params['alpha_fade'] = 0.35
    if 'symbols' not in plotting_params:
        plotting_params['symbols'] = [r'$\Sigma$', u'\u25BD', u'\u25B3',
                                      u'-\u2193', u'-\u2191', u'+\u2193',
                                      u'+\u2191']
    if 'width_scaling' not in plotting_params:
        plotting_params['width_scaling'] = 1.2
    if 'bar_type_space_scaling' not in plotting_params:
        plotting_params['bar_type_space_scaling'] = 0.015
    if 'pos_cumulative_inset' not in plotting_params:
        plotting_params['pos_cumulative_inset'] = [0.19, 0.12, 0.175, 0.175]
    if 'pos_text_size_inset' not in plotting_params:
        plotting_params['pos_text_size_inset'] = [0.81, 0.12, 0.08, 0.08]
    if 'xlabel' not in plotting_params:
        plotting_params['xlabel'] = r'Per type average score shift $\delta s_{avg,r}$ (%)'
    if 'ylabel' not in plotting_params:
        plotting_params['ylabel'] = r'Type rank $r$'
    if 'xlabel_fontsize' not in plotting_params:
        plotting_params['xlabel_fontsize'] = 20
    if 'ylabel_fontsize' not in plotting_params:
        plotting_params['ylabel_fontsize'] = 20
    if 'title_fontsize' not in plotting_params:
        plotting_params['title_fontsize'] = 18
    if 'label_fontsize' not in plotting_params:
        plotting_params['label_fontsize'] = 13
    if 'xtick_fontsize' not in plotting_params:
        plotting_params['xtick_fontsize'] = 14
    if 'ytick_fontsize' not in plotting_params:
        plotting_params['ytick_fontsize'] = 14
    if 'system_names' not in plotting_params:
        plotting_params['system_names']=['Sys. 1', 'Sys. 2']
    if 'serif' not in plotting_params:
        plotting_params['serif'] = False
    if 'tight' not in plotting_params:
        plotting_params['tight'] = True
    if 'y_margin' not in plotting_params:
        plotting_params['y_margin'] = 0.005

    return plotting_params

def set_serif():
    rcParams['font.family'] = 'serif'
    rcParams['mathtext.fontset'] = 'dejavuserif'

def get_bar_heights(type_scores, normalizer):
    """
    tuple: (bar 1 height, bar 2 bottom, bar 2 height)
    """
    # TODO: return a dictionary to save on text?
    heights_comp1 = []
    heights_comp2 = []
    heights_alpha = []
    heights_subtract = []
    bottoms = []
    bottoms_alpha = []
    bar_ends = []
    for (_,p_diff,s_diff,p_avg,s_ref_diff,_) in type_scores:
        comp1 = 100*p_diff*s_ref_diff/normalizer
        comp2 = 100*p_avg*s_diff/normalizer
        # Determine if direction of comp bars are congruent
        if np.sign(s_ref_diff*p_diff)*np.sign(s_diff) == 1:
            heights_comp1.append(comp1)
            heights_comp2.append(comp2)
            heights_alpha.append(0)
            heights_subtract.append(0)
            bar_ends.append(comp1+comp2)
            bottoms.append(comp1)
            bottoms_alpha.append(0)
        else:
            total_comp = comp1+comp2
            bottoms.append(0)
            bottoms_alpha.append(total_comp)
            if abs(comp1) > abs(comp2):
                heights_comp1.append(total_comp)
                heights_comp2.append(0)
                heights_subtract.append(comp2)
                heights_alpha.append(comp1-total_comp)
                bar_ends.append(comp1)
            else:
                heights_comp1.append(0)
                heights_comp2.append(total_comp)
                heights_subtract.append(comp1)
                heights_alpha.append(comp2-total_comp)
                bar_ends.append(comp2)
    return (heights_comp1, heights_comp2, heights_alpha, heights_subtract,
            bottoms, bottoms_alpha, bar_ends)

def get_bar_colors(type_scores, bar_heights, plotting_params):
    comp_bar_colors = get_comp_bar_colors(type_scores, plotting_params)
    fade_bar_colors = get_fade_bar_colors(bar_heights, comp_bar_colors)
    return comp_bar_colors+fade_bar_colors

def get_comp_bar_colors(type_scores, plotting_params):
    """

    """
    score_colors = plotting_params['score_colors']
    bar_colors_comp1 = []
    bar_colors_comp2 = []
    for (_,p_diff,s_diff,p_avg,s_ref_diff,_) in type_scores:
        # Get first p_diff/s_ref_diff comp colors
        if s_ref_diff > 0:
            if p_diff > 0:
                bar_colors_comp1.append(score_colors[0])
            else:
                bar_colors_comp1.append(score_colors[1])
        else:
            if p_diff > 0:
                bar_colors_comp1.append(score_colors[2])
            else:
                bar_colors_comp1.append(score_colors[3])
        # Get s_diff comp colors
        if s_diff > 0:
            bar_colors_comp2.append(score_colors[4])
        else:
            bar_colors_comp2.append(score_colors[5])
    return [bar_colors_comp1, bar_colors_comp2]

def get_bar_fade_colors(bar_heights, comp_bar_colors):
    """
    """
    # Unpack bar heights
    height_c1, height_c2, heights_alpha,_,_,_,_,_ = bar_heights
    # Get colors for how contributions cancel out
    bar_colors_alpha = []
    bar_colors_subtract = []
    for n in range(len(heights_c1)):
        # Check if heights_alpha takes away from heights_c1 or not
        if abs(heights_alpha[n]+heights_c1[n]) > abs(heights_c2[n]):
            bar_colors_alpha.append(comp_bar_colors[0][n])
            bar_colors_subtract.append(comp_bar_colors[1][n])
        else:
            bar_colors_alpha.append(comp_bar_colors[1][n])
            bar_colors_subtract.append(comp_bar_colors[0][n])
    return [bar_colors_alpha, bar_colors_subtract]

def plot_contributions(ax, bar_heights, bar_colors, plotting_params):
    """
    """
    # Unpack bar heights and colors
    heights_c1,heights_c2,heights_alpha,heights_subtract,bms,bms_alpha,ends=bar_heights
    colors_c1,colors_c2,colors_alpha,colors_fade = bar_colors
    # Set plotting params
    ys = range(1,len(bar_heights)+1)
    alpha = plotting_params['alpha']
    width = plotting_params['bar_width']
    linewidth = plotting_params['bar_linewidth']
    edgecolor = ['black']*top_n # hack b/c matplotlib has a bug
    # Plot main contributions
    ax.barh(ys, heights_c1, width, align='center', color=colors_c1,
            linewidth=linewidth, edgecolor=edgecolor, zorder=10)
    ax.barh(ys, heights_c2, width, left=bms, align='center', color=colors_c2,
            linewidth=linewidth, edgecolor=edgecolor, zorder=10,)
    # Plot the counteracting components as faded bar charts
    ax.barh(ys, heights_alpha, 0.8, left=bms_alpha, align='center',
            color=colors_alpha, alpha=alpha, linewidth=0.25,
            edgecolor=edgecolor, zorder=10)
    ax.barh(ys, heights_subtract, 0.8, left=bms, align='center',
            color=colors_subtract, alpha=0.35, linewidth=0.25,
            edgecolor=edgecolor, zorder=10)
    return ax

def plot_total_contribution_sums(ax, total_comp_sums, bar_ends, plotting_params):
    # Get overall sum and rescale according to contribution bars
    # +freq+score, +freq-score, -freq+score, -freq-score, +s_diff, -s_diff
    comp_bars = [sum(total_comp_sums)] + list(reversed(total_comp_sums))
    comp_scaling = abs(bar_ends[np.argmax(bar_ends)]\
                   /abs(comp_bars[np.argmax(comp_bars)]))
    comp_bars = [comp_scaling*s for s in comp_bars]
    # Set plotting params
    n = len(bar_ends)
    ys = [n+2, n+3.5, n+3.5, n+5, n+5, n+6.5, n+6.5]
    comp_colors = ['#707070'] + list(reversed(plotting_params['score_colors']))
    width = plotting_params['bar_width']
    linewidth = plotting_params['linewidth']
    edgecolor = ['black']*len(comp_bars)
    # Plot total contribution bars
    ax.barh(ys, comp_bars, width, align='center', color=comp_colors
            linewidth=linewidth, edgecolor=edgecolor)
    return ax, comp_bars

def get_bar_type_space(ax, plotting_params):
    # Estimate bar_type_space as a fraction of largest xlim
    x_width = 2*abs(max(ax.get_xlim(), key=lambda x: abs(x)))
    bar_type_space = bar_type_space_scaling*x_width
    return bar_type_space

def set_bar_labels(f, ax, type_labels, bar_ends, comp_bars, plotting_params):
    # Put together all bar heights
    n = len(bar_ends)
    all_bar_ends = bar_ends + comp_bars
    # Estimate bar_type_space as a fraction of largest xlim
    bar_type_space = get_bar_type_space(ax, plotting_params)
    # Get heights of all bars
    top_heights = [n+2, n+3.5, n+3.5, n+5, n+5, n+6.5, n+6.5]
    bar_heights = list(range(1, n+1)) + top_heights
    # Set all bar labels
    text_objs = []
    fontsize = plotting_params['label_fontsize']
    for bar_n,width in enumerate(range(len(all_bar_ends))):
        height = bar_heights[bar_n]
        if width < 0:
            ha='right'
            space = -1*bar_type_space
        else:
            ha='left'
            space = bar_type_space
        t = ax.text(width+space, height, type_labels[bar_n],
                    ha=ha, va='center', fontsize=fontsize, zorder=5)
        text_objs.append(t)
    # Adjust axes for labels
    ax = adjust_axes_for_labels(f, ax, bar_ends, comp_bars, text_objs,
                                bar_type_space, plotting_params)
    return ax

def adjust_axes_for_labels(f, ax, bar_ends, comp_bars, text_obj, bar_type_space,
                           plotting_params):
    # Get the max length
    lengths = []
    for bar_n,bar_end in enumerate(bar_ends):
        bar_length = bar_end
        bbox = text_objs[bar_n].get_window_extent(renderer=f.canvas.get_renderer())
        bbox = ax.transData.inverted().transform(bbox)
        text_length = abs(bbox[0][0]-bbox[1][0])
        if bar_length > 0:
            lengths.append(bar_length+text_length+bar_type_space)
        else:
            lengths.append(bar_length-text_length-bar_type_space)
    # Add the top component bars to the lengths to check
    comp_bars = [abs(b) for b in comp_bars]
    lengths += comp_bars
    # Get max length
    width_scaling = plotting_params['width_scaling']
    max_length = width_scaling*abs(sorted(lengths, key=lambda x: abs(x),
                                          reverse=True)[0])
    # Symmetrize the axis around that max length
    ax.set_xlim((-1*max_length, max_length))

    return ax

def set_ticks(ax, top_n, plotting_params):
    # Make xticks larger
    x_ticks = ['{:.1f}'.format(t) for t in ax.get_xticks()]
    ax.set_xticklabels(x_ticks, fontsize=plotting_params['xtick_fontsize'])
    # Flip y-axis tick labels and make sure every 5th tick is labeled
    y_ticks = list(range(1,top_n,5))+[top_n]
    y_tick_labels = [str(n) for n in (list(range(top_n,1,-5))+['1'])]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=plotting_params['ytick_fontsize'])

    return ax

def get_cumulative_inset(f, type2shift_score, top_n, plotting_params):
    # Get plotting params
    inset_pos = plotting_params['pos_cumulative_inset']
    # Get cumulative scores
    scores = sorted([100*s for s in type2shift_score.values()],
                     key=lambda x:abs(x), reverse=True)
    cum_scores = np.cumsum(scores)
    # Plot cumulative difference
    left, bottom, width, height = inset_pos
    in_ax = f.add_axes([left, bottom, width, height])
    in_ax.semilogy(cum_scores, range(1,len(cum_scores)+1), '-o', color='black',
                   linewidth=0.5, markersize=1.2)
    # Remove extra space around line plot
    in_ax.set_xlim((min(cum_scores),max(cum_scores)))
    in_ax.set_ylim((1, len(cum_scores)+1))
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

def get_text_size_inset(f, type2freq_1, type2freq_2, plotting_params):
    # Get plotting params
    system_names = plotting_params['system_names']
    inset_pos = plotting_params['pos_text_size_inset']
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
    Note: this annotation might only make sense for relative shifts
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
