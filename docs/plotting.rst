Plotting Options
================

The ``get_shift_graph`` method accepts keyword arguments that are passed through
to the plotting system. These let you customize the appearance of shift graphs.

Basic Usage
-----------

Pass keyword arguments directly to ``get_shift_graph``:

.. code-block:: python

    shift.get_shift_graph(
        top_n=50,
        title="My Shift Graph",
        detailed=True,
        show_plot=True,
    )

All parameters below can be passed as keyword arguments to ``get_shift_graph``.


Layout & Sizing
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``width``
     - ``7``
     - Figure width in inches
   * - ``height``
     - ``15``
     - Figure height in inches
   * - ``dpi``
     - ``200``
     - Resolution for saved figures
   * - ``tight``
     - ``True``
     - Whether to apply ``tight_layout``
   * - ``y_margin``
     - ``0.005``
     - Vertical margin around bars
   * - ``bar_width``
     - ``0.8``
     - Width of each horizontal bar
   * - ``bar_linewidth``
     - ``0.25``
     - Line width of bar borders
   * - ``bar_type_space_scaling``
     - ``0.015``
     - Fraction of x-range used as space between bar ends and labels
   * - ``width_scaling``
     - ``1.2``
     - Scaling factor for x-axis width to accommodate labels


Display Options
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``detailed``
     - ``True``
     - If ``True``, shows the two components of each bar (frequency and score
       contributions). If ``False``, shows only the total contribution
   * - ``show_total``
     - ``True``
     - Whether to show the cumulative sum bar at the top
   * - ``all_pos_contributions``
     - ``False``
     - If ``True``, all bars face the same direction (used by JSD shifts).
       Colors indicate which system the type is more associated with
   * - ``label_total_with_system_names``
     - ``False``
     - If ``True``, the two cumulative bars at the top of a non-detailed plot
       are labeled with ``system_names`` instead of being unlabeled. Enabled by
       default for ``ProportionShift``, where negative/positive contributions
       correspond to types more prevalent in system 1/system 2. Unlike
       ``all_pos_contributions``, this does not change bar directions
   * - ``show_score_diffs``
     - (auto)
     - Whether score difference bars (△/▽) are shown. Automatically set to
       ``True`` when ``type2score_1 != type2score_2``
   * - ``system_names``
     - ``["Text 1", "Text 2"]``
     - Names of the two systems, used in labels and title
   * - ``missing_symbol``
     - ``"*"``
     - Symbol appended to type labels that adopted a score from the other system


Text & Fonts
------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``font_family``
     - ``None``
     - Font family for the plot. Set to a font with broad Unicode support
       (e.g. ``"Noto Sans CJK SC"`` for Chinese, ``"Noto Sans"`` for general
       Unicode) if your types contain non-Latin characters. When set, this
       takes precedence over the ``serif`` option
   * - ``serif``
     - ``False``
     - If ``True``, uses serif fonts (DejaVu Serif). Ignored if
       ``font_family`` is set
   * - ``label_fontsize``
     - ``13``
     - Font size for type labels on bars
   * - ``title_fontsize``
     - ``18``
     - Font size for the title
   * - ``xlabel_fontsize``
     - ``20``
     - Font size for the x-axis label
   * - ``ylabel_fontsize``
     - ``20``
     - Font size for the y-axis label
   * - ``xtick_fontsize``
     - ``14``
     - Font size for x-axis tick labels
   * - ``ytick_fontsize``
     - ``14``
     - Font size for y-axis tick labels

**Example: Chinese text**

.. code-block:: python

    shift.get_shift_graph(
        font_family="Noto Sans CJK SC",
        show_plot=True,
    )


Axis & Tick Options
-------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``xlabel``
     - ``r"Score shift $\delta \Phi_{\tau}$ (%)"``
     - Label for the x-axis
   * - ``ylabel``
     - ``"Rank"``
     - Label for the y-axis
   * - ``title``
     - (auto)
     - Title for the plot. If not set, automatically shows weighted average
       scores for both systems
   * - ``tick_format``
     - ``"{:.1f}"``
     - Format string for x-axis tick labels
   * - ``every_nth_ytick``
     - ``5``
     - Show a y-axis tick label every Nth rank
   * - ``remove_xticks``
     - ``False``
     - Remove all x-axis ticks
   * - ``remove_yticks``
     - ``False``
     - Remove all y-axis ticks
   * - ``invisible_spines``
     - ``[]``
     - List of spines to hide: ``"left"``, ``"right"``, ``"top"``, ``"bottom"``


Colors
------

Colors are controlled via the ``score_colors`` dictionary. Pass a dict to
override specific keys:

.. code-block:: python

    shift.get_shift_graph(
        score_colors={
            "pos_s_pos_p": "#FF6B6B",
            "neg_s_neg_p": "#4ECDC4",
        }
    )

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``pos_s_pos_p``
     - ``#FFFF80``
     - Positive score, increased frequency (↑+)
   * - ``pos_s_neg_p``
     - ``#FDFFD2``
     - Positive score, decreased frequency (↓+)
   * - ``neg_s_pos_p``
     - ``#2F7CCE``
     - Negative score, increased frequency (↑−)
   * - ``neg_s_neg_p``
     - ``#C4CAFC``
     - Negative score, decreased frequency (↓−)
   * - ``pos_s``
     - ``#FECC5D``
     - Score increased (△)
   * - ``neg_s``
     - ``#9E75B7``
     - Score decreased (▽)
   * - ``pos_total``
     - ``#FECC5D``
     - Positive total contribution (simple mode)
   * - ``neg_total``
     - ``#9E75B7``
     - Negative total contribution (simple mode)
   * - ``all_pos_pos``
     - ``#FECC5D``
     - More associated with system 2 (JSD all-positive mode)
   * - ``all_pos_neg``
     - ``#9E75B7``
     - More associated with system 1 (JSD all-positive mode)
   * - ``total``
     - ``#707070``
     - Cumulative sum bar at top
   * - ``alpha_fade``
     - ``0.35``
     - Opacity for counteracting bar components


Insets
------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``pos_cumulative_inset``
     - ``[0.19, 0.12, 0.175, 0.175]``
     - Position of cumulative inset: ``[left, bottom, width, height]``
   * - ``pos_text_size_inset``
     - ``[0.81, 0.12, 0.08, 0.08]``
     - Position of text size inset: ``[left, bottom, width, height]``
   * - ``cumulative_xlabel``
     - (auto)
     - X-axis label for cumulative inset
   * - ``cumulative_ylabel``
     - ``None``
     - Y-axis label for cumulative inset
   * - ``cumulative_xticks``
     - (auto)
     - X-axis ticks for cumulative inset
   * - ``cumulative_xticklabels``
     - (auto)
     - X-axis tick labels for cumulative inset


Saving Figures
--------------

Pass ``filename`` to ``get_shift_graph`` to save the figure:

.. code-block:: python

    shift.get_shift_graph(filename="my_shift.png", show_plot=False)

The ``dpi`` parameter controls the resolution of saved figures.
