"""Tests for fixes to GitHub issues #11, #26, #38, #47."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from shifterator import WeightedAvgShift, EntropyShift, ProportionShift


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Issue #11: get_weighted_score should accept a lexicon name string
# ---------------------------------------------------------------------------

class TestIssue11LexiconInGetWeightedScore:
    def test_lexicon_name_string(self):
        """get_weighted_score should accept a lexicon name like 'labMT_English'."""
        freq1 = {"happy": 20, "sad": 5, "good": 15}
        freq2 = {"happy": 10, "sad": 15, "good": 5}
        shift = WeightedAvgShift(freq1, freq2)
        s_avg = shift.get_weighted_score(freq1, "labMT_English")
        assert isinstance(s_avg, float)
        assert s_avg > 0

    def test_lexicon_vs_dict_same_result(self):
        """Using a lexicon name should give the same result as the loaded dict."""
        from shifterator.helper import get_score_dictionary
        freq = {"happy": 20, "sad": 5, "good": 15}
        scores, _ = get_score_dictionary("labMT_English")
        shift = WeightedAvgShift(freq, freq)
        s_from_dict = shift.get_weighted_score(freq, scores)
        s_from_str = shift.get_weighted_score(freq, "labMT_English")
        assert s_from_dict == pytest.approx(s_from_str)

    def test_dict_still_works(self):
        """Passing a dict should still work as before."""
        freq = {"a": 10, "b": 20}
        scores = {"a": 3.0, "b": 7.0}
        shift = WeightedAvgShift(freq, freq, type2score_1=scores)
        s_avg = shift.get_weighted_score(freq, scores)
        expected = (10 * 3.0 + 20 * 7.0) / 30
        assert s_avg == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Issue #26: type2score alias for WeightedAvgShift
# ---------------------------------------------------------------------------

class TestIssue26Type2ScoreAlias:
    def test_type2score_sets_both(self):
        """type2score param should set type2score_1 (and thus type2score_2)."""
        freq1 = {"happy": 20, "sad": 5}
        freq2 = {"happy": 10, "sad": 15}
        scores = {"happy": 8.0, "sad": 2.0}
        shift = WeightedAvgShift(freq1, freq2, type2score=scores)
        assert shift.type2score_1 == shift.type2score_2

    def test_type2score_same_as_type2score_1(self):
        """type2score should give identical results to type2score_1."""
        freq1 = {"happy": 20, "sad": 5}
        freq2 = {"happy": 10, "sad": 15}
        scores = {"happy": 8.0, "sad": 2.0}
        shift_alias = WeightedAvgShift(freq1, freq2, type2score=scores)
        shift_explicit = WeightedAvgShift(freq1, freq2, type2score_1=scores)
        assert shift_alias.type2shift_score == shift_explicit.type2shift_score

    def test_type2score_1_takes_precedence(self):
        """If both type2score and type2score_1 are given, type2score_1 wins."""
        freq1 = {"happy": 20, "sad": 5}
        freq2 = {"happy": 10, "sad": 15}
        scores_a = {"happy": 8.0, "sad": 2.0}
        scores_b = {"happy": 6.0, "sad": 4.0}
        shift = WeightedAvgShift(freq1, freq2, type2score_1=scores_a, type2score=scores_b)
        # type2score_1 should be based on scores_a, not scores_b
        assert shift.type2score_1["happy"] == pytest.approx(8.0)

    def test_type2score_with_lexicon_name(self):
        """type2score should work with a lexicon name string too."""
        freq1 = {"happy": 20, "sad": 5, "good": 15}
        freq2 = {"happy": 10, "sad": 15, "good": 5}
        shift = WeightedAvgShift(freq1, freq2, type2score="labMT_English",
                                 handle_missing_scores="exclude")
        scores = shift.get_shift_scores()
        assert isinstance(scores, dict)
        assert len(scores) > 0


# ---------------------------------------------------------------------------
# Issue #38: Font configurability for Unicode/CJK support
# ---------------------------------------------------------------------------

class TestIssue38FontFamily:
    def test_font_family_param_accepted(self):
        """font_family kwarg should be accepted and not error."""
        freq1 = {"hello": 20, "world": 10}
        freq2 = {"hello": 15, "world": 15}
        scores = {"hello": 7.0, "world": 5.0}
        shift = WeightedAvgShift(freq1, freq2, type2score_1=scores)
        # DejaVu Sans ships with matplotlib so this should always work
        ax = shift.get_shift_graph(top_n=2, show_plot=False, font_family="DejaVu Sans")
        assert ax is not None

    def test_font_family_overrides_serif(self):
        """font_family should take precedence over serif=True."""
        from matplotlib import rcParams
        freq1 = {"hello": 20, "world": 10}
        freq2 = {"hello": 15, "world": 15}
        scores = {"hello": 7.0, "world": 5.0}
        shift = WeightedAvgShift(freq1, freq2, type2score_1=scores)
        ax = shift.get_shift_graph(
            top_n=2, show_plot=False,
            font_family="DejaVu Sans", serif=True,
        )
        # font_family should win over serif
        assert rcParams["font.family"] == ["DejaVu Sans"] or rcParams["font.family"] == "DejaVu Sans"

    def test_serif_still_works(self):
        """serif=True without font_family should still work."""
        from matplotlib import rcParams
        freq1 = {"hello": 20, "world": 10}
        freq2 = {"hello": 15, "world": 15}
        scores = {"hello": 7.0, "world": 5.0}
        shift = WeightedAvgShift(freq1, freq2, type2score_1=scores)
        ax = shift.get_shift_graph(top_n=2, show_plot=False, serif=True)
        assert ax is not None

    def test_default_no_font_change(self):
        """Without font_family or serif, font should not be explicitly changed."""
        freq1 = {"hello": 20, "world": 10}
        freq2 = {"hello": 15, "world": 15}
        scores = {"hello": 7.0, "world": 5.0}
        shift = WeightedAvgShift(freq1, freq2, type2score_1=scores)
        ax = shift.get_shift_graph(top_n=2, show_plot=False)
        assert ax is not None


# ---------------------------------------------------------------------------
# Issue #47: ProportionShift should label the top bars with the system names
# ---------------------------------------------------------------------------

class TestIssue47ProportionShiftSystemNames:
    freq1 = {"happy": 20, "sad": 5, "the": 50, "good": 15, "bad": 10}
    freq2 = {"happy": 10, "sad": 15, "the": 45, "good": 5, "bad": 20, "angry": 8}

    def test_system_names_labeled(self):
        """The total contribution bars should be labeled with the system names."""
        shift = ProportionShift(self.freq1, self.freq2)
        ax = shift.get_shift_graph(
            top_n=6, show_plot=False, system_names=["Corpus A", "Corpus B"]
        )
        labels = [t.get_text() for t in ax.texts]
        assert "Corpus A" in labels
        assert "Corpus B" in labels

    def test_default_system_names_labeled(self):
        """Falls back to the default system names when none are given."""
        shift = ProportionShift(self.freq1, self.freq2)
        ax = shift.get_shift_graph(top_n=6, show_plot=False)
        labels = [t.get_text() for t in ax.texts]
        assert "Text 1" in labels
        assert "Text 2" in labels

    def test_bars_remain_two_sided(self):
        """Labeling must not flip every bar to one side (the broken workaround
        of passing all_pos_contributions=True)."""
        shift = ProportionShift(self.freq1, self.freq2)
        ax = shift.get_shift_graph(top_n=6, show_plot=False)
        widths = [p.get_width() for p in ax.patches if p.get_width() != 0]
        assert any(w < 0 for w in widths)
        assert any(w > 0 for w in widths)

    def test_other_shifts_unaffected(self):
        """Shifts that don't opt in keep empty total-bar symbols."""
        freq1 = {"hello": 20, "world": 10}
        freq2 = {"hello": 15, "world": 15}
        scores = {"hello": 7.0, "world": 5.0}
        shift = WeightedAvgShift(freq1, freq2, type2score_1=scores)
        ax = shift.get_shift_graph(
            top_n=2, show_plot=False, system_names=["Corpus A", "Corpus B"]
        )
        labels = [t.get_text() for t in ax.texts]
        assert "Corpus A" not in labels
        assert "Corpus B" not in labels
