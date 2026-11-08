"""
evaluation/test_cases.py
========================
Five labelled test cases covering different quality situations.

These are SYNTHETIC tests only — they do not use Kuku's actual screenplay.
They exist to verify the scoring pipeline works correctly before running
on real content.

Run with:  pytest evaluation/test_cases.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from parser.txt_parser import Scene, DialogueLine, ActionLine, Episode
from analysis.screenplay_analyzer import PromoCandidate, analyze_episode
from scoring.promo_scorer import score_candidate
from evaluation.quality_checker import check_candidate
from parser.timestamp_parser import validate_duration


# ---------------------------------------------------------------------------
# Helper: build a minimal scene
# ---------------------------------------------------------------------------

def _make_scene(num, start, end, dialogue_text="", action_text="",
                char="CHARACTER", tone="") -> Scene:
    s = Scene(number=num, heading="INT. TEST ROOM – DAY",
              start=start, end=end)
    if dialogue_text:
        s.dialogue = [DialogueLine(character=char, tone=tone,
                                    text=dialogue_text,
                                    start=start + 0.5, end=start + 3.0)]
    if action_text:
        s.actions = [ActionLine(description=action_text,
                                 start=start, end=start + 2.0)]
    s.raw_text = (action_text or "") + " " + (dialogue_text or "")
    return s


def _make_candidate(hook=5.0, cliff=5.0, standalone=5.0,
                    emotion=5.0, curiosity=5.0,
                    duration=35.0, start=0.0) -> PromoCandidate:
    end = start + duration
    c = PromoCandidate(
        promo_id="TEST-001", episode=1,
        scene_range=(1, 1),
        start=start, end=end, duration=duration,
        start_ts="00:00.00", end_ts="00:35.00",
        hook_score=hook, hook_type="Test", hook_evidence="Test hook",
        cliffhanger_score=cliff, cliffhanger_type="Test",
        cliffhanger_evidence="Test cliff",
        standalone_score=standalone, standalone_explanation="Test context",
        emotional_intensity=emotion, viewer_curiosity=curiosity,
    )
    return c


# ---------------------------------------------------------------------------
# Test Case 1 — Strong hook, strong cliffhanger → high score, Post First
# ---------------------------------------------------------------------------

class TestCase1_StrongPromo:
    """
    SCENARIO: Opening has a violent action + shocking dialogue.
              Ending cuts on an unanswered question.
    EXPECTED: total_score ≥ 60, recommendation is NOT 'Needs Human Review',
              no quality flags (evidence field populated).
    """

    def setup_method(self):
        self.c = _make_candidate(hook=8.0, cliff=7.0, standalone=6.0,
                                  emotion=7.0, curiosity=6.0, duration=35.0)
        # Add evidence so the quality checker doesn't flag it
        self.c.evidence = 'EP1 [00:05.00] CHARACTER: "You swore you loved me!"'
        score_candidate(self.c)

    def test_total_score_high(self):
        # hook(8×2.5) + cliff(7×2.5) + standalone(6×2.0) + emotion(7×1.5) + curiosity(6×1.5)
        # = 20 + 17.5 + 12 + 10.5 + 9 = 69 → assert ≥ 60
        assert self.c.total_score >= 60, (
            f"Expected total ≥ 60, got {self.c.total_score}"
        )

    def test_recommendation_not_needs_review(self):
        assert "Human Review" not in self.c.recommendation, (
            f"Expected a positive recommendation, got '{self.c.recommendation}'"
        )

    def test_no_quality_flags(self):
        flags = check_candidate(self.c)
        assert len(flags) == 0, f"Unexpected flags: {flags}"


# ---------------------------------------------------------------------------
# Test Case 2 — Weak hook → should be flagged
# ---------------------------------------------------------------------------

class TestCase2_WeakHook:
    """
    SCENARIO: Opening is a slow establishing shot with no keywords.
    EXPECTED: hook_score < 3, quality_flags contains a hook warning.
    """

    def setup_method(self):
        self.c = _make_candidate(hook=1.5, cliff=6.0, standalone=5.0,
                                  emotion=4.0, curiosity=4.0, duration=40.0)
        score_candidate(self.c)

    def test_hook_score_low(self):
        assert self.c.hook_score < 3.0

    def test_quality_flag_raised(self):
        flags = check_candidate(self.c)
        hook_flags = [f for f in flags if "hook" in f.lower()]
        assert len(hook_flags) >= 1, "Expected a hook-weakness flag"


# ---------------------------------------------------------------------------
# Test Case 3 — Duration too short → flagged as invalid
# ---------------------------------------------------------------------------

class TestCase3_TooShort:
    """
    SCENARIO: Clip is only 15 seconds long (below 20 s minimum).
    EXPECTED: validate_duration returns False, quality_flags contains duration error.
    """

    def setup_method(self):
        self.c = _make_candidate(hook=7.0, cliff=7.0, standalone=6.0,
                                  emotion=5.0, curiosity=5.0, duration=15.0)
        score_candidate(self.c)

    def test_duration_validation_fails(self):
        valid, reason = validate_duration(self.c.start, self.c.end,
                                          min_sec=20.0, max_sec=60.0)
        assert not valid, "Expected validation to fail for 15 s clip"

    def test_quality_flag_duration(self):
        flags = check_candidate(self.c)
        dur_flags = [f for f in flags if "short" in f.lower() or "long" in f.lower()
                     or "duration" in f.lower() or "Too" in f]
        assert len(dur_flags) >= 1, f"Expected duration flag, got: {flags}"


# ---------------------------------------------------------------------------
# Test Case 4 — No standalone context → flagged, Needs Human Review
# ---------------------------------------------------------------------------

class TestCase4_NoContext:
    """
    SCENARIO: Clip is emotionally intense but has no identifying dialogue
              that would orient a new viewer.
    EXPECTED: standalone_score < 2, total_score < 50, Needs Human Review.
    """

    def setup_method(self):
        self.c = _make_candidate(hook=6.0, cliff=6.0, standalone=0.5,
                                  emotion=7.0, curiosity=3.0, duration=30.0)
        score_candidate(self.c)

    def test_standalone_low(self):
        assert self.c.standalone_score < 2.0

    def test_recommendation_human_review(self):
        assert "Human Review" in self.c.recommendation, (
            f"Expected 'Needs Human Review', got '{self.c.recommendation}'"
        )

    def test_quality_flag_context(self):
        flags = check_candidate(self.c)
        ctx_flags = [f for f in flags if "standalone" in f.lower()
                     or "context" in f.lower()]
        assert len(ctx_flags) >= 1, f"Expected context flag, got: {flags}"


# ---------------------------------------------------------------------------
# Test Case 5 — Scoring weights sum to 100
# ---------------------------------------------------------------------------

class TestCase5_WeightIntegrity:
    """
    SCENARIO: Verify the scoring configuration is internally consistent.
    EXPECTED: weights sum to exactly 100; changing weights changes the score.
    """

    def test_default_weights_sum_to_100(self):
        import config
        total = sum(config.SCORING_WEIGHTS.values())
        assert total == 100, f"Weights sum to {total}, expected 100"

    def test_custom_weights_change_score(self):
        c1 = _make_candidate(hook=8.0, cliff=4.0, standalone=5.0,
                              emotion=5.0, curiosity=5.0, duration=35.0)
        c2 = _make_candidate(hook=8.0, cliff=4.0, standalone=5.0,
                              emotion=5.0, curiosity=5.0, duration=35.0)

        # Default weights
        score_candidate(c1)

        # Weights that heavily favour hook
        score_candidate(c2, weights={
            "hook_strength": 50,
            "cliffhanger_strength": 20,
            "standalone_context": 15,
            "emotional_intensity": 10,
            "viewer_curiosity": 5,
        })

        # With higher hook weight, c2 should score higher (hook = 8)
        assert c2.total_score >= c1.total_score, (
            f"Hook-heavy weights should boost score: c1={c1.total_score}, c2={c2.total_score}"
        )

    def test_invalid_weights_raise_error(self):
        c = _make_candidate()
        with pytest.raises(ValueError):
            score_candidate(c, weights={
                "hook_strength": 10,
                "cliffhanger_strength": 10,
                "standalone_context": 10,
                "emotional_intensity": 10,
                "viewer_curiosity": 10,   # sums to 50, not 100
            })


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCase6_ScoreSafety:
    def test_scores_are_clamped_to_documented_range(self):
        c = _make_candidate(hook=99, cliff=-4, standalone=5, emotion=5, curiosity=5)
        score_candidate(c)
        assert 0 <= c.total_score <= 100

    def test_weight_keys_must_be_complete(self):
        c = _make_candidate()
        with pytest.raises(ValueError):
            score_candidate(c, weights={"hook_strength": 100})


class TestCase7_TimestampIntegrity:
    def test_mismatched_duration_is_flagged(self):
        c = _make_candidate(duration=35)
        c.end = 50
        flags = check_candidate(c)
        assert any("mismatch" in flag.lower() for flag in flags)
