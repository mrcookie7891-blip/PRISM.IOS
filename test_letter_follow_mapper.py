"""Tests for letter_follow_mapper."""

import unittest

from letter_follow_mapper import build_letter_follow_model


class LetterFollowMapperTests(unittest.TestCase):
    def test_simple_bigrams(self) -> None:
        model = build_letter_follow_model("abab")
        a = model.profile_for("a")
        b = model.profile_for("b")
        assert a is not None and b is not None
        self.assertAlmostEqual(a.as_dict()["b"], 1.0)
        self.assertAlmostEqual(b.as_dict()["a"], 1.0)

    def test_probabilities_in_unit_interval(self) -> None:
        model = build_letter_follow_model("the quick brown fox")
        for profile in model.profiles.values():
            total = sum(profile.probabilities)
            if total == 0:
                continue
            self.assertAlmostEqual(total, 1.0, places=7)
            for p in profile.probabilities:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_english_th_follows_h(self) -> None:
        model = build_letter_follow_model("the thee thou thread")
        t = model.profile_for("t")
        assert t is not None
        self.assertGreater(t.as_dict()["h"], 0.5)

    def test_binary_view(self) -> None:
        model = build_letter_follow_model("aa")
        binary = model.to_binary(threshold=0.5)
        self.assertEqual(binary["a"], (1,) + (0,) * 25)


if __name__ == "__main__":
    unittest.main()
