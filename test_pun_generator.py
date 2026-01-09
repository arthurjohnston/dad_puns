#!/usr/bin/env python3
"""Tests for pun_generator module."""

import unittest
from pun_generator import phoneme_edit_distance, get_stressed_vowel


class TestPhonemeEditDistance(unittest.TestCase):
    """Tests for phoneme_edit_distance function."""

    def test_identical_pronunciations(self):
        """Identical pronunciations should have distance 0."""
        pron = ['K', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron, pron), 0)

    def test_single_deletion(self):
        """Single phoneme deletion should have distance 1."""
        # cat [K AE1 T] vs at [AE1 T]
        pron1 = ['K', 'AE1', 'T']
        pron2 = ['AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_single_insertion(self):
        """Single phoneme insertion should have distance 1."""
        # cat [K AE1 T] vs cats [K AE1 T S]
        pron1 = ['K', 'AE1', 'T']
        pron2 = ['K', 'AE1', 'T', 'S']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_single_substitution_unstressed(self):
        """Substituting unstressed phoneme should have distance 1."""
        # cat [K AE1 T] vs bat [B AE1 T]
        pron1 = ['K', 'AE1', 'T']
        pron2 = ['B', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_substitution_stressed_vowel_uses_insert_delete(self):
        """Different stressed vowels use insert+delete (cost 2) instead of substitute."""
        # cat [K AE1 T] vs cut [K AH1 T] - different stressed vowels
        # Algorithm uses delete AE1 + insert AH1 = cost 2, not substitute = cost 1000
        pron1 = ['K', 'AE1', 'T']
        pron2 = ['K', 'AH1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_same_stressed_vowel_different_consonants(self):
        """Same stressed vowel with different consonants should be low distance."""
        # cat [K AE1 T] vs that [DH AE1 T]
        pron1 = ['K', 'AE1', 'T']
        pron2 = ['DH', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_stress_markers_ignored_for_matching(self):
        """Same vowel with different stress should still match."""
        # Unstressed vs no stress marker should be equal
        pron1 = ['K', 'AE0', 'T']
        pron2 = ['K', 'AE2', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0)

    def test_empty_pronunciation(self):
        """Empty pronunciation should equal length of other."""
        pron1 = ['K', 'AE1', 'T']
        pron2 = []
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 3)
        self.assertEqual(phoneme_edit_distance(pron2, pron1), 3)

    def test_multiple_edits(self):
        """Multiple edits should accumulate."""
        # dog [D AO1 G] vs cat [K AE1 T] - very different
        # D->K (1), AO1->delete+insert AE1 (2), G->T (1) = 4
        pron1 = ['D', 'AO1', 'G']
        pron2 = ['K', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 4)

    def test_similar_words_low_distance(self):
        """Similar sounding words should have low distance."""
        # can [K AE1 N] vs cat [K AE1 T]
        pron1 = ['K', 'AE1', 'N']
        pron2 = ['K', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_longer_words(self):
        """Test with longer pronunciations."""
        # elephant [EH1 L AH0 F AH0 N T] vs elegant [EH1 L AH0 G AH0 N T]
        pron1 = ['EH1', 'L', 'AH0', 'F', 'AH0', 'N', 'T']
        pron2 = ['EH1', 'L', 'AH0', 'G', 'AH0', 'N', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)


    def test_stressed_vowel_mismatch_penalized(self):
        """Words with different stressed vowels should have higher distance."""
        # Same consonant frame, different stressed vowels
        # bit [B IH1 T] vs bat [B AE1 T]
        pron1 = ['B', 'IH1', 'T']
        pron2 = ['B', 'AE1', 'T']
        # Should be 2 (delete + insert) not 1 (substitute)
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_stressed_vowel_match_vs_mismatch(self):
        """Matching stressed vowel should cost less than mismatched."""
        # cat [K AE1 T] vs can [K AE1 N] - same stressed vowel, distance 1
        same_stress_dist = phoneme_edit_distance(['K', 'AE1', 'T'], ['K', 'AE1', 'N'])
        # cat [K AE1 T] vs cut [K AH1 T] - different stressed vowel, distance 2
        diff_stress_dist = phoneme_edit_distance(['K', 'AE1', 'T'], ['K', 'AH1', 'T'])
        self.assertLess(same_stress_dist, diff_stress_dist)

    def test_secondary_stress_can_substitute(self):
        """Secondary stress (2) should allow normal substitution."""
        # Words with secondary stress vowels
        pron1 = ['K', 'AE2', 'T']
        pron2 = ['K', 'AH2', 'T']
        # Should be 1 (normal substitution) since neither is primary stress
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_primary_vs_secondary_stress(self):
        """Primary stress vs secondary stress should have high substitution cost."""
        pron1 = ['K', 'AE1', 'T']  # primary stress
        pron2 = ['K', 'AE2', 'T']  # secondary stress (same vowel, different stress)
        # Same vowel sound, just different stress marker - should be 0
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0)

    def test_cow_vs_count(self):
        """Real word example: cow vs count (same stressed vowel)."""
        # cow [K AW1] vs count [K AW1 N T]
        pron1 = ['K', 'AW1']
        pron2 = ['K', 'AW1', 'N', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_cat_vs_caught(self):
        """Real word example: cat vs caught (different stressed vowels)."""
        # cat [K AE1 T] vs caught [K AO1 T]
        pron1 = ['K', 'AE1', 'T']
        pron2 = ['K', 'AO1', 'T']
        # Different stressed vowels - should be 2 (delete + insert)
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_dog_vs_dig(self):
        """Real word example: dog vs dig (different stressed vowels)."""
        # dog [D AO1 G] vs dig [D IH1 G]
        pron1 = ['D', 'AO1', 'G']
        pron2 = ['D', 'IH1', 'G']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_dog_vs_fog(self):
        """Real word example: dog vs fog (same stressed vowel)."""
        # dog [D AO1 G] vs fog [F AO1 G]
        pron1 = ['D', 'AO1', 'G']
        pron2 = ['F', 'AO1', 'G']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)


class TestGetStressedVowel(unittest.TestCase):
    """Tests for get_stressed_vowel function."""

    def test_single_stressed_vowel(self):
        """Should return the vowel with primary stress."""
        # cat [K AE1 T]
        pron = ['K', 'AE1', 'T']
        self.assertEqual(get_stressed_vowel(pron), 'AE')

    def test_multiple_vowels_returns_primary(self):
        """Should return only the primary stressed vowel."""
        # about [AH0 B AW1 T]
        pron = ['AH0', 'B', 'AW1', 'T']
        self.assertEqual(get_stressed_vowel(pron), 'AW')

    def test_no_stressed_vowel(self):
        """Should return None if no primary stress."""
        pron = ['AH0', 'B', 'AW0', 'T']
        self.assertIsNone(get_stressed_vowel(pron))

    def test_empty_pronunciation(self):
        """Should return None for empty pronunciation."""
        self.assertIsNone(get_stressed_vowel([]))


if __name__ == '__main__':
    unittest.main()
