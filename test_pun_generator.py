#!/usr/bin/env python3
"""Tests for pun_generator module."""

import unittest
from pun_generator import phoneme_edit_distance, get_stressed_vowel, is_stressed_vowel, get_vowel, IPA_VOWELS


class TestPhonemeEditDistance(unittest.TestCase):
    """Tests for phoneme_edit_distance function."""

    def test_identical_pronunciations(self):
        """Identical pronunciations should have distance 0."""
        pron = ['k', 'ˈæ', 't']
        self.assertEqual(phoneme_edit_distance(pron, pron), 0)

    def test_single_deletion(self):
        """Single phoneme deletion should have distance 1."""
        # cat [k ˈæ t] vs at [ˈæ t]
        pron1 = ['k', 'ˈæ', 't']
        pron2 = ['ˈæ', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_single_insertion(self):
        """Single phoneme insertion should have distance 1."""
        # cat [k ˈæ t] vs cats [k ˈæ t s]
        pron1 = ['k', 'ˈæ', 't']
        pron2 = ['k', 'ˈæ', 't', 's']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_single_substitution_unstressed(self):
        """Substituting unstressed phoneme should have distance 1."""
        # cat [k ˈæ t] vs bat [b ˈæ t]
        pron1 = ['k', 'ˈæ', 't']
        pron2 = ['b', 'ˈæ', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_substitution_stressed_vowel_uses_insert_delete(self):
        """Different stressed vowels use insert+delete (cost 2) instead of substitute."""
        # cat [k ˈæ t] vs cut [k ˈʌ t] - different stressed vowels
        pron1 = ['k', 'ˈæ', 't']
        pron2 = ['k', 'ˈʌ', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_same_stressed_vowel_different_consonants(self):
        """Same stressed vowel with different consonants should be low distance."""
        # cat [k ˈæ t] vs that [ð ˈæ t]
        pron1 = ['k', 'ˈæ', 't']
        pron2 = ['ð', 'ˈæ', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_stress_markers_ignored_for_matching(self):
        """Same vowel with different stress should still match."""
        # Unstressed vs secondary stress should be equal
        pron1 = ['k', 'æ', 't']
        pron2 = ['k', 'ˌæ', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0)

    def test_empty_pronunciation(self):
        """Empty pronunciation should equal length of other."""
        pron1 = ['k', 'ˈæ', 't']
        pron2 = []
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 3)
        self.assertEqual(phoneme_edit_distance(pron2, pron1), 3)

    def test_multiple_edits(self):
        """Multiple edits should accumulate."""
        # dog [d ˈɑː ɡ] vs cat [k ˈæ t] - very different
        pron1 = ['d', 'ˈɑː', 'ɡ']
        pron2 = ['k', 'ˈæ', 't']
        # d->k (1), ˈɑː->delete+insert ˈæ (2), ɡ->t (1) = 4
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 4)

    def test_similar_words_low_distance(self):
        """Similar sounding words should have low distance."""
        # can [k ˈæ n] vs cat [k ˈæ t]
        pron1 = ['k', 'ˈæ', 'n']
        pron2 = ['k', 'ˈæ', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_longer_words(self):
        """Test with longer pronunciations."""
        # elephant vs elegant - one phoneme different
        pron1 = ['ˈɛ', 'l', 'ə', 'f', 'ə', 'n', 't']
        pron2 = ['ˈɛ', 'l', 'ə', 'ɡ', 'ə', 'n', 't']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1)

    def test_stressed_vowel_mismatch_penalized(self):
        """Words with different stressed vowels should have higher distance."""
        # Same consonant frame, different stressed vowels
        # bit [b ˈɪ t] vs bat [b ˈæ t]
        pron1 = ['b', 'ˈɪ', 't']
        pron2 = ['b', 'ˈæ', 't']
        # Should be 2 (delete + insert) not 1 (substitute)
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2)

    def test_stressed_vowel_match_vs_mismatch(self):
        """Matching stressed vowel should cost less than mismatched."""
        # cat [k ˈæ t] vs can [k ˈæ n] - same stressed vowel, distance 1
        same_stress_dist = phoneme_edit_distance(['k', 'ˈæ', 't'], ['k', 'ˈæ', 'n'])
        # cat [k ˈæ t] vs cut [k ˈʌ t] - different stressed vowel, distance 2
        diff_stress_dist = phoneme_edit_distance(['k', 'ˈæ', 't'], ['k', 'ˈʌ', 't'])
        self.assertLess(same_stress_dist, diff_stress_dist)

    def test_secondary_stress_can_substitute(self):
        """Secondary stress (ˌ) should allow normal substitution."""
        pron1 = ['k', 'ˌɛ', 't']
        pron2 = ['k', 'ˌɪ', 't']
        # Should be 1 (normal substitution) since neither is primary stress
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1.0)

    def test_returns_float(self):
        """phoneme_edit_distance should return a float."""
        pron = ['k', 'ˈæ', 't']
        result = phoneme_edit_distance(pron, pron)
        self.assertIsInstance(result, float)


class TestIsStressedVowel(unittest.TestCase):
    """Tests for is_stressed_vowel function."""

    def test_stressed_vowel(self):
        """Phoneme starting with ˈ and containing vowel should return True."""
        self.assertTrue(is_stressed_vowel('ˈæ'))
        self.assertTrue(is_stressed_vowel('ˈɑː'))
        self.assertTrue(is_stressed_vowel('ˈaʊ'))

    def test_unstressed_vowel(self):
        """Vowel without stress marker should return False."""
        self.assertFalse(is_stressed_vowel('æ'))
        self.assertFalse(is_stressed_vowel('ə'))

    def test_secondary_stress(self):
        """Secondary stress (ˌ) should return False."""
        self.assertFalse(is_stressed_vowel('ˌæ'))

    def test_stressed_consonant(self):
        """Consonant with stress marker should return False."""
        self.assertFalse(is_stressed_vowel('ˈk'))
        self.assertFalse(is_stressed_vowel('ˈt'))


class TestGetVowel(unittest.TestCase):
    """Tests for get_vowel function."""

    def test_simple_vowel(self):
        """Should extract vowel from simple phoneme."""
        self.assertEqual(get_vowel('æ'), 'æ')
        self.assertEqual(get_vowel('ɪ'), 'ɪ')

    def test_stressed_vowel(self):
        """Should strip stress marker and return vowel."""
        self.assertEqual(get_vowel('ˈæ'), 'æ')
        self.assertEqual(get_vowel('ˌɪ'), 'ɪ')

    def test_consonant(self):
        """Should return None for consonants."""
        self.assertIsNone(get_vowel('k'))
        self.assertIsNone(get_vowel('t'))
        self.assertIsNone(get_vowel('ˈk'))


class TestGetStressedVowel(unittest.TestCase):
    """Tests for get_stressed_vowel function."""

    def test_single_stressed_vowel(self):
        """Should return the vowel with primary stress."""
        # cat [k ˈæ t]
        pron = ['k', 'ˈæ', 't']
        self.assertEqual(get_stressed_vowel(pron), 'æ')

    def test_multiple_vowels_returns_primary(self):
        """Should return only the primary stressed vowel."""
        # about [ə b ˈaʊ t]
        pron = ['ə', 'b', 'ˈaʊ', 't']
        self.assertEqual(get_stressed_vowel(pron), 'a')

    def test_no_stressed_vowel(self):
        """Should return None if no primary stress."""
        pron = ['ə', 'b', 'aʊ', 't']
        self.assertIsNone(get_stressed_vowel(pron))

    def test_empty_pronunciation(self):
        """Should return None for empty pronunciation."""
        self.assertIsNone(get_stressed_vowel([]))


class TestIPAVowels(unittest.TestCase):
    """Tests for IPA_VOWELS set."""

    def test_common_vowels_present(self):
        """Common IPA vowels should be in the set."""
        common_vowels = ['a', 'e', 'i', 'o', 'u', 'æ', 'ɪ', 'ʊ', 'ə', 'ɛ', 'ɑ', 'ɔ', 'ʌ']
        for vowel in common_vowels:
            self.assertIn(vowel, IPA_VOWELS, f"'{vowel}' should be in IPA_VOWELS")

    def test_consonants_not_present(self):
        """Consonants should not be in the set."""
        consonants = ['k', 't', 'p', 'b', 'd', 'g', 's', 'z', 'f', 'v', 'm', 'n', 'l', 'r']
        for cons in consonants:
            self.assertNotIn(cons, IPA_VOWELS, f"'{cons}' should not be in IPA_VOWELS")


if __name__ == '__main__':
    unittest.main()
