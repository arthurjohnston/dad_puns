#!/usr/bin/env python3
"""Tests for pun_generator module."""

import unittest
from pun_generator import (
    phoneme_edit_distance, get_stressed_vowel, is_stressed_vowel,
    get_vowel, IPA_VOWELS, are_peer_phonemes, phone_to_peers,
    count_syllables, MIN_WORD_LENGTH, get_word_frequency, pun_rank
)


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
        # Use non-peer vowels (ɛ is front, ʌ is central - not peers)
        pron1 = ['k', 'ˌɛ', 't']
        pron2 = ['k', 'ˌʌ', 't']
        # Should be 1 (normal substitution) since neither is primary stress and not peers
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


class TestArePeerPhonemes(unittest.TestCase):
    """Tests for are_peer_phonemes function."""

    def test_same_phoneme_is_peer(self):
        """Same phoneme should be considered a peer of itself."""
        self.assertTrue(are_peer_phonemes('p', 'p'))
        self.assertTrue(are_peer_phonemes('æ', 'æ'))

    def test_peer_stops(self):
        """Voiceless stops p/t/k should be peers."""
        self.assertTrue(are_peer_phonemes('p', 'k'))
        self.assertTrue(are_peer_phonemes('p', 't'))
        self.assertTrue(are_peer_phonemes('t', 'k'))

    def test_peer_voiced_stops(self):
        """Voiced stops b/d/ɡ should be peers."""
        self.assertTrue(are_peer_phonemes('b', 'd'))
        self.assertTrue(are_peer_phonemes('b', 'ɡ'))
        self.assertTrue(are_peer_phonemes('d', 'ɡ'))

    def test_peer_fricatives_voiceless(self):
        """Voiceless fricatives f/θ/s/ʃ/h should be peers."""
        self.assertTrue(are_peer_phonemes('f', 's'))
        self.assertTrue(are_peer_phonemes('f', 'θ'))
        self.assertTrue(are_peer_phonemes('s', 'ʃ'))

    def test_peer_fricatives_voiced(self):
        """Voiced fricatives v/ð/z/ʒ should be peers."""
        self.assertTrue(are_peer_phonemes('v', 'z'))
        self.assertTrue(are_peer_phonemes('v', 'ð'))
        self.assertTrue(are_peer_phonemes('z', 'ʒ'))

    def test_peer_nasals(self):
        """Nasals m/n/ŋ should be peers."""
        self.assertTrue(are_peer_phonemes('m', 'n'))
        self.assertTrue(are_peer_phonemes('m', 'ŋ'))
        self.assertTrue(are_peer_phonemes('n', 'ŋ'))

    def test_peer_liquids(self):
        """Liquids l/ɹ should be peers."""
        self.assertTrue(are_peer_phonemes('l', 'ɹ'))
        self.assertTrue(are_peer_phonemes('ɹ', 'l'))

    def test_peer_glides(self):
        """Glides w/j should be peers."""
        self.assertTrue(are_peer_phonemes('w', 'j'))
        self.assertTrue(are_peer_phonemes('j', 'w'))

    def test_peer_front_vowels(self):
        """Front vowels i/ɪ/eɪ/ɛ/æ should be peers."""
        self.assertTrue(are_peer_phonemes('i', 'ɪ'))
        self.assertTrue(are_peer_phonemes('eɪ', 'ɛ'))
        self.assertTrue(are_peer_phonemes('æ', 'ɛ'))

    def test_peer_back_vowels(self):
        """Back vowels u/ʊ/oʊ/ɔ/ɑ should be peers."""
        self.assertTrue(are_peer_phonemes('u', 'ʊ'))
        self.assertTrue(are_peer_phonemes('oʊ', 'ɔ'))
        self.assertTrue(are_peer_phonemes('ɑ', 'ɔ'))

    def test_peer_diphthongs(self):
        """Diphthongs aɪ/aʊ/ɔɪ should be peers."""
        self.assertTrue(are_peer_phonemes('aɪ', 'aʊ'))
        self.assertTrue(are_peer_phonemes('aʊ', 'ɔɪ'))
        self.assertTrue(are_peer_phonemes('aɪ', 'ɔɪ'))

    def test_non_peers(self):
        """Phonemes from different groups should not be peers."""
        self.assertFalse(are_peer_phonemes('p', 'm'))  # stop vs nasal
        self.assertFalse(are_peer_phonemes('f', 'l'))  # fricative vs liquid
        self.assertFalse(are_peer_phonemes('b', 'p'))  # voiced vs voiceless stop
        self.assertFalse(are_peer_phonemes('i', 'u'))  # front vs back vowel

    def test_unknown_phoneme(self):
        """Unknown phonemes should not be peers with anything."""
        self.assertFalse(are_peer_phonemes('xx', 'p'))
        self.assertFalse(are_peer_phonemes('p', 'xx'))


class TestPhoneToPeers(unittest.TestCase):
    """Tests for phone_to_peers data structure."""

    def test_all_stops_present(self):
        """All stop consonants should be in phone_to_peers."""
        stops = ['p', 't', 'k', 'b', 'd', 'ɡ']
        for stop in stops:
            self.assertIn(stop, phone_to_peers)

    def test_all_fricatives_present(self):
        """All fricatives should be in phone_to_peers."""
        fricatives = ['f', 'θ', 's', 'ʃ', 'h', 'v', 'ð', 'z', 'ʒ']
        for fric in fricatives:
            self.assertIn(fric, phone_to_peers)

    def test_all_nasals_present(self):
        """All nasals should be in phone_to_peers."""
        nasals = ['m', 'n', 'ŋ']
        for nasal in nasals:
            self.assertIn(nasal, phone_to_peers)

    def test_symmetry(self):
        """Peer relationships should be symmetric."""
        for phoneme, peers in phone_to_peers.items():
            for peer in peers:
                if peer in phone_to_peers:
                    self.assertIn(phoneme, phone_to_peers[peer],
                        f"{phoneme} is peer of {peer} but not vice versa")


class TestCountSyllables(unittest.TestCase):
    """Tests for count_syllables function."""

    def test_one_syllable(self):
        """Single syllable words."""
        # cat [k ˈæ t]
        self.assertEqual(count_syllables(['k', 'ˈæ', 't']), 1)
        # dog [d ˈɑː ɡ]
        self.assertEqual(count_syllables(['d', 'ˈɑː', 'ɡ']), 1)

    def test_two_syllables(self):
        """Two syllable words."""
        # happy [h ˈæ p i]
        self.assertEqual(count_syllables(['h', 'ˈæ', 'p', 'i']), 2)

    def test_three_syllables(self):
        """Three syllable words."""
        # elephant [ˈɛ l ə f ə n t]
        self.assertEqual(count_syllables(['ˈɛ', 'l', 'ə', 'f', 'ə', 'n', 't']), 3)

    def test_empty(self):
        """Empty pronunciation."""
        self.assertEqual(count_syllables([]), 0)

    def test_consonants_only(self):
        """Consonants don't count as syllables."""
        self.assertEqual(count_syllables(['k', 't', 's']), 0)


class TestMinWordLength(unittest.TestCase):
    """Tests for MIN_WORD_LENGTH constant."""

    def test_min_word_length_value(self):
        """MIN_WORD_LENGTH should be at least 2."""
        self.assertGreaterEqual(MIN_WORD_LENGTH, 2)


class TestGetWordFrequency(unittest.TestCase):
    """Tests for get_word_frequency function."""

    def test_common_word(self):
        """Common words should have high frequency."""
        freq = get_word_frequency('the')
        self.assertGreater(freq, 0)

    def test_unknown_word(self):
        """Unknown words should return 0."""
        freq = get_word_frequency('xyzabc123notaword')
        self.assertEqual(freq, 0)

    def test_case_insensitive(self):
        """Frequency lookup should be case insensitive."""
        self.assertEqual(get_word_frequency('cat'), get_word_frequency('CAT'))


class TestPunRank(unittest.TestCase):
    """Tests for pun_rank function."""

    def test_returns_tuple(self):
        """pun_rank should return a tuple."""
        result = pun_rank(1.0, 1000)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_lower_distance_sorts_first(self):
        """Lower edit distance should always sort before higher."""
        rank_low = pun_rank(0.5, 100)
        rank_high = pun_rank(1.0, 1000000)  # Even with much higher frequency
        self.assertLess(rank_low, rank_high)

    def test_higher_frequency_sorts_first_same_distance(self):
        """Higher frequency should sort first when distance is equal."""
        rank_common = pun_rank(1.0, 1000000)
        rank_rare = pun_rank(1.0, 100)
        self.assertLess(rank_common, rank_rare)

    def test_same_values(self):
        """Same values should produce equal ranks."""
        self.assertEqual(pun_rank(1.0, 500), pun_rank(1.0, 500))


if __name__ == '__main__':
    unittest.main()
