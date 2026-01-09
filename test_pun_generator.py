#!/usr/bin/env python3
"""Tests for pun_generator module."""

import unittest
from pun_generator import phoneme_edit_distance, get_stressed_vowel, are_peer_phonemes, phone_to_peers


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
        # Words with secondary stress vowels that are NOT peers
        pron1 = ['K', 'ER2', 'T']
        pron2 = ['K', 'AH2', 'T']
        # Should be 1 (normal substitution) since neither is primary stress and not peers
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1.0)

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
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1.0)

    # Peer phoneme tests
    def test_peer_consonants_half_cost(self):
        """Peer consonants (P/T/K) should cost 0.5 to substitute."""
        # pat [P AE1 T] vs tat [T AE1 T] - P and T are peers
        pron1 = ['P', 'AE1', 'T']
        pron2 = ['T', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_voiced_stops_half_cost(self):
        """Peer voiced stops (B/D/G) should cost 0.5 to substitute."""
        # bad [B AE1 D] vs dad [D AE1 D] - B and D are peers
        pron1 = ['B', 'AE1', 'D']
        pron2 = ['D', 'AE1', 'D']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_fricatives_half_cost(self):
        """Peer fricatives (F/S/TH/SH/HH) should cost 0.5 to substitute."""
        # fat [F AE1 T] vs sat [S AE1 T] - F and S are peers
        pron1 = ['F', 'AE1', 'T']
        pron2 = ['S', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_nasals_half_cost(self):
        """Peer nasals (M/N/NG) should cost 0.5 to substitute."""
        # man [M AE1 N] vs nan [N AE1 N] - M and N are peers
        pron1 = ['M', 'AE1', 'N']
        pron2 = ['N', 'AE1', 'N']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_liquids_half_cost(self):
        """Peer liquids (L/R) should cost 0.5 to substitute."""
        # lap [L AE1 P] vs rap [R AE1 P] - L and R are peers
        pron1 = ['L', 'AE1', 'P']
        pron2 = ['R', 'AE1', 'P']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_glides_half_cost(self):
        """Peer glides (W/Y) should cost 0.5 to substitute."""
        # wet [W EH1 T] vs yet [Y EH1 T] - W and Y are peers
        pron1 = ['W', 'EH1', 'T']
        pron2 = ['Y', 'EH1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_front_vowels_half_cost(self):
        """Peer front vowels should cost 0.5 to substitute (unstressed)."""
        # bit [B IH0 T] vs bet [B EH0 T] - IH and EH are peers (unstressed)
        pron1 = ['B', 'IH0', 'T']
        pron2 = ['B', 'EH0', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_back_vowels_half_cost(self):
        """Peer back vowels should cost 0.5 to substitute (unstressed)."""
        # book [B UH0 K] vs balk [B AO0 K] - UH and AO are peers (unstressed)
        pron1 = ['B', 'UH0', 'K']
        pron2 = ['B', 'AO0', 'K']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_diphthongs_half_cost(self):
        """Peer diphthongs (AY/AW/OY) should cost 0.5 to substitute (unstressed)."""
        pron1 = ['B', 'AY0', 'T']
        pron2 = ['B', 'AW0', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_non_peer_consonants_full_cost(self):
        """Non-peer consonants should cost 1.0 to substitute."""
        # pat [P AE1 T] vs mat [M AE1 T] - P and M are NOT peers
        pron1 = ['P', 'AE1', 'T']
        pron2 = ['M', 'AE1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 1.0)

    def test_multiple_peer_substitutions(self):
        """Multiple peer substitutions should accumulate at 0.5 each."""
        # pit [P IH1 T] vs kit [K IH1 T] - P->K is peer (0.5)
        pron1 = ['P', 'IH1', 'T']
        pron2 = ['K', 'IH1', 'T']
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 0.5)

    def test_peer_vs_non_peer_comparison(self):
        """Peer substitution should cost less than non-peer."""
        # P->K (peers) should be less than P->M (non-peers)
        peer_dist = phoneme_edit_distance(['P', 'AE1', 'T'], ['K', 'AE1', 'T'])
        non_peer_dist = phoneme_edit_distance(['P', 'AE1', 'T'], ['M', 'AE1', 'T'])
        self.assertLess(peer_dist, non_peer_dist)

    def test_peer_stressed_vowel_still_penalized(self):
        """Even peer vowels should be penalized if primary stressed."""
        # bit [B IH1 T] vs bet [B EH1 T] - IH and EH are peers but both stressed
        pron1 = ['B', 'IH1', 'T']
        pron2 = ['B', 'EH1', 'T']
        # Should use insert/delete (cost 2) not substitute (even at 0.5)
        self.assertEqual(phoneme_edit_distance(pron1, pron2), 2.0)

    def test_returns_float(self):
        """phoneme_edit_distance should return a float."""
        pron = ['K', 'AE1', 'T']
        result = phoneme_edit_distance(pron, pron)
        self.assertIsInstance(result, float)


class TestArePeerPhonemes(unittest.TestCase):
    """Tests for are_peer_phonemes function."""

    def test_same_phoneme_is_peer(self):
        """Same phoneme should be considered a peer of itself."""
        self.assertTrue(are_peer_phonemes('P', 'P'))
        self.assertTrue(are_peer_phonemes('AE', 'AE'))

    def test_peer_stops(self):
        """Voiceless stops P/T/K should be peers."""
        self.assertTrue(are_peer_phonemes('P', 'K'))
        self.assertTrue(are_peer_phonemes('P', 'T'))
        self.assertTrue(are_peer_phonemes('T', 'K'))

    def test_peer_voiced_stops(self):
        """Voiced stops B/D/G should be peers."""
        self.assertTrue(are_peer_phonemes('B', 'D'))
        self.assertTrue(are_peer_phonemes('B', 'G'))
        self.assertTrue(are_peer_phonemes('D', 'G'))

    def test_peer_fricatives_voiceless(self):
        """Voiceless fricatives F/S/TH/SH/HH should be peers."""
        self.assertTrue(are_peer_phonemes('F', 'S'))
        self.assertTrue(are_peer_phonemes('F', 'TH'))
        self.assertTrue(are_peer_phonemes('S', 'SH'))

    def test_peer_fricatives_voiced(self):
        """Voiced fricatives V/DH/Z/ZH should be peers."""
        self.assertTrue(are_peer_phonemes('V', 'Z'))
        self.assertTrue(are_peer_phonemes('V', 'DH'))
        self.assertTrue(are_peer_phonemes('Z', 'ZH'))

    def test_peer_nasals(self):
        """Nasals M/N/NG should be peers."""
        self.assertTrue(are_peer_phonemes('M', 'N'))
        self.assertTrue(are_peer_phonemes('M', 'NG'))
        self.assertTrue(are_peer_phonemes('N', 'NG'))

    def test_peer_liquids(self):
        """Liquids L/R should be peers."""
        self.assertTrue(are_peer_phonemes('L', 'R'))
        self.assertTrue(are_peer_phonemes('R', 'L'))

    def test_peer_glides(self):
        """Glides W/Y should be peers."""
        self.assertTrue(are_peer_phonemes('W', 'Y'))
        self.assertTrue(are_peer_phonemes('Y', 'W'))

    def test_peer_front_vowels(self):
        """Front vowels IY/IH/EY/EH/AE should be peers."""
        self.assertTrue(are_peer_phonemes('IY', 'IH'))
        self.assertTrue(are_peer_phonemes('EY', 'EH'))
        self.assertTrue(are_peer_phonemes('AE', 'EH'))

    def test_peer_back_vowels(self):
        """Back vowels UW/UH/OW/AO/AA should be peers."""
        self.assertTrue(are_peer_phonemes('UW', 'UH'))
        self.assertTrue(are_peer_phonemes('OW', 'AO'))
        self.assertTrue(are_peer_phonemes('AA', 'AO'))

    def test_peer_diphthongs(self):
        """Diphthongs AY/AW/OY should be peers."""
        self.assertTrue(are_peer_phonemes('AY', 'AW'))
        self.assertTrue(are_peer_phonemes('AW', 'OY'))
        self.assertTrue(are_peer_phonemes('AY', 'OY'))

    def test_non_peers(self):
        """Phonemes from different groups should not be peers."""
        self.assertFalse(are_peer_phonemes('P', 'M'))  # stop vs nasal
        self.assertFalse(are_peer_phonemes('F', 'L'))  # fricative vs liquid
        self.assertFalse(are_peer_phonemes('B', 'P'))  # voiced vs voiceless stop
        self.assertFalse(are_peer_phonemes('IY', 'UW'))  # front vs back vowel

    def test_unknown_phoneme(self):
        """Unknown phonemes should not be peers with anything."""
        self.assertFalse(are_peer_phonemes('XX', 'P'))
        self.assertFalse(are_peer_phonemes('P', 'XX'))


class TestPhoneToPeers(unittest.TestCase):
    """Tests for phone_to_peers data structure."""

    def test_all_stops_present(self):
        """All stop consonants should be in phone_to_peers."""
        stops = ['P', 'T', 'K', 'B', 'D', 'G']
        for stop in stops:
            self.assertIn(stop, phone_to_peers)

    def test_all_fricatives_present(self):
        """All fricatives should be in phone_to_peers."""
        fricatives = ['F', 'TH', 'S', 'SH', 'HH', 'V', 'DH', 'Z', 'ZH']
        for fric in fricatives:
            self.assertIn(fric, phone_to_peers)

    def test_all_nasals_present(self):
        """All nasals should be in phone_to_peers."""
        nasals = ['M', 'N', 'NG']
        for nasal in nasals:
            self.assertIn(nasal, phone_to_peers)

    def test_symmetry(self):
        """Peer relationships should be symmetric."""
        for phoneme, peers in phone_to_peers.items():
            for peer in peers:
                if peer in phone_to_peers:
                    self.assertIn(phoneme, phone_to_peers[peer],
                        f"{phoneme} is peer of {peer} but not vice versa")


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
