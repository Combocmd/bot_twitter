#!/usr/bin/env python3
"""
Test cases for security and bug fixes in the rewriter module.

This script verifies:
1. Security filter rejects tweets with banned words (SKIP/APPROVE)
2. Alert emoji handling (only add 🚨 when source had alert keywords)
3. Source attribution logic (Cas A/B with tier-1 journalists)
4. Fallback mechanism for invalid LLM outputs
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rewriter import TweetRewriter
from src.config import get_config


class TestSecurityFixes(unittest.TestCase):

    def setUp(self):
        """Set up test environment with a clean rewriter instance."""
        self.config = get_config()
        self.rewriter = TweetRewriter(self.config)

    def test_security_filter_banned_words(self):
        """Test that security filter generates a fallback for tweets with banned words."""
        test_cases = [
            "Barcelona: SKIP",
            "APPROVE: This is a test",
            "This tweet contains SKIP and should be rejected",
            "This tweet contains APPROVE and should be rejected",
        ]

        for text in test_cases:
            with self.subTest(text=text):
                # When security check fails on input, rewrite() returns the fallback rewrite
                result = self.rewriter.rewrite(text, "test_source")
                self.assertTrue(result.success, f"Fallback should succeed for '{text}'")
                self.assertNotIn("SKIP", result.rewritten_text)
                self.assertNotIn("APPROVE", result.rewritten_text)

    def test_alert_emoji_handling(self):
        """Test that 🚨 is only added when source had alert keywords."""
        with patch.object(self.rewriter, '_llm_generate', return_value="FC Barcelona secure victory in match."):
            # Test with alert keyword
            result_with_alert = self.rewriter.rewrite("🚨 BREAKING: Barcelona news", "test_source")
            self.assertTrue(result_with_alert.success)
            self.assertTrue(result_with_alert.rewritten_text.startswith("🚨"))

            # Test without alert keyword
            result_without_alert = self.rewriter.rewrite("Barcelona news", "test_source")
            self.assertTrue(result_without_alert.success)
            self.assertFalse(result_without_alert.rewritten_text.startswith("🚨"))

    def test_source_attribution_cas_a(self):
        """Test Cas A attribution with explicit source mention in text."""
        with patch.object(self.rewriter, '_llm_generate', return_value="FC Barcelona sign new player."):
            result = self.rewriter.rewrite("Barcelona news via @FabrizioRomano", "test_source")
            self.assertTrue(result.success)
            self.assertIn("[@FabrizioRomano]", result.rewritten_text)

    def test_source_attribution_cas_b(self):
        """Test Cas B attribution with tier-1 journalist author."""
        with patch.object(self.rewriter, '_llm_generate', return_value="FC Barcelona sign new player."):
            result = self.rewriter.rewrite("Barcelona news today", "FabrizioRomano")
            self.assertTrue(result.success)
            self.assertIn("[@FabrizioRomano]", result.rewritten_text)

    def test_source_attribution_no_attribution(self):
        """Test no handle attribution when author is an aggregator (not tier-1) and no explicit source."""
        with patch.object(self.rewriter, '_llm_generate', return_value="FC Barcelona sign new player."):
            result = self.rewriter.rewrite("Barcelona news today", "BarcaUniversal")
            self.assertTrue(result.success)
            self.assertNotIn("[@BarcaUniversal]", result.rewritten_text)
            self.assertTrue(result.rewritten_text.endswith("🔵🔴"))

    def test_fallback_mechanism_empty_llm(self):
        """Test fallback mechanism when LLM returns empty string."""
        with patch.object(self.rewriter, '_llm_generate', return_value=""):
            result = self.rewriter.rewrite("Barcelona news today", "test_source")
            self.assertTrue(result.success)  # Fallback should succeed
            self.assertNotIn("Rewrite produced empty result", result.error)
            self.assertIn("🔵🔴", result.rewritten_text)

    def test_fallback_for_non_english(self):
        """Test fallback for non-English characters in LLM output."""
        with patch.object(self.rewriter, '_llm_generate', return_value="Texto en español sin ingles"):
            with patch.object(self.rewriter, '_is_english_only', return_value=False):
                result = self.rewriter.rewrite("Barcelona news today", "test_source")
                self.assertTrue(result.success)  # Should use fallback rewrite
                self.assertNotIn("Texto en español", result.rewritten_text)

    def test_fallback_for_security_violations(self):
        """Test fallback when LLM output contains security banned words."""
        with patch.object(self.rewriter, '_llm_generate', return_value="Barcelona: SKIP this game"):
            result = self.rewriter.rewrite("Barcelona news today", "test_source")
            self.assertTrue(result.success)  # Fallback handles it cleanly
            self.assertNotIn("SKIP", result.rewritten_text)


if __name__ == "__main__":
    unittest.main()