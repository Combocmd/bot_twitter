"""
Tests unitaires pour le rewriter - verification des corrections v1.1.9
"""
import sys
sys.path.insert(0, r"D:\projets\bot_twitter")

from src.rewriter import TweetRewriter
from unittest.mock import MagicMock, patch


def test_error_not_none_empty_response():
    """Test que last_error n'est jamais None quand la reponse est vide"""
    print("\n=== TEST 1: last_error ne doit jamais etre None (reponse vide) ===")
    
    rewriter = TweetRewriter()
    
    # Mock _chat pour retourner une chaine vide (simulation d'echech Ollama)
    with patch.object(rewriter, '_chat', return_value=""):
        result = rewriter.rewrite("Test tweet", "testuser", "en")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        assert result.error is not None, f"ERROR: last_error est None! Error = '{result.error}'"
        assert "vide" in result.error.lower(), f"ERROR: Message d'erreur inadapté: {result.error}"
        print("  PASS: Error message present et non-None")


def test_error_not_none_invalid_candidate():
    """Test que last_error n'est jamais None quand le candidat est invalide"""
    print("\n=== TEST 2: last_error ne doit jamais etre None (candidat invalide) ===")
    
    rewriter = TweetRewriter()
    
    # Mock _chat pour retourner un texte invalide (trop court)
    with patch.object(rewriter, '_chat', return_value="ok"):
        result = rewriter.rewrite("Test tweet beaucoup plus long", "testuser", "en")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        assert result.error is not None, f"ERROR: last_error est None! Error = '{result.error}'"
        print("  PASS: Error message present et non-None")


def test_anglicism_detection_french_context():
    """Test que les anglicismes sont detectes dans un contexte français"""
    print("\n=== TEST 3: Detection des anglicismes (contexte FR) ===")
    
    rewriter = TweetRewriter()
    
    # Tweet en français mais avec un anglicisme
    test_tweet = "Le joueur a montre un grand sportsmanship lors du match"
    
    # Mock _chat pour simuler une reponse avec anglicisme
    with patch.object(rewriter, '_chat', return_value=test_tweet):
        result = rewriter.rewrite(test_tweet, "testuser", "fr")
        print(f"  Original: {test_tweet[:50]}...")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        
        # L'auto-correction devrait detecter l'anglicisme
        assert result.error is not None, "ERROR: Anglicisme non detecte!"
        # Le message d'erreur doit mentionner sportsmanship ou etre coherent
        print("  PASS: Anglicisme detecte")


def test_phrase_incoherente():
    """Test detection de phrases incoherentes"""
    print("\n=== TEST 4: Detection de phrases incoherentes ===")
    
    rewriter = TweetRewriter()
    
    # Phrase incoherente
    test_tweet = "Le Barcelona ont perdu le match avec beaucoup de sportsmanship"
    
    with patch.object(rewriter, '_chat', return_value=test_tweet):
        result = rewriter.rewrite(test_tweet, "testuser", "fr")
        print(f"  Original: {test_tweet[:60]}...")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        
        # Verifier que l'erreur est captures
        assert result.error is not None, "ERROR: Probleme non detecte!"
        print("  PASS: Phrase incoherente detectee")


def test_version():
    """Test version du bot"""
    print("\n=== TEST 5: Version ===")
    
    from src import __version__
    print(f"  Version actuelle: {__version__}")
    assert __version__ == "1.1.9", f"ERROR: Version attendue 1.1.9, obtenu {__version__}"
    print("  PASS: Version correcte")


def test_rollback_behavior():
    """Test que le bot rollback correctement en cas d'echech repetes"""
    print("\n=== TEST 6: Comportement rollback ===")
    
    rewriter = TweetRewriter()
    
    # Simuler 3 echecs consecutifs
    call_count = [0]
    def mock_chat_fail(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] >= 3:
            return ""  # Echec apres 3 tentatives
        return "response"
    
    with patch.object(rewriter, '_chat', side_effect=mock_chat_fail):
        result = rewriter.rewrite("Test tweet", "testuser", "en")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        print(f"  Calls: {call_count[0]}")
        
        # Doit avoir essaie 3 fois et retourné une erreur
        assert result.error is not None, "ERROR: Pas d'erreur apres echecs multiples!"
        assert call_count[0] == 3, f"ERROR: Expected 3 calls, got {call_count[0]}"
        print("  PASS: Rollback correct apres 3 echecs")


def run_all_tests():
    """Executer tous les tests"""
    print("\n" + "="*60)
    print("TESTS UNITAIRES - Bot Twitter v1.1.9")
    print("="*60)
    
    tests = [
        test_error_not_none_empty_response,
        test_error_not_none_invalid_candidate,
        test_anglicism_detection_french_context,
        test_phrase_incoherente,
        test_version,
        test_rollback_behavior,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTATS: {passed} reussis, {failed} echoues")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
