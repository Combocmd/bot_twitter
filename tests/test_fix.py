"""
Test simple pour verifier le fix v1.1.9 - last_error ne doit jamais etre None
"""
import sys
sys.path.insert(0, r"D:\projets\bot_twitter")

from src.rewriter import TweetRewriter
from unittest.mock import patch


def main():
    print("="*60)
    print("TEST - Correction erreur 'réécriture impossible: None'")
    print("="*60)
    
    rewriter = TweetRewriter()
    
    # Test 1: Simulation d'Ollama qui retourne vide
    print("\n[TEST 1] Ollama retourne une chaine vide...")
    with patch.object(rewriter, '_chat', return_value=""):
        result = rewriter.rewrite("Test tweet", "testuser", "en")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        
        if result.error is None:
            print("  [FAIL] Error est toujours None!")
            return False
        elif "None" in result.error:
            print("  [FAIL] Error contient encore 'None'!")
            return False
        else:
            print("  [PASS] Error est maintenant un message valide")
    
    # Test 2: Simulation d'erreur Ollama
    print("\n[TEST 2] Ollama lance une exception...")
    with patch.object(rewriter, '_chat', side_effect=Exception("Connection refused")):
        result = rewriter.rewrite("Test tweet", "testuser", "en")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        
        if result.error is None:
            print("  [FAIL] Error est toujours None!")
            return False
        elif "Connection refused" in result.error:
            print("  [PASS] Error capture l'exception correctement")
        else:
            print(f"  [INFO] Error: {result.error}")
    
    # Test 3: Verification version
    print("\n[TEST 3] Verification version...")
    from src import __version__
    print(f"  Version: {__version__}")
    if __version__ != "1.1.9":
        print("  [FAIL] Version incorrecte!")
        return False
    print("  [PASS] Version 1.1.9")
    
    print("\n" + "="*60)
    print("TOUS LES TESTS SONT REUSSIS!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
