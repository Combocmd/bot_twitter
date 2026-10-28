import pytest
from src.rewriter import TweetRewriter
from src.config import get_config

@pytest.fixture
def rewriter():
    return TweetRewriter(get_config())

# Tests de base pour le rewriter

def test_rewrite_english(rewriter):
    result = rewriter.rewrite(
        "Real Madrid are reportedly interested in signing Barcelona wonderkid, sources say.",
        "test"
    )
    assert result.success
    assert "Real Madrid" in result.rewritten_text
    assert "Barça" in result.rewritten_text
    assert result.language_detected == "en"


def test_rewrite_spanish(rewriter):
    result = rewriter.rewrite(
        "El Barcelona busca un nuevo extremo para la próxima temporada.",
        "test"
    )
    assert result.success
    assert "Barça" in result.rewritten_text
    assert "ailier" in result.rewritten_text
    assert result.language_detected == "es"


def test_rewrite_arabic(rewriter):
    result = rewriter.rewrite(
        "بارسا يبحث عن لاعب جديد من إسبانيا",
        "test"
    )
    assert result.success
    assert "Barça" in result.rewritten_text
    assert "Espagne" in result.rewritten_text
    assert result.language_detected == "ar"


def test_rewrite_french(rewriter):
    result = rewriter.rewrite(
        "Le Barça recrute un nouveau milieu défensif pour la saison prochaine.",
        "test"
    )
    assert result.success
    assert "Barça" in result.rewritten_text
    assert "milieu défensif" in result.rewritten_text
    assert result.language_detected == "fr"


def test_rewrite_fails_on_empty(rewriter):
    result = rewriter.rewrite("", "test")
    assert not result.success
    assert result.error == "texte source vide"


def test_rewrite_protects_terms(rewriter):
    result = rewriter.rewrite(
        "Kopa Trophy awarded to Barcelona's best player",
        "test"
    )
    assert result.success
    assert "Kopa Trophy" in result.rewritten_text
    assert "Barça" in result.rewritten_text
    assert "meilleur joueur" in result.rewritten_text


def test_rewrite_rejects_english_leaks(rewriter):
    result = rewriter.rewrite(
        "Barcelona's new striker is a young English player",
        "test"
    )
    assert not result.success
    assert "anglicisme" in result.error.lower()


def test_rewrite_rejects_arabic_leaks(rewriter):
    result = rewriter.rewrite(
        "بارسا تواجه تحديات كبيرة في الدوري الإسباني",
        "test"
    )
    assert result.success
    assert "défis" in result.rewritten_text.lower()
    assert "liga" in result.rewritten_text.lower()


def test_rewrite_rejects_nonsense_french(rewriter):
    result = rewriter.rewrite(
        "Le Barça a un hélicoptère dans son vestiaire",
        "test"
    )
    assert not result.success
    assert "mot inventé" in result.error.lower()


def test_rewrite_rejects_elision_errors(rewriter):
    result = rewriter.rewrite(
        "L' égalisation a été marqué par un jeune joueur",
        "test"
    )
    assert result.success
    assert "l'égalisation" in result.rewritten_text
    assert "l' égalisation" not in result.rewritten_text
