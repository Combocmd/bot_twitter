# 🤖 Barcelona Twitter Bot - @UniversCule

**Version 1.0.1** — voir [CHANGELOG.md](CHANGELOG.md).

Bot Twitter autonome pour le média football **@UniversCule** (FC Barcelona).  
Surveille 7 comptes sources, réécrit l'info en français naturel, poste avec médias, répond aux commentaires pertinents.

## ✨ Fonctionnalités

| Fonction | Description |
|----------|-------------|
| **Veille multi-sources** | Polling 7 comptes toutes les 1 min |
| **Détection Barça only** | Mots-clés joueurs, club, compétitions, staff |
| **Dédoublonnage sémantique** | Embeddings locales (sentence-transformers) - évite les doublons cross-sources |
| **Réécriture humaine** | Ollama local (qwen2.5:7b) - style journaliste/supporter, traduction auto → FR |
| **Support médias** | Télécharge et re-upload images des tweets sources |
| **Planification humaine** | 8h-23h, 80 max/jour, publication immédiate |
| **Réponses intelligentes** | Filtre pertinence + bots, délai 20-30 min, FR uniquement |
| **100% Gratuit & Local** | Aucune API payante, tout tourne sur votre machine |

## 🏗 Architecture

```
src/
├── config.py          # Configuration centralisée (YAML + env)
├── database.py        # SQLite (tweets, sources, commentaires, médias, stats)
├── twitter_client.py  # Wrapper API v2 (rate limit, retry, media upload)
├── monitor.py         # Polling sources (1 min)
├── filter.py          # Détection contenu Barça (regex + contexte)
├── dedup.py           # Dédoublonnage par embeddings (cosine similarity)
├── rewriter.py        # Réécriture Ollama + limite 280 chars
├── scheduler.py       # Queue posting + timing humain (APScheduler)
├── reply_manager.py   # Surveillance mentions + réponses pertinentes
└── main.py            # Orchestration + CLI (start, test, status, stats)
```

## 📋 Prérequis

- **Python 3.10+**
- **Ollama** (pour le LLM local) - [ollama.com/download](https://ollama.com/download)
- **Compte Twitter Developer** (API v2, OAuth 1.0a, Read+Write)

## 🚀 Installation

```bash
# 1. Cloner / aller dans le dossier
cd D:\projets\bot_twitter

# 2. Lancer le setup automatique
python setup.py
#   → Crée .venv, installe deps, installe Ollama (Windows), pull qwen2.5:7b

# 3. Configurer les clés Twitter
# Éditez .env avec vos 4 clés (depuis developer.twitter.com)
```

## ⚙️ Configuration

### .env (obligatoire)
```env
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### config.yaml (personnalisable)
- `sources.accounts` : comptes à surveiller
- `barca_keywords` : mots-clés détection (joueurs, club, compétitions)
- `scheduling` : horaires, volume, intervalles
- `replies` : délais, seuils pertinence
- `deduplication` : seuil similarité (0.82), fenêtre temporelle

## 🎮 Utilisation

```bash
# Activer l'environnement
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac

# Test complet (dry-run - ne poste rien)
python -m src.main test

# Vérifier la config
python -m src.main status

# Voir les stats
python -m src.main stats

# Test rewriter sur un texte
python -m src.main test-rewrite "Barcelona sign new striker!"

# Test filtre Barça
python -m src.main test-filter "Real Madrid win 3-0"

# Démarrer le bot (daemon)
python -m src.main start

# Arrêt propre : Ctrl+C
```

## 🔄 Comment ça marche

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  POLLING    │───▶│   FILTRE    │───▶│  DEDUP      │───▶│  REWRITE    │
│   (1 min)   │    │  Barça only │    │  Sémantique │    │  (Ollama)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                      │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐                ▼
│  REPLIES    │◀───│  SCHEDULER  │◀───│   QUEUE     │    ┌─────────────┐
│  (20-30 min)│    │  Humain     │    │  (8h-23h)   │◀───│  POST TWEET │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

1. **Polling** : Récupère les nouveaux tweets des 7 sources (1 min, sources < 2 min)
2. **Filtre** : Garde seulement le contenu Barça (mots-clés + contexte)
3. **Mode match** : Détecté via le calendrier Barça (cache 6 h, `Europe/Madrid`) — pendant un match, seuls les tweets avec image/vidéo passent
4. **Dedup source** : Compare embeddings + texte source — rejette si > 82 % similaire à 48 h
5. **Rewrite** : Ollama reformule en français ; glossaire football, rejet des anglicismes, des scores/chiffres inventés et des élisions cassées ; deuxième tentative puis rejet
6. **Dedup événement** : Compare le texte réécrit aux tweets en file/publiés — même événement = un seul post ; une version avec média remplace la version sans média encore en file
7. **Media** : Télécharge images/vidéos sources → re-upload Twitter → attache au tweet
8. **Schedule** : Place en file avec timing humain (jitter, limites journalières)
9. **Post** : Worker APScheduler poste aux créneaux calculés ; session `@UniversCule` vérifiée avant chaque post, pause + arrêt de l'ingestion si elle est invalide. Jamais plus d'un post par minute (protection anti-burst), et si X renvoie sa limite quotidienne, la file est reportée à la réinitialisation sans perdre les tweets ni consommer d'essai.
10. **Replies** : Surveille les mentions sur NOS tweets → filtre pertinence → répond 20-30 min après

### Qualité éditoriale

Avant toute mise en file, un tweet doit passer :

- français correct, sans anglais résiduel (`the`, `goal`, `substitution`, `enters`…) ;
- vocabulaire football français (`égalise`, `remplacement`, `entre en jeu`, `sort`) ;
- aucun chiffre, score ou minute absent de la source ;
- pas d'élision cassée (`l' 2e`, `l'2e`) ;
- pas de doublon d'événement avec la file ou les publications récentes ;
- média annoncé mais non téléchargeable → tweet ignoré (jamais de texte seul à la place).

Si une règle échoue : nouvelle réécriture. Si la deuxième tentative échoue aussi, le tweet est rejeté (mieux vaut perdre un tweet que publier une erreur). Le motif exact est écrit dans les logs.

## 🛠 Personnalisation

### Ajouter des comptes sources
```yaml
# config.yaml
sources:
  accounts:
    - "Forca_Fcb"
    - "NouveauCompte"
```

### Modifier le style de réécriture
Éditez `src/rewriter.py` → `SYSTEM_PROMPT` et `FEW_SHOT_EXAMPLES`

### Ajuster la détection Barça
```yaml
# config.yaml
barca_keywords:
  players: [...]
  club: [...]
  competitions: [...]
```

### Changer le modèle Ollama
```yaml
# config.yaml
ollama:
  model: "llama3.1:8b"  # ou mistral:7b, qwen2.5:14b, etc.
```

## 📊 Monitoring & Logs

- **Logs** : `logs/bot.log` (rotation 10MB, 5 fichiers)
- **Stats journalières** : table `bot_stats` en base
- **CLI** : `python -m src.main status` / `stats`

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| Ollama non trouvé | `setup.py` l'installe, ou installez manuellement |
| Modèle non pullé | `ollama pull qwen2.5:7b` |
| Rate limit Twitter | Le bot gère automatiquement (backoff exponentiel) |
| Médias non uploadés | Vérifiez taille < 5MB, format supporté |
| Réponses pas envoyées | Vérifiez `replies.enabled: true`, logs pour debug |
| Erreur embedding | `pip install --upgrade sentence-transformers torch` |

## 📁 Structure des données

```
data/
├── bot.db              # SQLite (tweets, sources, replies, media, stats)
└── media_cache/        # Cache local médias téléchargés

logs/
└── bot.log             # Logs rotatifs
```

## 🔐 Sécurité

- **Jamais** committer `.env` (dans .gitignore)
- Clés Twitter en variables d'environnement seulement
- Base SQLite locale, pas de cloud
- Ollama 100% local, aucune donnée ne sort

## 📝 Licence

MIT - Libre d'utilisation et modification.

---

**Créé pour @UniversCule** - *Visca el Barça!* 🔵🔴