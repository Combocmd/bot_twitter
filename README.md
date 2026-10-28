# 🤖 Barcelona Twitter Bot - @UniverseCule

**Version 1.2.0** — International English edition for @UniverseCule.

Autonomous Twitter bot for FC Barcelona media **@UniverseCule**. Monitors 8 Barça sources, rewrites content in natural English, posts with media. No auto-replies, manual control only.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-source monitoring** | Polls 8 accounts every 1 minute |
| **Barça-only detection** | Keywords for players, club, competitions, staff |
| **Semantic deduplication** | Local embeddings (sentence-transformers) - prevents cross-source duplicates |
| **Human-like rewriting** | Local Ollama (qwen2.5:7b) - professional journalist style, auto-translation to English |
| **Media support** | Downloads and re-uploads images from source tweets |
| **Human scheduling** | 24/7 (configurable window), 80 max/day, immediate publishing |
| **100% Free & Local** | No paid API, everything runs on your machine |
| **Manual operation** | Start/stop manually via `demarrer.bat` / `arreter.bat` |

## 🏗 Architecture

```
src/
├── config.py          # Centralized config (YAML + env)
├── database.py        # SQLite (tweets, sources, media, stats)
├── twitter_client.py  # API v2 wrapper (rate limit, retry, media upload)
├── monitor.py         # Polling sources (1 min)
├── filter.py          # Barça content detection (regex + context)
├── dedup.py           # Deduplication via embeddings (cosine similarity)
├── rewriter.py        # Ollama rewriting + 280 char limit + fact-checking
├── scheduler.py       # Queue posting + human timing (APScheduler)
└── main.py            # Orchestration + CLI (start, test, status, stats)
```

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM) - [ollama.com/download](https://ollama.com/download)
- **Twitter Developer Account** (API v2, OAuth 1.0a, Read+Write)
- **Brave/Chrome** (for session export)

## 🚀 Installation

```bash
# 1. Clone / go to folder
cd D:\projets\bot_twitter

# 2. Run automatic setup
python setup.py
#   → Creates .venv, installs deps, installs Ollama (Windows), pulls qwen2.5:7b

# 3. Configure Twitter keys
# Edit .env with your 4 keys (from developer.twitter.com)
```

## ⚙️ Configuration

### .env (required)
```env
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TEMPERATURE=0.20
OLLAMA_TIMEOUT_SECONDS=45
```

### config.yaml (customizable)
- `sources.accounts` : accounts to monitor
- `barca_keywords` : detection keywords (players, club, competitions, staff)
- `scheduling` : hours, volume, intervals
- `deduplication` : similarity threshold (0.82), time window
- `ollama.temperature` : 0.20 for less hallucination

## 🎮 Usage

### Quick Start (Windows)

| Action | How |
|--------|-----|
| **Start** | Double-click `demarrer.bat` |
| **Stop** | Double-click `arreter.bat` (or Ctrl+C in window) |

The bot window must stay open while running. If an instance is already running, `demarrer.bat` refuses to start a second one and shows its PID.

### Commands

```bash
# Activate environment
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac

# Full test (dry-run - posts nothing)
python -m src.main test

# Check config & sessions
python -m src.main status

# View stats
python -m src.main stats

# Test rewriter on text
python -m src.main test-rewrite "Barcelona sign new striker!"

# Test Barça filter
python -m src.main test-filter "Real Madrid win 3-0"

# Start bot (daemon)
python -m src.main start

# Clean stop: Ctrl+C
```

## 🔄 How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  POLLING    │───▶│   FILTER    │───▶│  DEDUP      │───▶│  REWRITE    │
│   (1 min)   │    │  Barça only │    │  Semantic   │    │  (Ollama)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                       │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐                ▼
│  QUEUE      │◀───│  SCHEDULER  │◀───│   POST      │    ┌─────────────┐
│  (24h/24)   │    │  Human      │    │  TWEET      │◀───│  MEDIA      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

1. **Polling**: Fetches new tweets from 8 sources (1 min, sources < 2 min old)
2. **Filter**: Keeps only Barça content (keywords + football context)
3. **Match mode**: Detected via Barça calendar (6h cache, `Europe/Madrid`) — during matches, only tweets with media pass
4. **Source dedup**: Compares embeddings + source text — rejects if > 82% similar within 48h
5. **Rewrite**: Ollama reformulates in English; protected terms (player names, trophies), fact-checking (numbers, scores), anti-plagiarism, post-corrections; second attempt on failure then reject
6. **Event dedup**: Compares rewritten text against queued/published tweets — same event = single post; version with media replaces media-less version in queue
7. **Media**: Downloads source images/videos → re-uploads to X → attaches to tweet
8. **Schedule**: Queues with human timing (jitter, daily limits, min 60s between posts)
9. **Post**: APScheduler worker posts at calculated slots; @UniverseCule session verified before each post; pauses ingestion if invalid. Hard limit 80/day, max 1 post/minute. Daily limit hit → queue deferred to reset without losing tweets.

**No auto-replies** - replies are manual only.

### Editorial Quality Gates

Before queueing, every tweet must pass:

- ✅ Correct English, no residual foreign words
- ✅ Professional football terminology (brace, equaliser, finalize acquisition, touchline, bench)
- ✅ No invented numbers, scores, or minutes not in source
- ✅ Protected terms preserved (player names, trophies, BREAKING/JUST IN/OFFICIAL/EXCLUSIVE)
- ✅ No event duplicate with queue or recent publications
- ✅ Announced media fails download → tweet rejected (never text-only fallback)

If a gate fails: second rewrite attempt. If second attempt fails, tweet rejected (better to lose a tweet than publish an error). Exact reason logged.

## 🛠 Customization

### Add Source Accounts
```yaml
# config.yaml
sources:
  accounts:
    - "Forca_Fcb"
    - "NewAccount"
```

### Adjust Rewrite Style
Edit `src/rewriter.py` → `ENGLISH_REWRITE_SYSTEM` and `_PROTECTED_TERMS`

### Tune Barça Detection
```yaml
# config.yaml
barca_keywords:
  players: [...]
  club: [...]
  competitions: [...]
  staff: [...]
```

### Change Ollama Model
```yaml
# config.yaml
ollama:
  model: "llama3.1:8b"  # or mistral:7b, qwen2.5:14b, etc.
```

## 📊 Monitoring & Logs

- **Logs**: `logs/bot.log` (rotation 10MB, 5 files)
- **Daily stats**: `bot_stats` table in database
- **CLI**: `python -m src.main status` / `stats`

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not found | `setup.py` installs it, or install manually |
| Model not pulled | `ollama pull qwen2.5:7b` |
| Rate limit X | Bot handles automatically (exponential backoff) |
| Media not uploading | Check size < 5MB, supported format |
| Embedding error | `pip install --upgrade sentence-transformers torch` |
| Writer session invalid | Log into @UniverseCule in Brave/Chrome, export session to `data/x_universecule_state.json` |

## 📁 Data Structure

```
data/
├── bot.db              # SQLite (tweets, sources, media, stats)
├── media_cache/        # Local media download cache
├── x_reader_state.json # Read-only session (auto-imported from Firefox)
└── x_universecule_state.json  # Writer session (manual export from Brave/Chrome)

logs/
└── bot.log             # Rotating logs
```

## 🔐 Security

- **Never** commit `.env` (in .gitignore)
- Twitter keys in environment variables only
- Local SQLite database, no cloud
- Ollama 100% local, no data leaves your machine
- Browser sessions stored locally only

## 📝 License

MIT - Free to use and modify.

---

**Built for @UniverseCule** - *Visca el Barça!* 💙❤️