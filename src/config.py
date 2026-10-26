"""
Configuration loader with validation using Pydantic.
Loads config.yaml and merges with environment variables.
"""
import os
import re
from pathlib import Path
from typing import Any, List, Dict, Optional, Pattern
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

# Load .env file
from dotenv import load_dotenv
load_dotenv()


class TwitterConfig(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""

    @model_validator(mode="before")
    @classmethod
    def load_from_env(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        fields = ["api_key", "api_secret", "access_token", "access_token_secret"]
        for field in fields:
            if field not in data or not data[field] or str(data[field]).startswith("your_"):
                env_value = os.getenv(f"TWITTER_{field.upper()}")
                if env_value:
                    data[field] = env_value
        return data


class XWebConfig(BaseModel):
    """Browser session used to publish as @UniversCule (free, no API write)."""
    state_file: str = "data/x_universcule_state.json"
    profile_dir: str = "data/x_universcule_profile"
    expected_handle: str = "UniversCule"
    headless: bool = True
    timeout_seconds: int = 45


class XReaderConfig(BaseModel):
    """Read-only browser session used to poll public X profiles and threads."""
    state_file: str = "data/x_reader_state.json"
    headless: bool = True
    timeout_seconds: int = 45


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout_seconds: int = 60
    temperature: float = 0.75


class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.82
    batch_size: int = 32

    @field_validator("similarity_threshold")
    @classmethod
    def check_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return v


class TranslationConfig(BaseModel):
    """Dedicated local translator used for Arabic sources before Ollama."""
    enabled: bool = True
    model_name: str = "facebook/nllb-200-distilled-600M"
    max_new_tokens: int = 220
    num_beams: int = 3


class SourcesConfig(BaseModel):
    accounts: List[str] = Field(default_factory=list)
    rss_feeds: Dict[str, str] = Field(default_factory=dict)


class PollingConfig(BaseModel):
    interval_minutes: int = 10
    lookback_minutes: int = 45
    max_results_per_source: int = 20
    max_candidates_per_cycle: int = 10

    @field_validator("interval_minutes", "lookback_minutes", "max_results_per_source",
                     "max_candidates_per_cycle")
    @classmethod
    def check_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class BarcaKeywordsConfig(BaseModel):
    players: List[str] = Field(default_factory=list)
    club: List[str] = Field(default_factory=list)
    competitions: List[str] = Field(default_factory=list)
    staff: List[str] = Field(default_factory=list)

    def all_keywords(self) -> List[str]:
        return self.players + self.club + self.competitions + self.staff

    def compiled_patterns(self) -> List[Pattern]:
        """Compile all keywords as case-insensitive regex patterns with word boundaries."""
        patterns = []
        for kw in self.all_keywords():
            # Escape special regex chars, allow spaces
            escaped = re.escape(kw).replace(r"\ ", r"\s+")
            patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        return patterns


class BlacklistConfig(BaseModel):
    keywords: List[str] = Field(default_factory=list)

    def compiled_patterns(self) -> List[Pattern]:
        patterns = []
        for kw in self.keywords:
            escaped = re.escape(kw).replace(r"\ ", r"\s+")
            patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        return patterns


class SchedulingConfig(BaseModel):
    active_hours_start: int = 8
    active_hours_end: int = 23
    timezone: str = "Europe/Madrid"
    max_tweets_per_day: int = 80
    # The remaining slots are reserved for replies when replies are enabled.
    original_tweets_per_day: int = 80
    min_interval_minutes: int = 15
    max_interval_minutes: int = 45
    jitter_minutes: int = 5

    @model_validator(mode="after")
    def check_hours(self) -> "SchedulingConfig":
        if not (0 <= self.active_hours_start < self.active_hours_end <= 24):
            raise ValueError("Invalid active hours: start < end, both 0-24")
        if self.min_interval_minutes < 0 or self.max_interval_minutes < 0:
            raise ValueError("interval_minutes cannot be negative")
        if self.min_interval_minutes > self.max_interval_minutes:
            raise ValueError("min_interval_minutes must be <= max_interval_minutes")
        if self.max_tweets_per_day > 80:
            raise ValueError("max_tweets_per_day cannot exceed 80")
        if self.original_tweets_per_day > self.max_tweets_per_day:
            raise ValueError("original_tweets_per_day cannot exceed max_tweets_per_day")
        return self


class RepliesConfig(BaseModel):
    enabled: bool = True
    check_interval_minutes: int = 5
    min_delay_minutes: int = 20
    max_delay_minutes: int = 30
    relevance_threshold: float = 0.6
    language: str = "fr"
    max_replies_per_day: int = 15
    max_tweets_to_check: int = 8
    max_age_hours: int = 48
    min_comment_length: int = 4

    @field_validator("relevance_threshold")
    @classmethod
    def check_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        return v

    @field_validator("max_replies_per_day", "max_tweets_to_check", "max_age_hours")
    @classmethod
    def check_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class LengthConfig(BaseModel):
    max_chars: int = 280

    @field_validator("max_chars")
    @classmethod
    def check_max_chars(cls, v: int) -> int:
        if v > 280:
            raise ValueError("max_chars cannot exceed 280 (Twitter limit)")
        if v <= 0:
            raise ValueError("max_chars must be positive")
        return v


class DeduplicationConfig(BaseModel):
    content_similarity_threshold: float = 0.82
    time_window_hours: int = 48
    exact_hash_check: bool = True

    @field_validator("content_similarity_threshold")
    @classmethod
    def check_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("content_similarity_threshold must be between 0 and 1")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/bot.log"
    max_bytes: int = 10485760
    backup_count: int = 5
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


class DatabaseConfig(BaseModel):
    path: str = "data/bot.db"


class MatchCalendarConfig(BaseModel):
    enabled: bool = True
    refresh_hours: int = 6
    cache_file: str = "data/barca_matches.json"
    before_minutes: int = 15
    after_minutes: int = 45
    timezone: str = "Europe/Madrid"


class MediaConfig(BaseModel):
    enabled: bool = True
    max_size_mb: int = 5
    max_video_size_mb: int = 50
    timeout_seconds: int = 60
    cache_dir: str = "data/media_cache"
    max_images_per_tweet: int = 4
    max_media_per_tweet: int = 4

    @field_validator("max_images_per_tweet")
    @classmethod
    def check_images(cls, v: int) -> int:
        if not 1 <= v <= 4:
            raise ValueError("X accepts at most 4 images per tweet")
        return v


class Config(BaseModel):
    twitter: TwitterConfig = Field(default_factory=TwitterConfig)
    x_web: XWebConfig = Field(default_factory=XWebConfig)
    x_reader: XReaderConfig = Field(default_factory=XReaderConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    barca_keywords: BarcaKeywordsConfig = Field(default_factory=BarcaKeywordsConfig)
    blacklist_keywords: BlacklistConfig = Field(default_factory=BlacklistConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    replies: RepliesConfig = Field(default_factory=RepliesConfig)
    length: LengthConfig = Field(default_factory=LengthConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    match_calendar: MatchCalendarConfig = Field(default_factory=MatchCalendarConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)

    @property
    def twitter_credentials(self) -> Dict[str, str]:
        """Get Twitter credentials, raising if missing."""
        creds = {
            "api_key": self.twitter.api_key,
            "api_secret": self.twitter.api_secret,
            "access_token": self.twitter.access_token,
            "access_token_secret": self.twitter.access_token_secret,
        }
        missing = [k for k, v in creds.items() if not v or v.startswith("your_")]
        if missing:
            raise ValueError(f"Missing Twitter credentials: {missing}. Set in .env or config.yaml")
        return creds


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and merge with environment variables.
    Environment variables take precedence over YAML.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    # Create config from YAML, env vars will be picked up by Pydantic validators
    config = Config(**yaml_data)
    return config


# Global config instance (loaded on first import)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance (lazy load)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config