"""
Barcelona Twitter Bot - Main package.
"""
__version__ = "1.0.1"
__author__ = "UniversCule Bot"

from .config import load_config, get_config, Config
from .database import Database
from .twitter_client import TwitterClient, TweetData, UserData, TwitterError, RateLimitError, AuthError, MediaError
from .monitor import SourceMonitor
from .filter import BarcaFilter
from .dedup import ContentDeduplicator
from .rewriter import TweetRewriter
from .scheduler import PostScheduler
from .reply_manager import ReplyManager

__all__ = [
    "load_config",
    "get_config",
    "Config",
    "Database",
    "TwitterClient",
    "TweetData",
    "UserData",
    "TwitterError",
    "RateLimitError",
    "AuthError",
    "MediaError",
    "SourceMonitor",
    "BarcaFilter",
    "ContentDeduplicator",
    "TweetRewriter",
    "PostScheduler",
    "ReplyManager",
]