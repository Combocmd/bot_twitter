"""Vision analysis for images and videos using Gemini Vision API (primary) and Ollama local models (fallback).

Supports:
- Image analysis: player recognition, text extraction, scene description
- Video analysis: key frame extraction + analysis
- Integration with Barça filter/rewriter for enhanced importance scoring
"""

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import ollama
import requests
from loguru import logger

from .config import get_config
from .twitter_client import TwitterClient

# Optional Gemini import (new google-genai SDK)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


@dataclass
class VisionResult:
    """Result of vision analysis."""
    media_url: str
    media_type: str  # "image" or "video"
    description: str = ""
    players_detected: List[str] = field(default_factory=list)
    text_extracted: str = ""
    scene_type: str = ""  # "training", "match", "press_conference", "graphic", "meme", "celebration", "unknown"
    is_barca_related: bool = False
    confidence: float = 0.5  # 0-1, confidence in the analysis
    importance_boost: float = 0.0  # 0-1, added to filter importance
    key_facts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    analysis_time_ms: int = 0


@dataclass
class FrameInfo:
    """Info about extracted video frame."""
    path: str
    timestamp_sec: float
    index: int


class VisionAnalyzer:
    """Analyzes images and videos using local Ollama vision models."""

    # Barça player names for detection
    BARCA_PLAYERS = {
        "Yamal", "Lamine Yamal", "Pedri", "Gavi", "De Jong", "Frenkie de Jong",
        "Lewandowski", "Robert Lewandowski", "Raphinha", "Kounde", "Jules Kounde",
        "Araujo", "Ronald Araujo", "Balde", "Alejandro Balde", "Ter Stegen",
        "Marc-Andre ter Stegen", "Fermin", "Fermin Lopez", "Olmo", "Dani Olmo",
        "Torres", "Ferran Torres", "Fati", "Ansu Fati", "Garcia", "Eric Garcia",
        "Martinez", "Inigo Martinez", "Christensen", "Andreas Christensen",
        "Pena", "Inaki Pena", "Szczesny", "Wojciech Szczesny", "Fort", "Hector Fort",
        "Casado", "Marc Casado", "Bernal", "Marc Bernal", "Torre", "Pablo Torre",
        "Cubarsi", "Pau Cubarsi"
    }

    # Staff
    BARCA_STAFF = {"Flick", "Hansi Flick", "Deco", "Laporta", "Joan Laporta"}

    # Scene type keywords
    SCENE_KEYWORDS = {
        "training": ["training", "entrainement", "session", "exercise", "drill", "gym", "workout", "pitch", "terrain"],
        "match": ["match", "game", "stadium", "camp nou", "montjuic", "kickoff", "lineup", "starting xi", "bench", "substitute"],
        "press_conference": ["press conference", "presser", "media", "interview", "microphone", "podium", "rueda de prensa"],
        "celebration": ["celebrat", "celebration", "hug", "embrace", "joy", "victory", "win", "goal", "scored"],
        "graphic": ["graphic", "infographic", "stat", "statistic", "table", "chart", "formation", "tactic", "lineup graphic"],
        "meme": ["meme", "funny", "edit", "montage", "caption", "lol", "lmao"],
        "merch": ["jersey", "kit", "shirt", "store", "shop", "buy", "merch", "collection", "release"],
    }

    def __init__(self, config=None):
        self.config = config or get_config()
        self.vision_config = self.config.vision
        self.client = ollama.Client(host=self.config.ollama.host)
        self.twitter_client = TwitterClient(self.config)
        self._model_name: Optional[str] = None
        self._model_lock = threading.Lock()
        self._ffmpeg_available = self._check_ffmpeg()
        
        # Gemini Vision API configuration (primary - new google-genai SDK)
        self._gemini_client = None
        self._gemini_available = False
        if GEMINI_AVAILABLE:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                try:
                    # New google-genai SDK: initialize client (default v1 API)
                    self._gemini_client = genai.Client(api_key=gemini_api_key)
                    self._gemini_model = "gemini-flash-lite-latest"
                    self._gemini_available = True
                    logger.info("Gemini Vision API configured successfully (primary enabled) - using google-genai SDK with gemini-flash-lite-latest")
                except Exception as e:
                    logger.warning(f"Failed to configure Gemini Vision API: {e}")
            else:
                logger.info("GEMINI_API_KEY not set, running Ollama-only mode for vision")
        else:
            logger.info("google-genai not installed, running Ollama-only mode for vision")

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available for video frame extraction."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("ffmpeg not found - video analysis will be limited")
            return False

    def _get_model(self) -> str:
        """Get the best available vision model, pulling if needed."""
        if self._model_name:
            return self._model_name

        with self._model_lock:
            if self._model_name:
                return self._model_name

            try:
                # List available models
                result = self.client.list()
                models = result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
                available = []
                for item in models:
                    if isinstance(item, dict):
                        available.append(item.get("name") or item.get("model") or "")
                    else:
                        available.append(getattr(item, "name", "") or getattr(item, "model", "") or "")

                # Find first available vision model from config priority
                for model in self.vision_config.model_priority:
                    if any(model in name for name in available):
                        self._model_name = model
                        logger.info(f"Using vision model: {model}")
                        return model

                # No vision model found, try to pull first from priority
                first_model = self.vision_config.model_priority[0] if self.vision_config.model_priority else "moondream:2b"
                logger.info(f"No vision model found, pulling {first_model}...")
                self.client.pull(first_model)
                self._model_name = first_model
                return self._model_name

            except Exception as e:
                logger.error(f"Failed to get vision model: {e}")
                # Fallback
                first_model = self.vision_config.model_priority[0] if self.vision_config.model_priority else "moondream:2b"
                self._model_name = first_model
                return self._model_name

    def _download_media(self, url: str, timeout: int = 30) -> tuple[bytes, str, str]:
        """Download media from URL. Returns (content, media_type, extension)."""
        return self.twitter_client.download_media(url, timeout=timeout)

    def _extract_frames_from_video(self, video_content: bytes, max_frames: int = None) -> List[FrameInfo]:
        """Extract key frames from video using ffmpeg - optimized for speed with 1 frame."""
        if not self._ffmpeg_available:
            return []

        if max_frames is None:
            max_frames = 1  # Only 1 frame for speed

        frames = []
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                tmp_video.write(video_content)
                video_path = tmp_video.name

            # Get video duration
            duration_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=5)
            try:
                duration = float(result.stdout.strip())
            except ValueError:
                duration = 30.0  # Default assumption

            # Calculate frame timestamp - just middle frame for speed
            timestamp = duration * 0.5

            # Extract single frame
            temp_dir = Path(tempfile.gettempdir()) / "barca_vision_frames"
            temp_dir.mkdir(exist_ok=True)

            frame_path = temp_dir / f"frame_{int(time.time() * 1000)}.jpg"
            cmd = [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                "-vframes", "1", "-q:v", "2", str(frame_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=10)
            if frame_path.exists():
                frames.append(FrameInfo(
                    path=str(frame_path),
                    timestamp_sec=timestamp,
                    index=0
                ))

            # Cleanup video temp file
            try:
                os.unlink(video_path)
            except OSError:
                pass

        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")

        return frames

    def _resize_image_for_vision(self, image_content: bytes) -> bytes:
        """Resize image to max dimension for faster vision processing.
        
        LLaVA and similar models work best with images around 224-336px.
        Larger images waste compute with no accuracy improvement.
        """
        max_size = getattr(self.vision_config, 'max_image_size', 336)
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_content))
            
            # Convert to RGB if needed (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if larger than max_size
            if max(img.width, img.height) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85, optimize=True)
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"Image resize failed, using original: {e}")
            return image_content

    def _analyze_image_gemini(self, image_content: bytes, prompt: str) -> str:
        """Analyze a single image with Gemini Vision API using new google-genai SDK."""
        if not self._gemini_available or not self._gemini_client:
            logger.debug("Gemini Vision not available")
            return ""
        
        try:
            import base64
            import io
            from PIL import Image
            
            # Resize image for faster processing
            resized_content = self._resize_image_for_vision(image_content)
            
            # Convert to PIL Image for google-genai SDK
            img = Image.open(io.BytesIO(resized_content))
            
            # Optimized config from config
            temperature = getattr(self.vision_config, 'temperature', 0.1)
            num_predict = getattr(self.vision_config, 'num_predict', 200)
            
            # Use model name directly as string literal to avoid interpolation issues
            response = self._gemini_client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=num_predict,
                ),
            )
            
            if response and response.text:
                result = response.text.strip()
                logger.debug("Gemini Vision analysis succeeded")
                return result
            else:
                logger.warning("Gemini Vision returned empty response")
                return ""
                
        except Exception as e:
            # Check for HTTP 503 (High Demand / Server Error) and other server errors
            error_msg = str(e)
            if "503" in error_msg or "High Demand" in error_msg or "Server Error" in error_msg:
                logger.warning(f"Gemini Vision API server error (503/High Demand), will fallback to Ollama: {e}")
            elif "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                logger.warning(f"Gemini Vision API rate limit/quota exceeded, will fallback to Ollama: {e}")
            elif "404" in error_msg or "not found" in error_msg.lower():
                logger.warning(f"Gemini Vision model not found, will fallback to Ollama: {e}")
            else:
                logger.warning(f"Gemini Vision API error, will fallback to Ollama: {e}")
            return ""

    def _analyze_image_ollama(self, image_content: bytes, prompt: str) -> str:
        """Analyze a single image with Ollama vision model - optimized for speed."""
        model = self._get_model()
        try:
            import base64
            # Resize image for faster processing (LLaVA native resolution from config)
            resized_content = self._resize_image_for_vision(image_content)
            b64_image = base64.b64encode(resized_content).decode("utf-8")

            # Optimized options for speed from config
            num_predict = getattr(self.vision_config, 'num_predict', 150)
            temperature = getattr(self.vision_config, 'temperature', 0.1)
            
            options = {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_k": 20,
                "top_p": 0.9,
            }

            response = self.client.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image]
                }],
                options=options
            )
            return str(response.get("message", {}).get("content", "")).strip()
        except Exception as e:
            logger.warning(f"Ollama vision analysis failed: {e}")
            return ""

    def _analyze_image(self, image_content: bytes, prompt: str) -> str:
        """Analyze a single image - tries Gemini Vision first (primary), then Ollama (fallback)."""
        # Try Gemini Vision first (primary)
        if self._gemini_available:
            logger.debug("Attempting Gemini Vision analysis...")
            result = self._analyze_image_gemini(image_content, prompt)
            if result:
                logger.debug("Gemini Vision analysis succeeded")
                return result
            logger.info("Gemini Vision failed/empty, falling back to Ollama...")
        else:
            logger.debug("Gemini Vision not available, using Ollama...")
        
        # Fallback to Ollama (secondary - local)
        return self._analyze_image_ollama(image_content, prompt)

    def _build_analysis_prompt(self) -> str:
        """Build the prompt for vision analysis."""
        return (
            "Analyze this FC Barcelona related image/video frame. Provide a structured response:\n\n"
            "1. DESCRIPTION: Brief description (1 sentence)\n"
            "2. PLAYERS: List Barcelona players recognized (Yamal, Pedri, Gavi, Lewandowski, Raphinha, "
            "Kounde, Araujo, Balde, Ter Stegen, Fermin, Olmo, Torres, Fati, Eric Garcia, Martinez, Christensen, "
            "Pena, Szczesny, Fort, Casado, Bernal, Torre, Cubarsi, etc.)\n"
            "3. STAFF: Any coaches/staff (Flick, Deco, Laporta)\n"
            "4. TEXT: Any visible text (scoreboards, graphics, shirts, banners)\n"
            "5. SCENE TYPE: One of: training, match, press_conference, celebration, graphic, meme, merch, unknown\n"
            "6. BARCA RELATED: Yes/No - Is this clearly FC Barcelona content?\n"
            "7. CONFIDENCE: 0.0-1.0 - Your confidence in this analysis\n"
            "8. KEY FACTS: Specific facts (scores, minutes, injury signs, jersey numbers)\n\n"
            "Format exactly as:\n"
            "DESCRIPTION: ...\n"
            "PLAYERS: ...\n"
            "STAFF: ...\n"
            "TEXT: ...\n"
            "SCENE TYPE: ...\n"
            "BARCA RELATED: ...\n"
            "CONFIDENCE: ...\n"
            "KEY FACTS: ..."
        )

    def _parse_vision_response(self, response: str) -> Dict[str, Any]:
        """Parse structured vision response."""
        result = {
            "description": "",
            "players": [],
            "staff": [],
            "text": "",
            "scene_type": "unknown",
            "is_barca": False,
            "confidence": 0.5,
            "key_facts": []
        }

        try:
            lines = response.strip().split("\n")
            current_field = None
            for line in lines:
                line = line.strip()
                if line.startswith("DESCRIPTION:"):
                    result["description"] = line[len("DESCRIPTION:"):].strip()
                elif line.startswith("PLAYERS:"):
                    players_text = line[len("PLAYERS:"):].strip()
                    if players_text and players_text.lower() != "none":
                        result["players"] = [p.strip() for p in players_text.split(",") if p.strip()]
                elif line.startswith("STAFF:"):
                    staff_text = line[len("STAFF:"):].strip()
                    if staff_text and staff_text.lower() != "none":
                        result["staff"] = [s.strip() for s in staff_text.split(",") if s.strip()]
                elif line.startswith("TEXT:"):
                    result["text"] = line[len("TEXT:"):].strip()
                elif line.startswith("SCENE TYPE:"):
                    result["scene_type"] = line[len("SCENE TYPE:"):].strip().lower()
                elif line.startswith("BARCA RELATED:"):
                    result["is_barca"] = "yes" in line.lower()
                elif line.startswith("CONFIDENCE:"):
                    conf_text = line[len("CONFIDENCE:"):].strip()
                    try:
                        # Extract first number
                        import re
                        nums = re.findall(r"0?\.\d+|1\.0|0|1", conf_text)
                        if nums:
                            result["confidence"] = max(0.0, min(1.0, float(nums[0])))
                    except:
                        pass
                elif line.startswith("KEY FACTS:"):
                    facts_text = line[len("KEY FACTS:"):].strip()
                    if facts_text and facts_text.lower() != "none":
                        result["key_facts"] = [f.strip() for f in facts_text.split(";") if f.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse vision response: {e}")

        return result

    def analyze_image_url(self, url: str) -> VisionResult:
        """Analyze an image from URL."""
        start_time = time.time()
        result = VisionResult(media_url=url, media_type="image")

        try:
            # Download image
            content, media_type, ext = self._download_media(url)
            if media_type != "image":
                result.error = f"Not an image: {media_type}"
                return result

            # Analyze
            prompt = self._build_analysis_prompt()
            response = self._analyze_image(content, prompt)
            parsed = self._parse_vision_response(response)

            result.description = parsed["description"]
            result.players_detected = parsed["players"]
            result.text_extracted = parsed["text"]
            result.scene_type = parsed["scene_type"]
            result.is_barca_related = parsed["is_barca"]
            result.confidence = parsed["confidence"]
            result.key_facts = parsed["key_facts"]

            # Calculate importance boost
            result.importance_boost = self._calculate_importance_boost(result)

        except Exception as e:
            result.error = str(e)
            logger.warning(f"Image analysis failed for {url}: {e}")

        result.analysis_time_ms = int((time.time() - start_time) * 1000)
        return result

    def analyze_video_url(self, url: str) -> VisionResult:
        """Analyze a video from URL by extracting and analyzing key frames."""
        start_time = time.time()
        result = VisionResult(media_url=url, media_type="video")

        try:
            # Download video
            content, media_type, ext = self._download_media(url)
            if media_type != "video":
                result.error = f"Not a video: {media_type}"
                return result

            # Extract frames
            frames = self._extract_frames_from_video(content)
            if not frames:
                result.error = "Could not extract frames from video"
                return result

            # Analyze each frame
            prompt = self._build_analysis_prompt()
            all_descriptions = []
            all_players = set()
            all_staff = set()
            all_text = []
            scene_types = []
            confidences = []
            barca_votes = 0
            all_facts = []

            for frame in frames:
                try:
                    frame_content = Path(frame.path).read_bytes()
                    response = self._analyze_image(frame_content, prompt)
                    parsed = self._parse_vision_response(response)

                    if parsed["description"]:
                        all_descriptions.append(f"[{frame.timestamp_sec:.0f}s] {parsed['description']}")
                    all_players.update(parsed["players"])
                    all_staff.update(parsed["staff"])
                    if parsed["text"]:
                        all_text.append(f"[{frame.timestamp_sec:.0f}s] {parsed['text']}")
                    scene_types.append(parsed["scene_type"])
                    confidences.append(parsed["confidence"])
                    if parsed["is_barca"]:
                        barca_votes += 1
                    all_facts.extend(parsed["key_facts"])

                except Exception as e:
                    logger.warning(f"Frame analysis failed: {e}")
                finally:
                    # Cleanup frame
                    try:
                        os.unlink(frame.path)
                    except OSError:
                        pass

            # Aggregate results
            result.description = " | ".join(all_descriptions)
            result.players_detected = list(all_players)
            result.text_extracted = " | ".join(all_text)
            # Most common scene type
            if scene_types:
                result.scene_type = max(set(scene_types), key=scene_types.count)
            # Average confidence
            result.confidence = sum(confidences) / len(confidences) if confidences else 0.5
            result.is_barca_related = barca_votes >= len(frames) / 2  # Majority vote
            result.key_facts = list(dict.fromkeys(all_facts))  # Deduplicate

            # Calculate importance boost
            result.importance_boost = self._calculate_importance_boost(result)

        except Exception as e:
            result.error = str(e)
            logger.warning(f"Video analysis failed for {url}: {e}")

        result.analysis_time_ms = int((time.time() - start_time) * 1000)
        return result

    def _calculate_importance_boost(self, result: VisionResult) -> float:
        """Calculate importance boost based on vision analysis."""
        if not self.vision_config.importance_boost_enabled:
            return 0.0

        boost = 0.0

        # Barça related content
        if result.is_barca_related:
            boost += 0.2

        # Players detected
        if result.players_detected:
            boost += min(len(result.players_detected) * 0.1, 0.3)

        # Staff detected
        if any(s in str(result.players_detected) for s in self.BARCA_STAFF):
            boost += 0.15

        # Scene type importance
        scene_boosts = {
            "match": 0.3,
            "press_conference": 0.25,
            "training": 0.15,
            "celebration": 0.1,
            "graphic": 0.05,
            "meme": -0.2,
            "merch": -0.15,
        }
        boost += scene_boosts.get(result.scene_type, 0)

        # Text extracted (scores, stats, etc.)
        if result.text_extracted:
            # Check for numbers, scores
            import re
            if re.search(r"\d+[-:]\d+", result.text_extracted):  # Score pattern
                boost += 0.15
            if re.search(r"\b\d{1,2}'\b", result.text_extracted):  # Minute marker
                boost += 0.1

        # Key facts
        if result.key_facts:
            boost += min(len(result.key_facts) * 0.05, 0.2)

        return max(-0.3, min(0.5, boost))  # Clamp between -0.3 and +0.5

    def analyze_media(self, media_urls: List[str], media_types: List[str]) -> List[VisionResult]:
        """Analyze multiple media items (images and/or videos)."""
        results = []
        for url, mtype in zip(media_urls, media_types):
            if mtype in ("photo", "image"):
                results.append(self.analyze_image_url(url))
            elif mtype in ("video", "animated_gif"):
                results.append(self.analyze_video_url(url))
            else:
                results.append(VisionResult(
                    media_url=url,
                    media_type=mtype,
                    error=f"Unsupported media type: {mtype}"
                ))
        return results

    def get_combined_analysis(self, media_urls: List[str], media_types: List[str]) -> Dict[str, Any]:
        """Get combined analysis for all media in a tweet."""
        if not media_urls:
            return {
                "has_media": False,
                "vision_results": [],
                "total_importance_boost": 0.0,
                "all_players": [],
                "all_text": "",
                "dominant_scene": "unknown",
                "is_barca_media": False
            }

        results = self.analyze_media(media_urls, media_types)

        all_players = []
        all_text = []
        all_facts = []
        scene_types = []
        barca_count = 0
        total_boost = 0.0

        for r in results:
            all_players.extend(r.players_detected)
            if r.text_extracted:
                all_text.append(r.text_extracted)
            all_facts.extend(r.key_facts)
            scene_types.append(r.scene_type)
            if r.is_barca_related:
                barca_count += 1
            total_boost += r.importance_boost

        # Deduplicate players
        all_players = list(dict.fromkeys(all_players))

        return {
            "has_media": True,
            "vision_results": results,
            "total_importance_boost": total_boost / len(results) if results else 0.0,
            "all_players": all_players,
            "all_text": " | ".join(all_text),
            "dominant_scene": max(set(scene_types), key=scene_types.count) if scene_types else "unknown",
            "is_barca_media": barca_count > len(results) / 2,
            "all_key_facts": list(dict.fromkeys(all_facts))
        }


def get_vision_analyzer(config=None) -> VisionAnalyzer:
    """Factory function to get vision analyzer instance."""
    return VisionAnalyzer(config)