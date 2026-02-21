"""
Voice pipeline: audio recording + cactus_transcribe (local Whisper).

Falls back gracefully if Cactus or sox is not available.
"""

import os
import sys
import json
import tempfile
import subprocess
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

WHISPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "cactus", "weights", "whisper-small")
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Try importing Cactus
try:
    from cactus import cactus_init, cactus_transcribe, cactus_destroy
    CACTUS_AVAILABLE = True
except ImportError:
    CACTUS_AVAILABLE = False

# Check for sox (recording)
def _sox_available() -> bool:
    try:
        subprocess.run(["sox", "--version"], capture_output=True, timeout=3)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

SOX_AVAILABLE = _sox_available()

# Whisper model handle (lazy init, reused across calls)
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None and CACTUS_AVAILABLE:
        if os.path.exists(WHISPER_MODEL_PATH):
            try:
                _whisper_model = cactus_init(WHISPER_MODEL_PATH)
            except Exception:
                pass
    return _whisper_model


def record_audio(duration: float = 5.0, output_path: str = None) -> str:
    """
    Record audio via sox. Returns path to WAV file.
    Raises RuntimeError if sox is unavailable.
    """
    if not SOX_AVAILABLE:
        raise RuntimeError(
            "sox is not installed. Install with: brew install sox\n"
            "Voice input unavailable — use keyboard input instead."
        )
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # sox: record mono 16kHz WAV (Whisper's expected format)
    cmd = [
        "sox", "-d",               # default input device (microphone)
        "-r", "16000",             # 16 kHz sample rate
        "-c", "1",                 # mono
        "-b", "16",                # 16-bit
        output_path,
        "trim", "0", str(duration),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)
    if result.returncode != 0:
        raise RuntimeError(f"sox recording failed: {result.stderr.decode()}")
    return output_path


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a WAV file using cactus_transcribe (local Whisper).
    Falls back to mock text if Cactus is unavailable.
    """
    model = _get_whisper_model()
    if model is None:
        raise RuntimeError(
            "Whisper model not available. "
            "Run: cactus download whisper-small\n"
            "Or ensure Cactus is installed and the model is downloaded."
        )
    raw = cactus_transcribe(model, audio_path, prompt=WHISPER_PROMPT)
    data = json.loads(raw)
    return data.get("response", "").strip()


def record_and_transcribe(duration: float = 5.0) -> str:
    """
    Full pipeline: record → transcribe → return text.
    Returns transcribed string or raises on error.
    """
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        record_audio(duration=duration, output_path=tmp_path)
        return transcribe_audio(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def is_voice_available() -> bool:
    """Return True if both sox (recording) and Whisper (transcription) are ready."""
    return SOX_AVAILABLE and CACTUS_AVAILABLE and os.path.exists(WHISPER_MODEL_PATH)


def voice_status() -> dict:
    return {
        "sox_available": SOX_AVAILABLE,
        "cactus_available": CACTUS_AVAILABLE,
        "whisper_model_exists": os.path.exists(WHISPER_MODEL_PATH),
        "voice_ready": is_voice_available(),
    }
