"""Central ASR / VAD configuration; Silero weights are bundled by silero-vad."""
import logging
import os
import re
from datetime import datetime

# Silero's 16 kHz recurrent model requires exactly 512 samples per call.
SAMPLE_RATE = 16000
VAD_WINDOW = 512
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "250"))
END_SILENCE_MS = int(os.environ.get("VAD_END_SILENCE_MS", "500"))
SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", "100"))
MAX_SPEECH_SECONDS = float(os.environ.get("VAD_MAX_SPEECH_SECONDS", "15"))
FORCED_OVERLAP_MS = int(os.environ.get("VAD_FORCED_OVERLAP_MS", "250"))
EMOTION_RING_SECONDS = float(os.environ.get("EMOTION_RING_SECONDS", "1"))
EMOTION_INTERVAL_SECONDS = float(os.environ.get("EMOTION_INTERVAL_SECONDS", "0.35"))
# 离线服务器需指定完整的 CTranslate2 模型目录，不能使用 Transformers 权重目录。
ASR_MODEL_SIZE = os.environ.get("LOCAL_WHISPER_DIR") or os.environ.get("ASR_MODEL_SIZE", "large-v3")
ASR_LOCAL_FILES_ONLY = os.environ.get("ASR_LOCAL_FILES_ONLY", "1").lower() not in ("0", "false", "no")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "zh") or None
ASR_BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "5"))
ASR_NO_SPEECH_THRESHOLD = float(os.environ.get("ASR_NO_SPEECH_THRESHOLD", "0.6"))
ASR_LOG_PROB_THRESHOLD = float(os.environ.get("ASR_LOG_PROB_THRESHOLD", "-1.0"))
ASR_TEMPERATURE = float(os.environ.get("ASR_TEMPERATURE", "0"))
LOG_LEVEL = os.environ.get("ASR_LOG_LEVEL", "INFO").upper()

if not (0 < VAD_THRESHOLD < 1 and MIN_SPEECH_MS > 0 and END_SILENCE_MS > 0
        and SPEECH_PAD_MS >= 0 and MAX_SPEECH_SECONDS > 0
        and 0 <= FORCED_OVERLAP_MS < MAX_SPEECH_SECONDS * 1000
        and EMOTION_RING_SECONDS > 0 and EMOTION_INTERVAL_SECONDS > 0):
    raise ValueError("Invalid ASR/VAD timing configuration")

_LOGGER = logging.getLogger("himal.asr")
_LOGGER.setLevel(LOG_LEVEL)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
_LOGGER.propagate = False

_NOISE_TAG = re.compile(
    r"\[\s*(?:blank[ _]audio|silence|music|noise|applause|inaudible|无声|静音|音乐|噪音|掌声)\s*\]"
    r"|\(\s*(?:blank[ _]audio|silence|music|noise|applause|inaudible|无声|静音|音乐|噪音|掌声)\s*\)"
    r"|<\|[^|]*\|>",
    re.IGNORECASE,
)


def is_valid_transcript(text):
    if not isinstance(text, str):
        return False
    meaningful = _NOISE_TAG.sub("", text)
    return bool(re.search(r"[\u3400-\u9fff]", meaningful)) or len(re.findall(r"[a-zA-Z]", meaningful)) >= 2


def log_event(kind, message, level=logging.INFO):
    _LOGGER.log(level, f"[{kind}] {datetime.now().astimezone().isoformat(timespec='milliseconds')} {message}")
