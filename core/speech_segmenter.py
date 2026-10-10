"""16 kHz streaming Silero VAD. Each stream owns its recurrent model state."""
import io
import logging
import threading
from dataclasses import dataclass

import numpy as np

from core.asr_config import (
    SAMPLE_RATE, VAD_WINDOW, VAD_THRESHOLD, MIN_SPEECH_MS, END_SILENCE_MS,
    SPEECH_PAD_MS, MAX_SPEECH_SECONDS, FORCED_OVERLAP_MS, log_event,
)

_LOAD_LOCK = threading.Lock()
_MODEL_BYTES = None

class SileroProbability:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def __call__(self, window):
        global _MODEL_BYTES
        import torch
        with self.lock:
            if self.model is None:
                with _LOAD_LOCK:
                    if _MODEL_BYTES is None:
                        from silero_vad import load_silero_vad
                        prototype = load_silero_vad(onnx=False)
                        weights = io.BytesIO()
                        torch.jit.save(prototype, weights)
                        _MODEL_BYTES = weights.getvalue()
                    # Cache weights, never share the mutable recurrent state.
                    self.model = torch.jit.load(io.BytesIO(_MODEL_BYTES), map_location="cpu")
                    self.model.reset_states()
            with torch.inference_mode():
                return float(self.model(torch.from_numpy(window), SAMPLE_RATE).item())


@dataclass
class SpeechSegment:
    audio: np.ndarray
    start_sample: int
    end_sample: int
    reason: str


class SpeechSegmenter:
    def __init__(self, probability=None):
        self.probability = probability if probability is not None else SileroProbability()
        self._lock = threading.Lock()
        self._pending = np.empty(0, dtype=np.float32)
        self._buffer = np.empty(0, dtype=np.float32)
        self._base = 0
        self._position = 0
        self._start = None
        self._last_voice = None
        self._voice_samples = 0
        self._new_voice_samples = 0
        self._pad = SAMPLE_RATE * SPEECH_PAD_MS // 1000
        self._min = SAMPLE_RATE * MIN_SPEECH_MS // 1000
        self._silence = SAMPLE_RATE * END_SILENCE_MS // 1000
        self._max = int(SAMPLE_RATE * MAX_SPEECH_SECONDS)
        self._overlap = SAMPLE_RATE * FORCED_OVERLAP_MS // 1000

    @property
    def buffered_samples(self):
        return self._buffer.size + self._pending.size


    def _trim(self, start):
        self._buffer = self._buffer[max(0, start - self._base):].copy()
        self._base = start

    def _emit(self, end, reason):
        accepted = self._voice_samples >= self._min and self._new_voice_samples > 0 and end > self._start
        log_event("VAD][BUFFER", f"start={self._start / SAMPLE_RATE:.3f}s end={end / SAMPLE_RATE:.3f}s "
                  f"duration={(end - self._start) / SAMPLE_RATE:.3f}s "
                  f"speech={self._voice_samples / SAMPLE_RATE:.3f}s reason={reason} accepted={accepted}")
        if not accepted:
            return None
        return SpeechSegment(self._buffer[self._start - self._base:end - self._base].copy(),
                             self._start, end, reason)

    def _window(self, window, valid_samples):
        start = self._position
        self._buffer = np.concatenate((self._buffer, window[:valid_samples]))
        self._position += valid_samples
        probability = self.probability(window)
        voiced = probability >= VAD_THRESHOLD
        log_event("VAD", f"start={start / SAMPLE_RATE:.3f}s end={self._position / SAMPLE_RATE:.3f}s "
                  f"probability={probability:.4f} voiced={voiced}", level=logging.DEBUG)
        if voiced:
            if self._start is None:
                self._start = max(self._base, start - self._pad)
                log_event("VAD", f"speech_start={start / SAMPLE_RATE:.3f}s probability={probability:.4f}")
            self._last_voice = self._position
            self._voice_samples += valid_samples
            self._new_voice_samples += valid_samples
        if self._start is not None:
            if not voiced and self._position - self._last_voice >= self._silence:
                end = min(self._position, self._last_voice + self._pad)
                segment = self._emit(end, "silence")
                self._start = self._last_voice = None
                self._voice_samples = self._new_voice_samples = 0
                self._trim(max(end, self._position - self._pad))
                return segment
            if self._position - self._start >= self._max:
                # Exact sample boundary, retaining the rest of this VAD window.
                end = self._start + self._max
                segment = self._emit(end, "max_duration")
                self._start = end - self._overlap
                self._new_voice_samples = self._position - end if voiced else 0
                self._voice_samples = min(self._voice_samples, self._overlap) + self._new_voice_samples
                self._trim(self._start)
                return segment
        else:
            self._trim(max(self._base, self._position - self._pad))
        return None

    def feed(self, audio):
        with self._lock:
            self._pending = np.concatenate((self._pending, np.asarray(audio, dtype=np.float32).reshape(-1)))
            segments = []
            count = self._pending.size // VAD_WINDOW
            for index in range(count):
                segment = self._window(self._pending[index * VAD_WINDOW:(index + 1) * VAD_WINDOW], VAD_WINDOW)
                if segment is not None:
                    segments.append(segment)
            self._pending = self._pending[count * VAD_WINDOW:].copy()
            return segments

    def flush(self):
        with self._lock:
            segments = []
            if self._pending.size:
                window = np.zeros(VAD_WINDOW, dtype=np.float32)
                window[:self._pending.size] = self._pending
                segment = self._window(window, self._pending.size)
                if segment is not None:
                    segments.append(segment)
                self._pending = np.empty(0, dtype=np.float32)
            if self._start is not None:
                end = max(self._start, min(self._position, self._last_voice + self._pad))
                segment = self._emit(end, "flush")
                if segment is not None:
                    segments.append(segment)
            self._start = self._last_voice = None
            self._voice_samples = self._new_voice_samples = 0
            self._buffer = np.empty(0, dtype=np.float32)
            self._base = self._position
            return segments
