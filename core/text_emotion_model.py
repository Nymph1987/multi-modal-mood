# -*- coding: utf-8 -*-
"""
core/text_emotion_model.py
功能：文本情感分析与语音转写（ASR）模块
算法：Whisper (ASR) + TextBlob (语义极性分析) + 关键词匹配
说明：将语音转写为文本后，通过极性分析与自定义情感词典综合计算语义层面的情感倾向
"""
import time
import torch
import numpy as np
import os
import threading
import re
from opencc import OpenCC

_SIMPLIFIER = OpenCC('t2s')


def normalize_text(text):
    return _SIMPLIFIER.convert(text.strip())

from core.asr_config import (
    SAMPLE_RATE, ASR_MODEL_SIZE, ASR_LOCAL_FILES_ONLY, ASR_LANGUAGE, ASR_BEAM_SIZE,
    ASR_NO_SPEECH_THRESHOLD, ASR_LOG_PROB_THRESHOLD, ASR_TEMPERATURE,
    is_valid_transcript, log_event,
)
from core.speech_segmenter import SpeechSegmenter
from textblob import TextBlob
from transformers import pipeline

class TextEmotionModel:
    _asr_lock = threading.RLock()
    _asr_models = {}

    def __init__(self, device='cpu'):
        self.device = device
        self.classifier = "textblob"
        self.emotion_pipeline = None
        self.asr_pipeline = None
        self._loading_asr = False
        self.asr_load_error = None
        self.asr_infer_error = None
        self.emotion_load_error = None
        self.last_predict_error = None
        print("[TextModel] Initializing text emotion pipelines...")

        self.load_local_emotion_classifier()
        
        # ASR weights are loaded only on the first speech transcription.

    def load_local_emotion_classifier(self):
        model_path = os.environ.get('TEXT_EMOTION_MODEL_DIR') or "/mnt/F/emotional/himal-my-model/model/twitter-roberta-base-emotion"
        self.model_language = os.environ.get('TEXT_EMOTION_MODEL_LANGUAGE', 'en').lower()
        self.emotion_load_error = None
        if not os.path.exists(model_path):
            self.emotion_load_error = f'Local emotion model not found: {model_path}'
            print(f"[TextModel] Local emotion model not found at {model_path}. Falling back to TextBlob.")
            return

        try:
            from transformers import AutoModelForSequenceClassification
            if not os.path.isfile(os.path.join(model_path, 'model.safetensors')) and not os.path.isfile(os.path.join(model_path, 'model.safetensors.index.json')):
                raise RuntimeError('RoBERTa 缺少 safetensors 安全权重；当前 torch 版本不能安全加载旧 bin。请部署可信安全权重，暂用 TextBlob，语音转写不受此错误影响。')
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, use_safetensors=True, local_files_only=True,
                dtype=torch.float16 if torch.cuda.is_available() and self.device != 'cpu' else torch.float32,
            )
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=model_path,
                framework="pt",
                top_k=None,
                device=0 if torch.cuda.is_available() and self.device != 'cpu' else -1,
                dtype=torch.float16 if torch.cuda.is_available() and self.device != 'cpu' else torch.float32
            )
            labels = {str(label).lower() for label in model.config.id2label.values()}
            allowed = {'anger', 'joy', 'sadness', 'neutral', 'optimism', 'fear', 'surprise', 'disgust', 'positive', 'negative'}
            if not labels or not labels <= allowed:
                raise RuntimeError('config.id2label 必须提供真实语义标签，不能将 LABEL_0 等猜测为情绪。')
            self.classifier = os.path.basename(os.path.normpath(model_path))
            print(f"[TextModel] Local emotion classifier loaded from {model_path}.")
        except Exception as e:
            self.emotion_load_error = str(e)
            print(f"[TextModel] Emotion classifier load failed, falling back to TextBlob: {e}")
            self.emotion_pipeline = None
            self.classifier = "textblob"

    def load_local_asr(self):
        with self._asr_lock:
            if self.asr_pipeline is not None:
                return self.asr_pipeline
            self._loading_asr = True
            self.asr_load_error = None
            try:
                from faster_whisper import WhisperModel
                backend = 'cuda' if self.device != 'cpu' and torch.cuda.is_available() else 'cpu'
                device_index = int(self.device.split(':')[1]) if ':' in self.device and backend == 'cuda' else 0
                # CPU does not support fp16; CTranslate2 uses int8 on CPU.
                compute_type = 'float16' if backend == 'cuda' else 'int8'
                key = (ASR_MODEL_SIZE, backend, device_index, compute_type)
                if key not in self._asr_models:
                    log_event('ASR', f'loading model={ASR_MODEL_SIZE} device={backend}:{device_index} compute_type={compute_type}; CPU不支持fp16')
                    self._asr_models[key] = WhisperModel(
                        ASR_MODEL_SIZE, device=backend, device_index=device_index, compute_type=compute_type,
                        local_files_only=ASR_LOCAL_FILES_ONLY,
                    )
                self.asr_pipeline = self._asr_models[key]
                return self.asr_pipeline
            except Exception as e:
                error = str(e)
                if ASR_LOCAL_FILES_ONLY and 'Cannot find an appropriate cached snapshot' in error:
                    error = (
                        f'离线 Whisper 模型 {ASR_MODEL_SIZE} 未找到。'
                        '请部署完整 faster-whisper 模型（含 model.bin、config.json、tokenizer.json），'
                        '并设置 LOCAL_WHISPER_DIR 为模型目录后重启服务；Wav2Vec2 权重不能用于 Whisper 转写。'
                    )
                self.asr_load_error = error
                log_event('ASR', f'load_failed={error}')
                return None
            finally:
                self._loading_asr = False

    def transcribe_segment(self, audio_data, sample_rate=16000):
        """Decode a segment already admitted by the connection's Silero VAD."""
        self.asr_infer_error = None
        try:
            audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            if not audio.size:
                return None
            if sample_rate != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            with self._asr_lock:
                model = self.load_local_asr()
                if model is None:
                    return None
                start = time.monotonic()
                segments, _ = model.transcribe(
                    audio, language=ASR_LANGUAGE, beam_size=ASR_BEAM_SIZE,
                    temperature=ASR_TEMPERATURE, vad_filter=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
                    log_prob_threshold=ASR_LOG_PROB_THRESHOLD,
                )
                texts = []
                # faster-whisper decodes lazily: iteration must remain under the lock.
                for segment in segments:
                    valid = is_valid_transcript(segment.text)
                    log_event('ASR', f'start={segment.start:.3f}s end={segment.end:.3f}s '
                              f'duration={segment.end - segment.start:.3f}s avg_logprob={segment.avg_logprob:.4f} '
                              f'no_speech_prob={segment.no_speech_prob:.4f} accepted={valid} text={segment.text.strip()!r}')
                    if valid:
                        texts.append(segment.text.strip())
                text = normalize_text(' '.join(' '.join(texts).split()))
                log_event('ASR', f'audio_duration={audio.size / SAMPLE_RATE:.3f}s latency={time.monotonic() - start:.3f}s accepted={is_valid_transcript(text)} text={text!r}')
                return text if is_valid_transcript(text) else None
        except Exception as e:
            self.asr_infer_error = str(e)
            log_event('ASR', f'transcribe_failed={e}')
            return None

    def transcribe(self, audio_data, sample_rate=16000):
        """Independent audio uses exactly the streaming Silero segmentation config."""
        try:
            audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            if sample_rate != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            segmenter = SpeechSegmenter()
            segments = segmenter.feed(audio) + segmenter.flush()
            texts = [self.transcribe_segment(segment.audio) for segment in segments]
            text = ' '.join(value for value in texts if value)
            return text if is_valid_transcript(text) else None
        except Exception as e:
            log_event('ASR', f'vad_failed={e}')
            return None

    def predict_from_text(self, input_text):
        """
        Pure Sentiment: Text -> Emotion
        Input: str
        Output: {text, emotion, confidence, probs}
        """
        if not input_text or len(input_text.strip()) < 1:
            return None
        return self._predict_text_emotion(input_text.strip())

    def _predict_with_textblob(self, input_text):
        blob = TextBlob(input_text)
        polarity = blob.sentiment.polarity

        emotion = "neutral"
        probs = {}

        lower_text = input_text.lower()
        anger_kw = ["hate", "bad", "terrible", "worst", "angry", "kill", "die", "stupid", "讨厌", "坏", "糟糕", "愤怒", "生气", "死", "笨", "滚"]
        sad_kw = ["sad", "cry", "lonely", "depressed", "sorry", "miss", "hurt", "伤心", "难过", "哭", "孤独", "抑郁", "抱歉", "对不起", "想念", "疼"]
        joy_kw = ["love", "good", "great", "best", "happy", "amazing", "wonderful", "like", "爱", "好", "棒", "开心", "快乐", "喜欢", "赞", "高兴"]

        if any(x in lower_text for x in anger_kw):
            emotion = "anger"
            probs = {"anger": 0.8, "sadness": 0.1, "neutral": 0.1, "joy": 0.0, "optimism": 0.0}
        elif any(x in lower_text for x in sad_kw):
            emotion = "sadness"
            probs = {"sadness": 0.8, "neutral": 0.1, "anger": 0.1, "joy": 0.0, "optimism": 0.0}
        elif any(x in lower_text for x in joy_kw):
            emotion = "joy"
            probs = {"joy": 0.7, "optimism": 0.1, "neutral": 0.2, "sadness": 0.0, "anger": 0.0}
        elif polarity > 0.1:
            joy_score = min(0.95, 0.55 + (polarity * 0.35))
            optimism_score = min(0.25, max(0.05, polarity * 0.2))
            neutral_score = max(0.0, 1.0 - joy_score - optimism_score)
            emotion = "joy"
            probs = {"joy": joy_score, "optimism": optimism_score, "neutral": neutral_score, "sadness": 0.0, "anger": 0.0}
        elif polarity < -0.1:
            sadness_score = min(0.95, 0.55 + (abs(polarity) * 0.35))
            neutral_score = max(0.0, 1.0 - sadness_score)
            emotion = "sadness"
            probs = {"sadness": sadness_score, "neutral": neutral_score, "joy": 0.0, "anger": 0.0, "optimism": 0.0}
        else:
            emotion = "neutral"
            probs = {"neutral": 0.8, "joy": 0.1, "sadness": 0.1, "anger": 0.0, "optimism": 0.0}

        return emotion, probs

    def _predict_chinese_rules(self, text):
        words = {
            'joy': ('开心', '快乐', '高兴', '幸福', '兴奋', '喜欢', '满意', '喜悦'),
            'sadness': ('难过', '伤心', '悲伤', '失落', '失望', '痛苦', '孤独', '委屈'),
            'anger': ('生气', '愤怒', '恼火', '气愤', '愤慨', '讨厌', '烦躁'),
            'fear': ('害怕', '恐惧', '担心', '焦虑', '紧张'),
            'optimism': ('期待', '有信心', '充满希望', '乐观'),
        }
        scores = dict.fromkeys((*words, 'neutral'), 0.0)
        for clause in re.split(r'[，。！？；,!?;]|但是|不过|但', text):
            for label, terms in words.items():
                for match in re.finditer('|'.join(map(re.escape, terms)), clause):
                    prefix = clause[max(0, match.start() - 8):match.start()]
                    negations = re.findall(r'不是|没有|并非|不|没|无', prefix)
                    negated = len(negations) % 2 == 1 and not prefix.endswith(('不得不', '忍不住'))
                    target = 'sadness' if negated and label == 'joy' else ('neutral' if negated else label)
                    weight = 1.0
                    if re.search(r'很|非常|特别|太|真的', prefix):
                        weight = 1.4
                    scores[target] += weight
        signal = sum(scores.values())
        if not signal:
            probs = {label: (0.55 if label == 'neutral' else 0.09) for label in scores}
        else:
            # Heuristic support scores, not calibrated model probabilities.
            probs = {label: value + (0.8 if label == 'neutral' else 0.08) for label, value in scores.items()}
            total = sum(probs.values())
            probs = {label: value / total for label, value in probs.items()}
        return max(probs, key=probs.get), probs

    def _predict_text_emotion(self, input_text):
        start_ts = time.time()
        if not input_text or len(input_text.strip()) < 1:
            return None

        try:
            self.last_predict_error = None
            input_text = normalize_text(input_text)
            contains_cjk = bool(re.search(r'[\u3400-\u9fff]', input_text))
            fallback_emotion, fallback_probs = (self._predict_chinese_rules(input_text) if contains_cjk
                                              else self._predict_with_textblob(input_text))
            final_probs = dict(fallback_probs)
            source = 'chinese-rules' if contains_cjk else 'textblob'
            model_name = 'zh-rules-baseline-v1' if contains_cjk else 'textblob'

            if self.emotion_pipeline is not None and contains_cjk == (self.model_language == 'zh'):
                raw_outputs = self.emotion_pipeline(input_text, truncation=True)
                if raw_outputs and isinstance(raw_outputs, list) and len(raw_outputs) == 1 and isinstance(raw_outputs[0], list):
                    raw_outputs = raw_outputs[0]

                model_probs = {}
                for item in raw_outputs or []:
                    label = str(item.get("label", "")).lower()
                    score = float(item.get("score", 0.0))
                    if label:
                        model_probs[label] = score

                if model_probs:
                    total = sum(model_probs.values())
                    if total > 0:
                        model_probs = {k: float(v / total) for k, v in model_probs.items()}

                    final_probs = model_probs
                    source = 'local-transformers'
                    model_name = self.classifier

            emotion = max(final_probs, key=final_probs.get)
            return {
                "text": input_text,
                "emotion": emotion,
                "confidence": float(final_probs.get(emotion, 0.0)),
                "probs": final_probs,
                "latency": time.time() - start_ts,
                "is_local": True,
                "model": model_name,
                "source": source,
                "labels": list(final_probs),
                "confidence_kind": 'model-score' if source == 'local-transformers' else 'heuristic-uncalibrated'
            }
        except Exception as e:
            self.last_predict_error = str(e)
            print(f"[TextModel] Emotion Error: {e}")
            return None

    def predict(self, input_data, sample_rate=16000):
        """
        Input: input_data (str or numpy array)
        Output: {text, emotion, confidence, probs}
        """
        input_text = ""
        
        # Case 1: Raw text passed from frontend
        if isinstance(input_data, str):
            input_text = input_data.strip()
        
        # Case 2: Audio waveform passed (Local ASR)
        elif hasattr(input_data, 'shape') or isinstance(input_data, (list, tuple)):
            input_text = self.transcribe(input_data, sample_rate)
            if input_text:
                print(f"[TextModel] Local ASR recognized: {input_text}")
            
        if not input_text or len(input_text.strip()) < 1:
            return None
        return self._predict_text_emotion(input_text)
