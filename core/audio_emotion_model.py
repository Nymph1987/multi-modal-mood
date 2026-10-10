# -*- coding: utf-8 -*-
"""
功能：语音情感识别模块
算法：Wav2Vec 2.0 (基于 SpeechBrain 微调模型)
说明：从原始波形中提取深层声学特征并分类，模型不可用时不生成情绪结果
"""
import os
import sys
import time
import torch
import numpy as np
import librosa
import logging
from contextlib import contextmanager

# Suppress SpeechBrain and Transformers logs globally
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Add the IEMOCAP model directory to sys.path to import custom_interface
# Update: User renamed model folder to wav2vec2-base
IEMOCAP_PATH = "/mnt/F/emotional/himal-my-model/model/wav2vec2-base"
if os.path.exists(IEMOCAP_PATH):
    sys.path.append(IEMOCAP_PATH)

try:
    from custom_interface import CustomEncoderWav2vec2Classifier
    HAS_SB = True
except ImportError:
    HAS_SB = False

class AudioEmotionModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.classifier = None
        self.last_predict_error = None
        self.labels = ["neutral", "happy", "sad", "angry"]
        
        # IEMOCAP Labels Mapping
        # IEMOCAP typically: [neu, hap, ang, sad] or similar
        self.iemocap_mapping = {
            "neu": "neutral",
            "neutral": "neutral",
            "hap": "happy",
            "happy": "happy",
            "joy": "happy",
            "ang": "angry",
            "angry": "angry",
            "sad": "sad",
            "sadness": "sad"
        }

        # Force load SpeechBrain model - NO FALLBACK
        if not HAS_SB:
            raise ImportError("SpeechBrain library not found. Please install it with: pip install speechbrain")
            
        if not os.path.exists(IEMOCAP_PATH):
            raise FileNotFoundError(f"IEMOCAP model not found at {IEMOCAP_PATH}")

        print(f"[AudioModel] Loading local Wav2Vec2-IEMOCAP model from {IEMOCAP_PATH}...")
        try:
            # 仅使用本地目录，不修改影响其他模型下载的全局离线环境。

            from pathlib import Path
            from packaging.version import Version

            cache = Path(__file__).resolve().parents[1] / 'wav2vec2_checkpoints/models--facebook--wav2vec2-base/snapshots'
            configured_base = os.getenv("LOCAL_WAV2VEC2_DIR")
            if configured_base is not None:
                if not configured_base.strip():
                    raise ValueError("LOCAL_WAV2VEC2_DIR 为空，请指定完整的本地基础模型目录。")
                candidates = [Path(configured_base).expanduser()]
            else:
                candidates = [Path("/mnt/F/emotional/himal-my-model/model/facebook/wav2vec2-base")]
                candidates.extend(sorted(path for path in cache.glob('*') if path.is_dir()))

            safe_dirs = []
            bin_dirs = []
            incomplete = []
            for candidate in candidates:
                missing = [name for name in ('config.json', 'preprocessor_config.json')
                           if not (candidate / name).is_file()]
                has_safe = (candidate / 'model.safetensors').is_file()
                has_bin = (candidate / 'pytorch_model.bin').is_file()
                if missing or not (has_safe or has_bin):
                    if not (has_safe or has_bin):
                        missing.append('model.safetensors 或 pytorch_model.bin')
                    weight_note = '（存在 model.safetensors，但目录不完整）' if has_safe else ''
                    incomplete.append(f"{candidate}{weight_note} 缺少：{', '.join(missing)}")
                    continue
                if has_safe:
                    safe_dirs.append(candidate)
                elif has_bin:
                    bin_dirs.append(candidate)

            # 配置与权重必须位于同一目录，不拼接不同 revision，也不下载缺失文件。
            if safe_dirs:
                local_base = str(safe_dirs[0])
                weight_format = 'model.safetensors'
            elif bin_dirs and Version(torch.__version__.split('+')[0]) >= Version('2.6'):
                local_base = str(bin_dirs[0])
                weight_format = 'pytorch_model.bin'
            else:
                details = '; '.join(incomplete) or '未找到完整的 model.safetensors 目录'
                if bin_dirs:
                    raise RuntimeError(
                        f"当前 torch={torch.__version__}，加载 pytorch_model.bin 要求 torch>=2.6；"
                        f"没有完整的 safetensors 基础模型。{details}。缓存路径：{cache}。"
                        "请提供同一 revision 的完整本地模型，不能复制其他 revision 的配置。"
                    )
                prefix = 'LOCAL_WAV2VEC2_DIR 配置无效' if configured_base is not None else '本地 Wav2Vec2 基础模型不完整'
                raise FileNotFoundError(f"{prefix}：{details}。缓存路径：{cache}。不会回退下载。")
            print(f"[AudioModel] 本地基础模型目录：{local_base}；权重格式：{weight_format}")
            for filename in ('wav2vec2.ckpt', 'model.ckpt', 'label_encoder.txt'):
                if not os.path.isfile(os.path.join(IEMOCAP_PATH, filename)):
                    raise FileNotFoundError(f'Local IEMOCAP file unavailable: {filename}')
            
            # Some versions check for preprocessor_config.json, others config.json
            # We set the path to the directory containing config.json
            
            # 本地路径覆盖在解析超参数时生效，基础模型和预训练权重均不使用仓库名。
            hparams_file = "hyperparams.yaml"
            hp_path = os.path.join(IEMOCAP_PATH, hparams_file)
            
            if not os.path.isfile(hp_path):
                raise FileNotFoundError(f'Local hyperparameters unavailable: {hp_path}')
            # Add IEMOCAP path to sys.path again just in case
            if IEMOCAP_PATH not in sys.path:
                sys.path.append(IEMOCAP_PATH)
                
            # Import CustomEncoderWav2vec2Classifier dynamically if not already imported
            if 'custom_interface' not in sys.modules:
                from custom_interface import CustomEncoderWav2vec2Classifier
            else:
                CustomEncoderWav2vec2Classifier = sys.modules['custom_interface'].CustomEncoderWav2vec2Classifier

            # Suppress transformers logging during loading
            import transformers
            transformers.logging.set_verbosity_error()

            with suppress_stdout():
                self.classifier = CustomEncoderWav2vec2Classifier.from_hparams(
                    source=IEMOCAP_PATH,
                    hparams_file=hparams_file,
                    run_opts={"device": self.device},
                    overrides={"wav2vec2_hub": local_base, "pretrained_path": IEMOCAP_PATH},
                    savedir=IEMOCAP_PATH
                )
            print("[AudioModel] Local IEMOCAP model loaded successfully.")
        except Exception as e:
            print(f"[AudioModel] CRITICAL ERROR: Failed to load IEMOCAP model: {e}")
            import traceback
            traceback.print_exc()
            raise e # Raise error to let user know something is wrong instead of silent fallback

    def predict(self, audio_waveform, sample_rate=16000):
        """
        Input: audio_waveform (np.float32, 16kHz)
        Output: {emotion, confidence, probs}
        """
        start_ts = time.time()
        self.last_predict_error = None
        
        try:
            # Ensure audio is float32 and 1D
            y = audio_waveform.flatten().astype(np.float32)
            
            # --- Path A: Use Wav2Vec2-IEMOCAP model ---
            if self.classifier:
                # Convert to torch tensor
                waveform_tensor = torch.from_numpy(y).to(self.device)
                
                # SpeechBrain expects [batch, time]
                # Ensure input length is sufficient (at least ~0.5s) to avoid errors
                if waveform_tensor.shape[0] < 8000:
                     return None # Too short

                waveform_tensor = waveform_tensor.unsqueeze(0) 

                with torch.no_grad():
                    # classify_batch returns (out_prob, score, index, text_lab)
                    out_prob, score, index, text_lab = self.classifier.classify_batch(waveform_tensor)
                
                # 单条语音允许额外的单例维度，标签必须按编码器的真实类别顺序解码。
                posterior = out_prob.detach().float().cpu().numpy().reshape(-1)
                if posterior.size != out_prob.shape[-1]:
                    raise ValueError(f'Expected single posterior, got {tuple(out_prob.shape)}')
                if not np.all(np.isfinite(posterior)) or np.any(posterior < 0) or not np.isclose(posterior.sum(), 1.0, atol=1e-3):
                    raise ValueError('Invalid class posterior probabilities')
                class_indices = torch.arange(out_prob.shape[-1], device=out_prob.device)
                class_labels = [
                    str(label).lower()
                    for label in self.classifier.hparams.label_encoder.decode_torch(class_indices)
                ]

                if len(class_labels) != posterior.size:
                    raise ValueError('Label encoder and posterior size mismatch')
                probs = {
                    "neutral": 0.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "angry": 0.0
                }
                for label, prob in zip(class_labels, posterior):
                    mapped_label = self.iemocap_mapping.get(label, label)
                    if mapped_label not in probs:
                        raise ValueError(f'Unsupported IEMOCAP label: {label}')
                    probs[mapped_label] += float(prob)

                emotion = max(probs, key=probs.get)
                confidence = probs[emotion]

                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "probs": probs,
                    "latency": time.time() - start_ts,
                    "model": "wav2vec2-iemocap"
                }

            self.last_predict_error = 'SpeechBrain classifier unavailable; neural audio inference was not performed'
            return None

        except Exception as e:
            self.last_predict_error = str(e)
            print(f"[AudioModel] Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            return None
