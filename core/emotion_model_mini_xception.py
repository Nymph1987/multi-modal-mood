# -*- coding: utf-8 -*-
"""
core/emotion_model_mini_xception.py
功能：表情识别模块
算法：Mini-Xception (基于深度可分离卷积的轻量级 CNN)
说明：输入裁剪后的人脸图像，输出 7 种基础表情分类概率，专为实时端侧推理设计
"""
import os
import time
import threading

import cv2
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None


class MiniXceptionEmotionModel:
    def __init__(self, weights_path=None):
        if weights_path is None:
            # Updated to use centralized model directory
            weights_path = "/mnt/F/emotional/himal-my-model/model/face_classification/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

        self.weights_path = weights_path
        self.labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Mini-Xception model file not found: {self.weights_path}")

        self._inference_lock = threading.Lock()
        self.model = self._load_model(self.weights_path)
        input_shape = getattr(self.model, "input_shape", None)
        if input_shape is None or len(input_shape) < 3:
            self.input_size = (48, 48)
        else:
            self.input_size = (int(input_shape[1]), int(input_shape[2]))

    def _load_model(self, path):
        if tf is None:
            raise ImportError("Missing dependency: tensorflow. Mini-Xception requires TensorFlow CPU inference.")
        # 轻量视觉模型固定在CPU加载，不影响torch和Whisper使用GPU。
        with tf.device('/CPU:0'):
            return tf.keras.models.load_model(path, compile=False)

    def _preprocess(self, gray_face):
        x = gray_face.astype("float32")
        x = x / 255.0
        x = x - 0.5
        x = x * 2.0
        return x

    def predict(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.input_size)
        gray = self._preprocess(gray)
        gray = np.expand_dims(gray, axis=-1)
        x = np.expand_dims(gray, axis=0)

        # 共享模型串行推理，直接调用避免predict图缓存意外使用GPU/cuDNN。
        with self._inference_lock, tf.device('/CPU:0'):
            probs = self.model(x, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        emotion = self.labels[idx]
        confidence = float(probs[idx])
        probs_dict = {label: float(prob) for label, prob in zip(self.labels, probs)}
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probs": probs_dict,
            "timestamp": time.time()
        }
