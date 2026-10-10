# -*- coding: utf-8 -*-
"""
core/vision_pipeline.py
功能：视觉处理管线模块
实现：集成人脸检测与表情识别算法，对连续视频帧进行时域平滑（EMA）处理，输出稳定的人脸位置与表情概率
"""
import cv2
import time
import numpy as np
from core.face_detector_yolov8 import YoloV8FaceDetector
from core.emotion_model_mini_xception import MiniXceptionEmotionModel

class VisionPipeline:
    def __init__(self, device='cpu'):
        print(f"[VisionPipeline] Initializing YOLOv8 Face Detector on {device}...")
        self.detector = YoloV8FaceDetector(device=device)
        print("[VisionPipeline] Initializing Mini-Xception Emotion Model...")
        self.classifier = MiniXceptionEmotionModel()
        
        # Temporal Smoothing: Keep track of previous probabilities
        self.ema_alpha = 0.3 # Smoothing factor (0.0 to 1.0)
        self.prev_probs = None 
        
    def predict(self, frame_bgr):
        """
        Input: BGR Image
        Output: List of {bbox, emotion, confidence, probs}
        """
        start_ts = time.time()
        faces = self.detector.detect_faces(frame_bgr)
        use_temporal_smoothing = len(faces) == 1
        
        results = []
        for (x, y, w, h) in faces:
            # Crop and classify
            face_img = frame_bgr[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
                
            res = self.classifier.predict(face_img)
            probs = res.get("probs", {res["emotion"]: res["confidence"]})
            
            # --- Temporal Smoothing (EMA) ---
            # Only smooth when a single face is present; otherwise a shared cache
            # can leak one person's emotion into another person's result.
            if use_temporal_smoothing and self.prev_probs is None:
                self.prev_probs = probs
            elif use_temporal_smoothing:
                new_probs = {}
                for label in probs:
                    old_val = self.prev_probs.get(label, 0.0)
                    new_val = probs[label]
                    # EMA Formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
                    new_probs[label] = self.ema_alpha * new_val + (1 - self.ema_alpha) * old_val
                
                self.prev_probs = new_probs
                
                # Re-determine dominant emotion based on smoothed probabilities
                top_label = max(new_probs, key=new_probs.get)
                res["emotion"] = top_label
                res["confidence"] = new_probs[top_label]
                probs = new_probs

            results.append({
                "bbox": [x, y, w, h],
                "emotion": res["emotion"],
                "confidence": res["confidence"],
                "probs": probs
            })
            
        # Reset smoothing cache whenever tracking continuity is unreliable.
        if not use_temporal_smoothing:
            self.prev_probs = None
        elif not faces:
            self.prev_probs = None

        return results
