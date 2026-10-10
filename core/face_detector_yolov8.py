# -*- coding: utf-8 -*-
"""
core/face_detector_yolov8.py
功能：人脸检测模块
算法：YOLOv8-Face (基于 ONNX 格式的深度卷积神经网络)
说明：用于从原始图像中定位人脸坐标，支持 GPU (CUDA) 与 CPU 推理
"""
import os
import urllib.request

import cv2
import numpy as np


class YoloV8FaceDetector:
    def __init__(
        self,
        weights_path=None,
        imgsz=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='cpu'
    ):
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device

        if weights_path is None:
            # Updated to use centralized model directory
            weights_path = "/mnt/F/emotional/himal-my-model/model/yolov8-face/weights/yolov8n-face-lindevs.onnx"

        self.weights_path = weights_path
        self._ensure_weights()
        self.net = cv2.dnn.readNetFromONNX(self.weights_path)
        self._using_cuda = False
        
        if 'cuda' in self.device:
            print(f"[YoloV8] Attempting to use CUDA (device={self.device})")
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self._using_cuda = True
            except Exception as e:
                print(f"[YoloV8] Warning: Failed to set CUDA backend: {e}. Falling back to CPU.")
                self._set_cpu_backend()
        else:
            self._set_cpu_backend()

    def _set_cpu_backend(self):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._using_cuda = False

    def _ensure_weights(self):
        if os.path.exists(self.weights_path):
            return

        if os.getenv("MODEL_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
            raise FileNotFoundError(f"YOLOv8-Face ONNX weights not found: {self.weights_path}")

        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        url = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx"
        print(f"[System] Downloading YOLOv8-Face ONNX weights: {url}")
        urllib.request.urlretrieve(url, self.weights_path)
        print(f"[System] Saved weights to: {self.weights_path}")

    def detect_faces(self, image_bgr):
        height, width = image_bgr.shape[:2]
        length = max((height, width))
        scale = length / float(self.imgsz)

        padded = np.zeros((length, length, 3), dtype=np.uint8)
        padded[0:height, 0:width] = image_bgr

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(self.imgsz, self.imgsz),
            swapRB=True,
        )

        self.net.setInput(blob)
        try:
            outputs = self.net.forward()
        except cv2.error as e:
            if self._using_cuda:
                print(f"[YoloV8] CUDA inference failed, retrying on CPU: {e}")
                self._set_cpu_backend()
                self.net.setInput(blob)
                outputs = self.net.forward()
            else:
                raise
        outputs = cv2.transpose(outputs[0])

        boxes = []
        scores = []
        for i in range(outputs.shape[0]):
            max_score = float(np.amax(outputs[i][4:]))
            if max_score < self.conf_threshold:
                continue

            x = (outputs[i][0] - (0.5 * outputs[i][2])) * scale
            y = (outputs[i][1] - (0.5 * outputs[i][3])) * scale
            w = outputs[i][2] * scale
            h = outputs[i][3] * scale

            boxes.append([x, y, w, h])
            scores.append(max_score)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        faces = []
        if indices is None or len(indices) == 0:
            return faces

        for idx in indices:
            i = int(idx) if not isinstance(idx, (list, tuple, np.ndarray)) else int(idx[0])
            x, y, w, h = boxes[i]
            x = int(round(max(0, x)))
            y = int(round(max(0, y)))
            w = int(round(max(0, w)))
            h = int(round(max(0, h)))
            if w > 0 and h > 0:
                faces.append((x, y, w, h))

        return faces
