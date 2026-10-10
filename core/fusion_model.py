# -*- coding: utf-8 -*-
"""
core/fusion_model.py
功能：多模态情感决策融合模块
算法：基于信息熵（Entropy）的自适应动态权重分配算法（Decision-level Fusion）
说明：实时评估各模态置信度，动态调整视觉、听觉、文本的决策权重，输出综合情感判定结果
"""
import numpy as np

class MultimodalFusion:
    def __init__(self):
        # Unified emotion labels (Ekman-like)
        self.emotions = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        
        # Default weights (used as base for dynamic adjustment)
        self.base_weights = {
            "vision": 0.4,
            "audio": 0.3,
            "text": 0.3
        }
        
        # Label mapping for different models to unified space
        self.label_map = {
            # Vision (Mini-Xception)
            "happiness": "happy",
            "joy": "happy",
            "sadness": "sad",
            "anger": "angry",
            # Audio (IEMOCAP labels: neu, hap, ang, sad, etc.)
            "neu": "neutral",
            "hap": "happy",
            "ang": "angry",
            "sad": "sad",
            "fru": "angry", # frustrated -> angry
            "exc": "happy", # excited -> happy
            "sur": "surprise",
            "fea": "fear",
            "dis": "disgust",
            # Text (CardiffNLP labels: anger, joy, optimism, sadness)
            "optimism": "happy",
        }

    def _to_vector(self, probs_dict):
        """Map probs_dict to unified emotion vector."""
        if not probs_dict:
            return None
            
        vec = np.zeros(len(self.emotions), dtype=np.float32)
        for label, probability in probs_dict.items():
            label = label.lower()
            target = self.label_map.get(label, label)
            if target not in self.emotions:
                continue
            if not np.isfinite(probability) or probability < 0:
                return None
            vec[self.emotions.index(target)] += probability

        # Normalize only distributions with recognized, non-zero probability mass.
        s = np.sum(vec)
        return vec / s if np.isfinite(s) and s > 0 else None

    def _calculate_dynamic_weights(self, vectors):
        """
        Calculate dynamic weights based on the confidence (entropy) of each modality.
        Lower entropy (higher confidence) -> Higher weight.
        """
        dynamic_weights = {}
        total_confidence = 0.0
        
        # 1. Calculate confidence score for each modality (1 - normalized entropy)
        # Or simpler: Max probability as a proxy for confidence
        scores = {}
        for mod, vec in vectors.items():
            # Using max probability as a simple confidence metric
            max_p = np.max(vec)
            # Entropy based: H = -sum(p * log(p))
            # Normalized Entropy: H_norm = H / log(N)
            # Confidence = 1 - H_norm
            
            # Handling edge case of p=0
            p_safe = np.clip(vec, 1e-7, 1.0)
            entropy = -np.sum(p_safe * np.log(p_safe))
            max_entropy = np.log(len(self.emotions))
            norm_entropy = entropy / max_entropy
            confidence = 1.0 - norm_entropy
            
            # Boost score with base weight preference
            score = confidence * (1.0 + self.base_weights.get(mod, 0.3))
            scores[mod] = max(0.01, score) # Ensure non-zero
            # 声学情绪可靠性较低，在归一化前将其权重分数降至原来的四分之一。
            if mod == "audio":
                scores[mod] *= 0.25
            total_confidence += scores[mod]
            
        # 2. Normalize weights
        if total_confidence > 0:
            for mod in vectors:
                dynamic_weights[mod] = scores[mod] / total_confidence
        else:
            # Fallback to equal weights if calculation fails
            for mod in vectors:
                dynamic_weights[mod] = 1.0 / len(vectors)
                
        return dynamic_weights

    def fuse(self, vision_probs=None, audio_probs=None, text_probs=None):
        """
        Late Fusion using dynamic weighted average based on modality confidence.
        """
        vectors = {}
        for modality, probs in (('vision', vision_probs), ('audio', audio_probs), ('text', text_probs)):
            vector = self._to_vector(probs)
            if vector is not None:
                vectors[modality] = vector
        
        if not vectors:
            return {"emotion": "neutral", "confidence": 0.0, "probs": {}}

        # Use dynamic weights instead of static default weights
        weights = self._calculate_dynamic_weights(vectors)
        
        final_vec = np.zeros(len(self.emotions), dtype=np.float32)
        for k, vec in vectors.items():
            final_vec += vec * weights[k]
            
        idx = np.argmax(final_vec)
        return {
            "emotion": self.emotions[idx],
            "confidence": float(final_vec[idx]),
            "fused_probs": {e: float(p) for e, p in zip(self.emotions, final_vec)},
            "active_modalities": list(vectors.keys()),
            "weights_used": {k: float(v) for k, v in weights.items()}
        }
