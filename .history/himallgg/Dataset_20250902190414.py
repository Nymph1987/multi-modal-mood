import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MMDataset(Dataset):
    def __init__(self, samples, speaker_to_idx=None, device="cpu",
                 missing_rate=0.0, missing_modalities=None, seed=42):
        """
        samples: dict of lists
        missing_rate: float, [0,1] 随机缺失概率
        missing_modalities: list, 指定缺失模态 (['text', 'audio', 'vision'])
        """
        self.samples = samples
        self.speaker_to_idx = speaker_to_idx if speaker_to_idx else {}
        self.device = device
        self.missing_rate = missing_rate
        self.missing_modalities = missing_modalities if missing_modalities else []
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples["raw_text"])

    def __getitem__(self, idx):
        """返回单个样本的 dict，并应用缺失策略"""
        sample = {k: v[idx] for k, v in self.samples.items()}

        # 固定缺失模态
        for m in self.missing_modalities:
            if m in sample:
                sample[m] = np.zeros_like(sample[m])

        # 按概率缺失
        if self.missing_rate > 0:
            for m in ["text", "audio", "vision"]:
                if self.rng.random() < self.missing_rate:
                    if m in sample:
                        sample[m] = np.zeros_like(sample[m])

        return sample

    def collate_fn(self, batch):
        """合并 batch"""
        batch_dict = {k: [d[k] for d in batch] for k in batch[0]}
        return self.padding(batch_dict)

    def padding(self, samples):
        texts = samples["text"]
        audios = samples["audio"]
        visuals = samples["vision"]
        labels = samples["classification_labels"]
        raw_texts = samples["raw_text"]

        batch_size = len(texts)
        text_len_tensor = torch.tensor([len(t) for t in texts]).long()
        mx = text_len_tensor.max().item()

        # 自动获取维度
        text_dim = texts[0][0].shape[-1]
        audio_dim = audios[0][0].shape[-1]
        visual_dim = visuals[0][0].shape[-1]

        # 初始化张量
        text_tensor = torch.zeros((batch_size, mx, text_dim))
        audio_tensor = torch.zeros((batch_size, mx, audio_dim))
        visual_tensor = torch.zeros((batch_size, mx, visual_dim))
        speaker_tensor = torch.zeros((batch_size, mx)).long()

        mask_text = torch.zeros((batch_size, mx))
        mask_audio = torch.zeros((batch_size, mx))
        mask_visual = torch.zeros((batch_size, mx))

        for i in range(batch_size):
            cur_len = len(texts[i])

            text_tensor[i, :cur_len, :] = torch.from_numpy(texts[i]).float()
            audio_tensor[i, :cur_len, :] = torch.from_numpy(audios[i]).float()
            visual_tensor[i, :cur_len, :] = torch.from_numpy(visuals[i]).float()

            mask_text[i, :cur_len] = 1
            mask_audio[i, :cur_len] = 1
            mask_visual[i, :cur_len] = 1

            if "speakers" in samples:
                speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx.get(s, 0) for s in samples["speakers"][i]]
                )

        label_tensor = torch.tensor([l for l in labels]).long()

        return {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "mask_text": mask_text,
            "mask_audio": mask_audio,
            "mask_visual": mask_visual,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "sentence": raw_texts,
        }
