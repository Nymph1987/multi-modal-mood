import math
import random
import torch
import numpy as np


class Dataset:

    def __init__(self, samples, batch_size, missing_rate=None, missing_modalities=None, seed=42):
        """
        samples: list of sample objects (包含 text/audio/visual/speaker/label/sentence)
        batch_size: 批大小
        missing_rate: dict, 比如 {"text":0.0,"audio":0.3,"visual":0.5}
        missing_modalities: list, 指定哪些模态可能缺失，比如 ["audio","visual"]
        seed: 随机种子，保证可复现
        """
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {"M": 0, "F": 1}

        self.missing_rate = missing_rate if missing_rate else {}
        self.missing_modalities = missing_modalities if missing_modalities else ["text", "audio", "visual"]
        random.seed(seed)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()

        text_tensor = torch.zeros((batch_size, mx, 1024))
        audio_tensor = torch.zeros((batch_size, mx, 1582))
        visual_tensor = torch.zeros((batch_size, mx, 342))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        sentence = []

        for i, s in enumerate(samples):
            cur_len = len(s.text)

            # text
            tmp = [torch.from_numpy(t).float() for t in s.text]
            tmp = torch.stack(tmp)
            text_tensor[i, :cur_len, :] = tmp

            # audio
            tmp = [torch.from_numpy(t).float() for t in s.audio]
            tmp = torch.stack(tmp)
            audio_tensor[i, :cur_len, :] = tmp

            # visual
            tmp = [torch.from_numpy(t).float() for t in s.visual]
            tmp = torch.stack(tmp)
            visual_tensor[i, :cur_len, :] = tmp

            # speaker
            speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])

            labels.extend(s.label)
            sentence.extend([s.sentence])

        # label
        label_tensor = torch.tensor(labels).long()

        # ====== 随机缺失逻辑（仅对指定模态） ======
        if "text" in self.missing_modalities and self.missing_rate.get("text", 0) > 0:
            mask = torch.rand(batch_size) < self.missing_rate["text"]
            text_tensor[mask] = 0.0

        if "audio" in self.missing_modalities and self.missing_rate.get("audio", 0) > 0:
            mask = torch.rand(batch_size) < self.missing_rate["audio"]
            audio_tensor[mask] = 0.0

        if "visual" in self.missing_modalities and self.missing_rate.get("visual", 0) > 0:
            mask = torch.rand(batch_size) < self.missing_rate["visual"]
            visual_tensor[mask] = 0.0

        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "sentence": sentence,
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)
