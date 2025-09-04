import math
import random
import torch
import numpy as np


class Dataset:

    MODAL_MAP = {"t": "text", "a": "audio", "v": "visual"}

    def __init__(self, samples, batch_size, missing_rate=0.0, missing_modalities=None, seed=42):
        """
        samples: list of sample objects (包含 text/audio/visual/speaker/label/sentence)
        batch_size: 批大小
        missing_rate: float 或 dict, 单模态缺失率
                      float: 对所有指定模态统一缺失率
                      dict: {"text":0.1, "audio":0.3, "visual":0.5}
        missing_modalities: list, 指定缺失模态的缩写 ['t','a','v']
        seed: 随机种子
        """
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {"M": 0, "F": 1}

        if isinstance(missing_rate, float):
            # 统一缺失率
            self.missing_rate = {m: missing_rate for m in ["text","audio","visual"]}
        else:
            self.missing_rate = missing_rate

        if missing_modalities:
            self.missing_modalities = [self.MODAL_MAP[m] for m in missing_modalities]
        else:
            self.missing_modalities = ["text", "audio", "visual"]

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
            text_tensor[i, :cur_len, :] = torch.stack(tmp)

            # audio
            tmp = [torch.from_numpy(t).float() for t in s.audio]
            audio_tensor[i, :cur_len, :] = torch.stack(tmp)

            # visual
            tmp = [torch.from_numpy(t).float() for t in s.visual]
            visual_tensor[i, :cur_len, :] = torch.stack(tmp)

            # speaker
            speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])

            labels.extend(s.label)
            sentence.extend([s.sentence])

        label_tensor = torch.tensor(labels).long()

        # ====== 随机缺失逻辑 ======
        for mod_name, tensor in zip(["text","audio","visual"],
                                    [text_tensor, audio_tensor, visual_tensor]):
            if mod_name in self.missing_modalities and self.missing_rate.get(mod_name, 0) > 0:
                mask = torch.rand(batch_size) < self.missing_rate[mod_name]
                tensor[mask] = 0.0

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

    def collate_fn(self, batch):
        # batch 已经是 dict 形式
        return batch[0]

    def shuffle(self):
        random.shuffle(self.samples)
