import math
import random
import torch


class Dataset:

    def __init__(self, samples, batch_size, missing_modalities=None):
        """
        samples: 数据样本列表
        batch_size: 批大小
        missing_modalities: dict, 指定缺失模态及缺失率，例如：
                            {"audio": 0.3, "visual": 0.5}
        """
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'M': 0, 'F': 1}

        # 缺失模态配置
        self.missing_modalities = missing_modalities if missing_modalities else {}

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
    
        label_tensor = torch.tensor(labels).long()

        # === 缺失模态处理 ===
        if "audio" in self.missing_modalities:
            mask = (torch.rand(audio_tensor.shape[0]) < self.missing_modalities["audio"])
            for i in range(batch_size):
                if mask[i]:
                    audio_tensor[i] = 0  # 整个样本的 audio 缺失

        if "visual" in self.missing_modalities:
            mask = (torch.rand(visual_tensor.shape[0]) < self.missing_modalities["visual"])
            for i in range(batch_size):
                if mask[i]:
                    visual_tensor[i] = 0  # 整个样本的 visual 缺失

        if "text" in self.missing_modalities:
            mask = (torch.rand(text_tensor.shape[0]) < self.missing_modalities["text"])
            for i in range(batch_size):
                if mask[i]:
                    text_tensor[i] = 0  # 整个样本的 text 缺失

        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "visual_tensor": visual_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "sentence": sentence
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)
