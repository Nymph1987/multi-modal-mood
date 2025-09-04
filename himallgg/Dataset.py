import math
import random
import torch
import numpy as np

class Dataset:

    MODAL_MAP = {"t": "text", "a": "audio", "v": "vision"}  # 注意这里用 vision 而不是 visual

    def __init__(self, samples, batch_size, missing_rate=0.0, missing_modalities=None, seed=42, device="cuda:0"):
        self.device = device
        self.batch_size = batch_size
        # random.seed(seed)  # 注释掉这行，避免重复设置种子

        # ====== 自动转换 dict-of-lists 到 list-of-dicts ======
        if isinstance(samples, dict):
            num_samples = len(samples["text"])
            list_of_dicts = []
            for i in range(num_samples):
                sample = {k: samples[k][i] for k in samples}
                list_of_dicts.append(sample)
            self.samples = list_of_dicts
        elif isinstance(samples, list):
            self.samples = samples
        else:
            raise ValueError("samples must be a dict or a list of dicts")

        self.num_batches = math.ceil(len(self.samples) / batch_size)

        # ====== 缺失模态设置 ======
        if isinstance(missing_rate, float):
            self.missing_rate = {m: missing_rate for m in ["text", "audio", "vision"]}
        else:
            self.missing_rate = missing_rate

        if missing_modalities:
            self.missing_modalities = [self.MODAL_MAP[m] for m in missing_modalities]
        else:
            self.missing_modalities = ["text", "audio", "vision"]

        # ====== 调试打印 ======
        print("==== Dataset initialized ====")
        if len(self.samples) > 0:
            print(f"Sample keys: {list(self.samples[0].keys())}")
        print("============================")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, f"batch_idx {index} >= {self.num_batches}"
        return self.samples[index * self.batch_size: (index + 1) * self.batch_size]

    def padding(self, samples):
        batch_size = len(samples)

        # 转成 tensor
        text_list = [torch.tensor(s["text"], dtype=torch.float32) for s in samples]
        audio_list = [torch.tensor(s["audio"], dtype=torch.float32) for s in samples]
        vision_list = [torch.tensor(s["vision"], dtype=torch.float32) for s in samples]
        label_list = [torch.tensor(s["classification_labels"], dtype=torch.long) for s in samples]

        # 找到最大序列长度
        max_text_len = max([x.shape[0] for x in text_list])
        max_label_len = max([x.shape[0] for x in label_list])
        pad_len = max(max_text_len, max_label_len)

        def pad_sequence(seq, pad_len, pad_value=0):
            cur_len = seq.shape[0]
            if cur_len == pad_len:
                return seq
            pad_shape = (pad_len - cur_len,) + seq.shape[1:]
            pad_tensor = torch.full(pad_shape, pad_value, dtype=seq.dtype)
            return torch.cat([seq, pad_tensor], dim=0)

        # padding 并堆叠
        text_tensor_full = torch.stack([pad_sequence(s, pad_len) for s in text_list])
        audio_tensor_full = torch.stack([pad_sequence(s, pad_len) for s in audio_list])
        vision_tensor_full = torch.stack([pad_sequence(s, pad_len) for s in vision_list])

        # 改为序列标签 + padding，形状 [B, pad_len]
        label_tensor = torch.stack([pad_sequence(s, pad_len, pad_value=0) for s in label_list])

        text_len_tensor = torch.tensor([len(s) for s in text_list], dtype=torch.long)

        # 初始化缺失版本
        text_tensor = text_tensor_full.clone()
        audio_tensor = audio_tensor_full.clone()
        vision_tensor = vision_tensor_full.clone()

        mask_audio = torch.ones((batch_size, pad_len), dtype=torch.long)
        mask_visual = torch.ones((batch_size, pad_len), dtype=torch.long)

        # ====== 随机缺失模态 ======
        for i in range(batch_size):
            if "audio" in self.missing_modalities and random.random() < self.missing_rate.get("audio", 0):
                audio_tensor[i] = 0.0
                mask_audio[i] = 0
            if "vision" in self.missing_modalities and random.random() < self.missing_rate.get("vision", 0):
                vision_tensor[i] = 0.0
                mask_visual[i] = 0
            if "text" in self.missing_modalities and random.random() < self.missing_rate.get("text", 0):
                text_tensor[i] = 0.0
                # text 没有对应 mask，直接置 0 即可

        # ====== Speaker padding（如果样本里有 speakers） ======
        if "speakers" in samples[0]:
            speaker_tensor = torch.stack([
                pad_sequence(torch.tensor(s["speakers"], dtype=torch.long), pad_len, pad_value=0)
                for s in samples
            ])
        else:
            speaker_tensor = torch.zeros((batch_size, pad_len), dtype=torch.long)

        # ====== 移动到 device ======
        text_tensor = text_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        vision_tensor = vision_tensor.to(self.device)
        text_tensor_full = text_tensor_full.to(self.device)
        audio_tensor_full = audio_tensor_full.to(self.device)
        vision_tensor_full = vision_tensor_full.to(self.device)
        label_tensor = label_tensor.to(self.device)
        text_len_tensor = text_len_tensor.to(self.device)
        mask_audio = mask_audio.to(self.device)
        mask_visual = mask_visual.to(self.device)
        speaker_tensor = speaker_tensor.to(self.device)

        return {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "vision_tensor": vision_tensor,
            "label_tensor": label_tensor,
            "text_tensor_full": text_tensor_full,
            "audio_tensor_full": audio_tensor_full,
            "vision_tensor_full": vision_tensor_full,
            "mask_audio": mask_audio,
            "mask_visual": mask_visual,
            "speaker_tensor": speaker_tensor,
        }
