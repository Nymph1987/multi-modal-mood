import math
import random
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Dataset:
    MODAL_MAP = {"t": "text", "a": "audio", "v": "vision"}

    def __init__(self, samples, batch_size=1, missing_rate=0.0, missing_modalities=None, seed=42, device="cuda:0", num_workers=4):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        random.seed(seed)

        # dict-of-lists -> list-of-dicts
        if isinstance(samples, dict):
            num_samples = len(samples["text"])
            self.samples = [{k: samples[k][i] for k in samples} for i in range(num_samples)]
        elif isinstance(samples, list):
            self.samples = samples
        else:
            raise ValueError("samples must be dict or list of dicts")

        self.num_batches = math.ceil(len(self.samples) / batch_size)

        # 缺失模态
        if isinstance(missing_rate, float):
            self.missing_rate = {m: missing_rate for m in ["text", "audio", "vision"]}
        else:
            self.missing_rate = missing_rate

        if missing_modalities:
            self.missing_modalities = [self.MODAL_MAP[m] for m in missing_modalities]
        else:
            self.missing_modalities = ["text", "audio", "vision"]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches
        return self.samples[index * self.batch_size: (index + 1) * self.batch_size]

    # def padding(self, samples):
    #     batch_size = len(samples)

    #     # 使用线程池加速 tensor 转换
    #     with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    #         text_list = list(executor.map(lambda s: torch.tensor(s["text"], dtype=torch.float32), samples))
    #         audio_list = list(executor.map(lambda s: torch.tensor(s["audio"], dtype=torch.float32), samples))
    #         vision_list = list(executor.map(lambda s: torch.tensor(s["vision"], dtype=torch.float32), samples))

    #     max_len = max([t.shape[0] for t in text_list])

    #     def pad_seq(x, length, pad_value=0.0):
    #         if x.shape[0] == length:
    #             return x
    #         pad_shape = (length - x.shape[0],) + x.shape[1:]
    #         return torch.cat([x, torch.full(pad_shape, pad_value, dtype=x.dtype)], dim=0)

    #     # 并行 padding
    #     with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    #         text_tensor_full = torch.stack(list(executor.map(lambda t: pad_seq(t, max_len), text_list)))
    #         audio_tensor_full = torch.stack(list(executor.map(lambda a: pad_seq(a, max_len), audio_list)))
    #         vision_tensor_full = torch.stack(list(executor.map(lambda v: pad_seq(v, max_len), vision_list)))

    #     # masks
    #     text_len_tensor = torch.tensor([t.shape[0] for t in text_list], dtype=torch.long)
    #     mask_audio = torch.ones((batch_size, max_len), dtype=torch.long)
    #     mask_visual = torch.ones((batch_size, max_len), dtype=torch.long)

    #     text_tensor = text_tensor_full.clone()
    #     audio_tensor = audio_tensor_full.clone()
    #     vision_tensor = vision_tensor_full.clone()

    #     # 并行处理随机缺失模态
    #     def apply_missing(i):
    #         if "audio" in self.missing_modalities and random.random() < self.missing_rate.get("audio", 0):
    #             audio_tensor[i] = 0.0
    #             mask_audio[i] = 0
    #         if "vision" in self.missing_modalities and random.random() < self.missing_rate.get("vision", 0):
    #             vision_tensor[i] = 0.0
    #             mask_visual[i] = 0
    #         if "text" in self.missing_modalities and random.random() < self.missing_rate.get("text", 0):
    #             text_tensor[i] = 0.0

    #     with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    #         list(executor.map(apply_missing, range(batch_size)))

    #     # speaker padding
    #     if "speakers" in samples[0]:
    #         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
    #             speaker_tensor = torch.stack(
    #                 list(executor.map(lambda s: pad_seq(torch.tensor(s["speakers"], dtype=torch.long), max_len, 0), samples))
    #             )
    #     else:
    #         speaker_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)

    #     # labels flatten
    #     label_list = [torch.tensor(s["classification_labels"], dtype=torch.long) for s in samples]
    #     label_tensor = torch.cat(label_list, dim=0)
    #     label_len_tensor = torch.tensor([len(s["classification_labels"]) for s in samples], dtype=torch.long)

    #     # 移动到 device
    #     tensors_to_move = [text_tensor, audio_tensor, vision_tensor,
    #                        text_tensor_full, audio_tensor_full, vision_tensor_full,
    #                        text_len_tensor, mask_audio, mask_visual, speaker_tensor,
    #                        label_tensor, label_len_tensor]

    #     tensors_to_move = [t.to(self.device) for t in tensors_to_move]
    #     (text_tensor, audio_tensor, vision_tensor,
    #      text_tensor_full, audio_tensor_full, vision_tensor_full,
    #      text_len_tensor, mask_audio, mask_visual, speaker_tensor,
    #      label_tensor, label_len_tensor) = tensors_to_move

    #     return {
    #         "text_tensor_full": text_tensor_full,
    #         "audio_tensor_full": audio_tensor_full,
    #         "vision_tensor_full": vision_tensor_full,
    #         "text_tensor": text_tensor,
    #         "audio_tensor": audio_tensor,
    #         "vision_tensor": vision_tensor,
    #         "text_len_tensor": text_len_tensor,
    #         "mask_audio": mask_audio,
    #         "mask_visual": mask_visual,
    #         "speaker_tensor": speaker_tensor,
    #         "label_tensor": label_tensor,
    #         "label_len_tensor": label_len_tensor
    #     }
    
    def padding(self, samples):
        batch_size = len(samples)

        # -------------------------------
        # 将 modalities 转成 numpy array
        # -------------------------------
        text_np = [np.array(s["text"], dtype=np.float32) for s in samples]
        audio_np = [np.array(s["audio"], dtype=np.float32) for s in samples]
        vision_np = [np.array(s["vision"], dtype=np.float32) for s in samples]

        max_len = max([t.shape[0] for t in text_np])

        def pad_seq_np(x, length, pad_value=0.0):
            pad_len = length - x.shape[0]
            if pad_len > 0:
                pad_shape = ((0, pad_len),) + tuple((0,0) for _ in range(x.ndim-1))
                x = np.pad(x, pad_shape, mode='constant', constant_values=pad_value)
            return x

        text_tensor_full = torch.from_numpy(np.stack([pad_seq_np(t, max_len) for t in text_np])).float()
        audio_tensor_full = torch.from_numpy(np.stack([pad_seq_np(a, max_len) for a in audio_np])).float()
        vision_tensor_full = torch.from_numpy(np.stack([pad_seq_np(v, max_len) for v in vision_np])).float()

        # masks
        text_len_tensor = torch.tensor([t.shape[0] for t in text_np], dtype=torch.long)
        mask_audio = torch.ones((batch_size, max_len), dtype=torch.long)
        mask_visual = torch.ones((batch_size, max_len), dtype=torch.long)

        # 复制为可修改 tensor
        text_tensor = text_tensor_full.clone()
        audio_tensor = audio_tensor_full.clone()
        vision_tensor = vision_tensor_full.clone()

        # 随机缺失模态
        for i in range(batch_size):
            if "audio" in self.missing_modalities and random.random() < self.missing_rate.get("audio", 0):
                audio_tensor[i] = 0.0
                mask_audio[i] = 0
            if "vision" in self.missing_modalities and random.random() < self.missing_rate.get("vision", 0):
                vision_tensor[i] = 0.0
                mask_visual[i] = 0
            if "text" in self.missing_modalities and random.random() < self.missing_rate.get("text", 0):
                text_tensor[i] = 0.0

        # speaker padding
        if "speakers" in samples[0]:
            speaker_np = [np.array(s["speakers"], dtype=np.int64) for s in samples]
            speaker_tensor = torch.from_numpy(np.stack([pad_seq_np(s, max_len, 0) for s in speaker_np])).long()
        else:
            speaker_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)

        # labels flatten
        label_list = [torch.tensor(s["classification_labels"], dtype=torch.long) for s in samples]
        label_tensor = torch.cat(label_list, dim=0)
        label_len_tensor = torch.tensor([len(s["classification_labels"]) for s in samples], dtype=torch.long)

        # 移动到 device
        tensors_to_move = [text_tensor, audio_tensor, vision_tensor,
                        text_tensor_full, audio_tensor_full, vision_tensor_full,
                        text_len_tensor, mask_audio, mask_visual, speaker_tensor,
                        label_tensor, label_len_tensor]
        tensors_to_move = [t.to(self.device) for t in tensors_to_move]

        (text_tensor, audio_tensor, vision_tensor,
        text_tensor_full, audio_tensor_full, vision_tensor_full,
        text_len_tensor, mask_audio, mask_visual, speaker_tensor,
        label_tensor, label_len_tensor) = tensors_to_move

        return {
            "text_tensor_full": text_tensor_full,
            "audio_tensor_full": audio_tensor_full,
            "vision_tensor_full": vision_tensor_full,
            "text_tensor": text_tensor,
            "audio_tensor": audio_tensor,
            "vision_tensor": vision_tensor,
            "text_len_tensor": text_len_tensor,
            "mask_audio": mask_audio,
            "mask_visual": mask_visual,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "label_len_tensor": label_len_tensor
        }
