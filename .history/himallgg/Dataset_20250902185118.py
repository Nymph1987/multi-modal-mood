import torch
from torch.utils.data import Dataset

class MMDataset(Dataset):
    def __init__(self, samples, speaker_to_idx=None, device="cpu"):
        """
        samples: dict of lists, keys 包含
            'raw_text', 'text', 'text_bert', 'audio', 'vision',
            'id', 'audio_lengths', 'vision_lengths',
            'annotations', 'classification_labels', 'regression_labels'
        speaker_to_idx: dict, 如果需要映射 speaker -> idx
        """
        self.samples = samples
        self.speaker_to_idx = speaker_to_idx if speaker_to_idx else {}
        self.device = device

    def __len__(self):
        return len(self.samples["raw_text"])

    def __getitem__(self, idx):
        """返回单个样本的 dict"""
        return {k: v[idx] for k, v in self.samples.items()}

    def collate_fn(self, batch):
        """
        batch: list of dict
        return: dict of torch.Tensor
        """
        # 把 batch 合并成 dict of lists
        batch_dict = {k: [d[k] for d in batch] for k in batch[0]}

        return self.padding(batch_dict)

    def padding(self, samples):
        """
        samples: dict of lists
        return: dict of torch.Tensor
        """
        texts = samples["text"]
        audios = samples["audio"]
        visuals = samples["vision"]
        labels = samples["classification_labels"]
        raw_texts = samples["raw_text"]

        batch_size = len(texts)
        text_len_tensor = torch.tensor([len(t) for t in texts]).long()
        mx = torch.max(text_len_tensor).item()

        # 初始化张量 (假设特征维度固定：text=1024, audio=1582, vision=342)
        text_tensor = torch.zeros((batch_size, mx, 1024))
        audio_tensor = torch.zeros((batch_size, mx, 1582))
        visual_tensor = torch.zeros((batch_size, mx, 342))
        speaker_tensor = torch.zeros((batch_size, mx)).long()

        # mask
        mask_text = torch.zeros((batch_size, mx)).bool()
        mask_audio = torch.zeros((batch_size, mx)).bool()
        mask_visual = torch.zeros((batch_size, mx)).bool()

        for i in range(batch_size):
            cur_len = len(texts[i])

            # text
            text_tensor[i, :cur_len, :] = torch.stack(
                [torch.from_numpy(t).float() for t in texts[i]]
            )
            mask_text[i, :cur_len] = 1

            # audio
            audio_tensor[i, :cur_len, :] = torch.stack(
                [torch.from_numpy(a).float() for a in audios[i]]
            )
            mask_audio[i, :cur_len] = 1

            # vision
            visual_tensor[i, :cur_len, :] = torch.stack(
                [torch.from_numpy(v).float() for v in visuals[i]]
            )
            mask_visual[i, :cur_len] = 1

            # speaker (如果有)
            if "speakers" in samples:
                speaker_tensor[i, :cur_len] = torch.tensor(
                    [self.speaker_to_idx.get(s, 0) for s in samples["speakers"][i]]
                )

        # labels
        label_tensor = torch.tensor(labels).long()

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
