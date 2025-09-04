# 文件顶部 import 区域
import math
import random
from transformers import BertTokenizer,BertModel
import torch


class Dataset(object):

    def __init__(self, samples, device='cpu', batch_size=32, missing_rate: float = 0.0, missing_modalities: list = None):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size)
        self.speaker_to_idx = {'M': 0, 'F': 1}
        # missing configuration
        self.missing_rate = missing_rate
        # 支持 ['t','a','v'] 或 []（空表示不生成缺失）
        self.missing_modalities = missing_modalities or []


    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def _apply_missing(self, tensor, lengths, modality_name):
        """ Randomly mask a portion (missing_rate) of time steps for the specified modality per sample. """
        if modality_name not in self.missing_modalities or self.missing_rate <= 0:
            mask = torch.ones_like(lengths).unsqueeze(-1)  # placeholder per-sample mask
            return tensor, tensor.clone(), mask
        B, L, D = tensor.shape
        device = tensor.device
        full = tensor.clone()
        masked = tensor.clone()
        mask = torch.zeros(B, L, device=device)
        for i in range(B):
            valid_len = lengths[i].item()
            if valid_len <= 0:
                continue
            num_missing = max(1, int(valid_len * self.missing_rate))
            idx = torch.randperm(valid_len, device=device)[:num_missing]
            masked[i, idx, :] = 0.0
            mask[i, :valid_len] = 1.0
        return masked, full, mask
    
    #padding 随机mask
    # def padding(self, samples):
    #     print("==== Checking samples ====")
    #     print("Samples length:", len(samples))
    #     print("First sample type:", type(samples[0]))
    #     print("First sample content:", samples[0])

    #     batch_size = len(samples)
    #     text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
    #     mx = torch.max(text_len_tensor).item()
        
    #     text_tensor = torch.zeros((batch_size, mx, 1024))
    #     audio_tensor = torch.zeros((batch_size, mx, 1582))
    #     visual_tensor = torch.zeros((batch_size, mx, 342))
    #     speaker_tensor = torch.zeros((batch_size, mx)).long()
    #     labels = []
    #     sentence = []
    #     for i, s in enumerate(samples):
    #         cur_len = len(s.text)
    #         tmp = [torch.from_numpy(t).float() for t in s.text]

    #         tmp = torch.stack(tmp)
    #         text_tensor[i, :cur_len, :] = tmp

    #         # add multi-modal info
    #         tmp = [torch.from_numpy(t).float() for t in s.audio]
    #         tmp = torch.stack(tmp)
    #         audio_tensor[i, :cur_len, :] = tmp


    #         tmp = [torch.from_numpy(t).float() for t in s.visual]
    #         tmp = torch.stack(tmp)
    #         visual_tensor[i, :cur_len, :] = tmp

            
    #         speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])
    #         labels.extend(s.label)
    #         sentence.extend([s.sentence])
    

    #     label_tensor = torch.tensor(labels).long()

    #     # full copies
    #     text_tensor_full = text_tensor.clone()
    #     audio_tensor_full = audio_tensor.clone()
    #     visual_tensor_full = visual_tensor.clone()

    #     # generate random masks per position (respecting sequence length)
    #     if self.missing_rate > 0.0:
    #         mask_base = torch.zeros((batch_size, mx))
    #         for i, L in enumerate(text_len_tensor.tolist()):
    #             if L > 0:
    #                 keep = torch.bernoulli(torch.full((L,), 1.0 - self.missing_rate))
    #                 if keep.sum() == 0:
    #                     keep[random.randrange(L)] = 1.0
    #                 mask_base[i, :L] = keep
    #         mask_base = mask_base.unsqueeze(-1)  # [B,L,1]
    #         mask_t = mask_base.clone() if ('t' in self.missing_modalities) else torch.ones_like(mask_base)
    #         mask_a = mask_base.clone() if ('a' in self.missing_modalities) else torch.ones_like(mask_base)
    #         mask_v = mask_base.clone() if ('v' in self.missing_modalities) else torch.ones_like(mask_base)
    #     else:
    #         mask_t = torch.ones((batch_size, mx, 1))
    #         mask_a = torch.ones((batch_size, mx, 1))
    #         mask_v = torch.ones((batch_size, mx, 1))

    #     # apply masks to obtain missing-view tensors
    #     text_tensor_missing = self._apply_missing_tensor(text_tensor, mask_t.squeeze(-1))
    #     audio_tensor_missing = self._apply_missing_tensor(audio_tensor, mask_a.squeeze(-1))
    #     visual_tensor_missing = self._apply_missing_tensor(visual_tensor, mask_v.squeeze(-1))

    #     data = {
    #         "text_len_tensor": text_len_tensor,
    #         # 缺失视图
    #         "text_tensor": text_tensor_missing,
    #         "audio_tensor": audio_tensor_missing,
    #         "visual_tensor": visual_tensor_missing,
    #         # 完整视图
    #         "text_tensor_full": text_tensor_full,
    #         "audio_tensor_full": audio_tensor_full,
    #         "visual_tensor_full": visual_tensor_full,
    #         # mask（1=保留/观察到，0=缺失，用于重建损失）
    #         "mask_text": mask_t.squeeze(-1).long(),
    #         "mask_audio": mask_a.squeeze(-1).long(),
    #         "mask_visual": mask_v.squeeze(-1).long(),
    #         "speaker_tensor": speaker_tensor,
    #         "label_tensor": label_tensor,
    #         'sentence': sentence
    #     }

    #     return data
    def padding(self, samples):
        print("==== Checking samples ====")
        print("Samples length:", len(samples))
        print("First sample type:", type(samples[0]))
        #print("First sample keys:", samples[0].keys() if isinstance(samples[0], dict) else dir(samples[0]))

        batch_size = len(samples)

        # 用 dict 形式访问
        text_len_tensor = torch.tensor([len(s['text']) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        
        text_tensor = torch.zeros((batch_size, mx, 1024))
        audio_tensor = torch.zeros((batch_size, mx, 1582))
        visual_tensor = torch.zeros((batch_size, mx, 342))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        sentence = []

        for i, s in enumerate(samples):
            cur_len = len(s['text'])
            tmp = [torch.from_numpy(t).float() for t in s.get('text', [])]
            tmp = torch.stack(tmp)
            text_tensor[i, :cur_len, :] = tmp

            tmp = [torch.from_numpy(t).float() for t in s.get('audio', [])]
            if tmp:
                tmp = torch.stack(tmp)
                audio_tensor[i, :cur_len, :] = tmp

            tmp = [torch.from_numpy(t).float() for t in s.get('visual', [])]
            if tmp:
                tmp = torch.stack(tmp)
                visual_tensor[i, :cur_len, :] = tmp

            # speaker
            speaker_tensor[i, :cur_len] = torch.tensor(
                [self.speaker_to_idx.get(c, 0) for c in s.get('speaker', ['M']*cur_len)])
            
            labels.extend(s.get('label', [0]*cur_len))
            sentence.extend([s.get('sentence', "")])

        label_tensor = torch.tensor(labels).long()

        # full copies
        text_tensor_full = text_tensor.clone()
        audio_tensor_full = audio_tensor.clone()
        visual_tensor_full = visual_tensor.clone()

        # masks
        if self.missing_rate > 0.0:
            mask_base = torch.zeros((batch_size, mx))
            for i, L in enumerate(text_len_tensor.tolist()):
                if L > 0:
                    keep = torch.bernoulli(torch.full((L,), 1.0 - self.missing_rate))
                    if keep.sum() == 0:
                        keep[random.randrange(L)] = 1.0
                    mask_base[i, :L] = keep
            mask_base = mask_base.unsqueeze(-1)
            mask_t = mask_base.clone() if ('t' in self.missing_modalities) else torch.ones_like(mask_base)
            mask_a = mask_base.clone() if ('a' in self.missing_modalities) else torch.ones_like(mask_base)
            mask_v = mask_base.clone() if ('v' in self.missing_modalities) else torch.ones_like(mask_base)
        else:
            mask_t = torch.ones((batch_size, mx, 1))
            mask_a = torch.ones((batch_size, mx, 1))
            mask_v = torch.ones((batch_size, mx, 1))

        # apply masks
        text_tensor_missing = self._apply_missing_tensor(text_tensor, mask_t.squeeze(-1))
        audio_tensor_missing = self._apply_missing_tensor(audio_tensor, mask_a.squeeze(-1))
        visual_tensor_missing = self._apply_missing_tensor(visual_tensor, mask_v.squeeze(-1))

        data = {
            "text_len_tensor": text_len_tensor,
            "text_tensor": text_tensor_missing,
            "audio_tensor": audio_tensor_missing,
            "visual_tensor": visual_tensor_missing,
            "text_tensor_full": text_tensor_full,
            "audio_tensor_full": audio_tensor_full,
            "visual_tensor_full": visual_tensor_full,
            "mask_text": mask_t.squeeze(-1).long(),
            "mask_audio": mask_a.squeeze(-1).long(),
            "mask_visual": mask_v.squeeze(-1).long(),
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            'sentence': sentence
        }

        return data


    def shuffle(self):
        random.shuffle(self.samples)

    def _apply_missing_tensor(self, tensor, mask):
        """Apply missing mask to tensor: mask 0 means set to zero (missing)"""
        return tensor * mask.unsqueeze(-1).float()


def tokenize(tokenizer,data):
        TEXT_length = max(data['text_len_tensor'].tolist())
        ad = []
        sd = []
        for paragraph in data['sentence']:
            for text in paragraph:
                tokened = tokenizer(text)
                input_ids = tokened['input_ids']
                mask = tokened['attention_mask']
                if len(input_ids) < TEXT_length:
                    pad_len = (TEXT_length - len(input_ids))
                    input_ids += [0] * pad_len
                    mask += [0] * pad_len
            ad.append(input_ids[:TEXT_length])
            sd.append(mask[:TEXT_length])
        return torch.tensor(ad),torch.tensor(sd)




