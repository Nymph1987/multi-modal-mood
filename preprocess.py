# import argparse
# from tqdm import tqdm
# import pickle
# import logging
# import os
# import himallgg

# # 初始化 logger
# log = himallgg.utils.get_logger()
# log.setLevel(logging.INFO)  # 确保 info 日志可以显示


# # =============== IEMOCAP 处理 ===================
# def split_iemocap(args):
#     himallgg.utils.set_seed(args.seed)

#     # 加载原始特征
#     video_ids, video_speakers, video_labels, video_text, \
#         video_audio, video_visual, video_sentence, trainVids, \
#         test_vids = pickle.load(open('data/iemocap/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

#     # 加载 roberta 特征（可选）
#     [speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences,
#      train_ids, test_ids, valid_ids] = pickle.load(
#         open('data/iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')

#     # 划分 train/dev/test
#     dev_size = int(len(trainVids) * 0.1)
#     trainVids = list(trainVids)
#     train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

#     splits = {'train': train_vids, 'dev': dev_vids, 'test': test_vids}
#     out_data = {}

#     for split_name, vids in splits.items():
#         split_dict = {
#             'raw_text': [], 'text': [], 'text_bert': [], 'audio': [], 'vision': [],
#             'id': [], 'audio_lengths': [], 'vision_lengths': [],
#             'annotations': [], 'classification_labels': [], 'regression_labels': []
#         }
#         for vid in tqdm(vids, desc=f"Building {split_name}"):
#             raw_text = video_sentence[vid]
#             text_feat = roberta1[vid] if roberta1 is not None else video_text[vid]
#             audio_feat = video_audio[vid]
#             vision_feat = video_visual[vid]

#             split_dict['raw_text'].append(raw_text)
#             split_dict['text'].append(text_feat)
#             split_dict['text_bert'].append(text_feat)
#             split_dict['audio'].append(audio_feat)
#             split_dict['vision'].append(vision_feat)
#             split_dict['id'].append([vid])
#             split_dict['audio_lengths'].append([len(audio_feat)])
#             split_dict['vision_lengths'].append([len(vision_feat)])
#             split_dict['annotations'].append(video_labels[vid])
#             split_dict['classification_labels'].append(video_labels[vid])
#             split_dict['regression_labels'].append(video_labels[vid])

#         out_data[split_name] = split_dict

#     return out_data


# # =============== MOSI 处理 ===================
# def split_mosi(args):
#     pkl_path = 'data/mosi/unaligned_50.pkl'
#     if not os.path.exists(pkl_path):
#         raise FileNotFoundError(f"Not found MOSI features at {pkl_path}")

#     pack = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

#     splits = {'train': 'train', 'dev': 'valid', 'test': 'test'}
#     out_data = {}

#     for split_name, key in splits.items():
#         src = pack[key]
#         split_dict = {
#             'raw_text': src['raw_text'],
#             'text': src['text'],
#             'text_bert': src['text_bert'],
#             'audio': src['audio'],
#             'vision': src['vision'],
#             'id': src['id'],
#             'audio_lengths': src['audio_lengths'],
#             'vision_lengths': src['vision_lengths'],
#             'annotations': src['annotations'],
#             'classification_labels': src['classification_labels'],
#             'regression_labels': src['regression_labels']
#         }
#         out_data[split_name] = split_dict

#     return out_data


# # =============== 主入口 ===================
# def main(args):
#     himallgg.utils.set_seed(args.seed)

#     if args.dataset == 'iemocap':
#         data = split_iemocap(args)
#     elif args.dataset == 'mosi':
#         data = split_mosi(args)
#     else:
#         raise ValueError(f"Unsupported dataset: {args.dataset}")

#     # 保存预处理好的原始数据
#     himallgg.utils.save_pkl(data, args.data)
#     log.info(f"Preprocessed data saved to {args.data}")

#     if args.generate_missing:
#         log.info("Note: Missing-view dataset will be generated dynamically in Dataset class using "
#                  f"missing_rate={args.missing_rate}, missing_modalities={args.missing_modalities}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="preprocess.py")
#     parser.add_argument("--data", type=str, required=True, help="Path to save processed data")
#     parser.add_argument("--dataset", type=str, required=True, choices=["iemocap", "mosi"], help="Dataset name.")
#     parser.add_argument("--seed", type=int, default=1024, help="Random seed.")
#     parser.add_argument("--generate_missing", action="store_true",
#                         help="Enable missing-view dataset (handled dynamically in Dataset class)")
#     parser.add_argument("--missing_rate", type=float, default=0.0,
#                         help="Missing rate for missing-view (used in Dataset class)")
#     parser.add_argument("--missing_modalities", type=str, default="av",
#                         help="Modalities to apply missing, from 't','a','v' (used in Dataset class)")

#     args = parser.parse_args()
#     main(args)




import argparse
from tqdm import tqdm
import pickle
import logging
import os
import random
import numpy as np
import himallgg

# 初始化 logger
log = himallgg.utils.get_logger()
log.setLevel(logging.INFO)  # 确保 info 日志可以显示


def split_iemocap(args):
    himallgg.utils.set_seed(args.seed)

    # 加载原始特征
    video_ids, video_speakers, video_labels, video_text, \
        video_audio, video_visual, video_sentence, trainVids, \
        test_vids = pickle.load(open('data/iemocap/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    # 加载 roberta 特征（可选）
    [speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, train_ids, test_ids, valid_ids] = \
        pickle.load(open('data/iemocap/iemocap_features_roberta.pkl','rb'))

    # 划分 train/dev/test
    dev_size = int(len(trainVids) * 0.1)
    trainVids = list(trainVids)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    splits = {'train': train_vids, 'dev': dev_vids, 'test': test_vids}
    out_data = {}

    for split_name, vids in splits.items():
        split_dict = {'raw_text': [], 'text': [], 'text_bert': [], 'audio': [], 'vision': [],
                      'id': [], 'audio_lengths': [], 'vision_lengths': [],
                      'annotations': [], 'classification_labels': [], 'regression_labels': []}
        for vid in tqdm(vids, desc=f"Building {split_name}"):
            raw_text = video_sentence[vid]
            text_feat = roberta1[vid] if roberta1 is not None else video_text[vid]
            audio_feat = video_audio[vid]
            vision_feat = video_visual[vid]
            split_dict['raw_text'].append(raw_text)
            split_dict['text'].append(text_feat)
            split_dict['text_bert'].append(text_feat)
            split_dict['audio'].append(audio_feat)
            split_dict['vision'].append(vision_feat)
            split_dict['id'].append([vid])
            split_dict['audio_lengths'].append([len(audio_feat)])
            split_dict['vision_lengths'].append([len(vision_feat)])
            split_dict['annotations'].append(video_labels[vid])
            split_dict['classification_labels'].append(video_labels[vid])
            split_dict['regression_labels'].append(video_labels[vid])
        out_data[split_name] = split_dict

    return out_data


def split_mosi(args):
    pkl_path = 'data/mosi/unaligned_50.pkl'  # 你的实际 MOSI 文件
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Not found MOSI features at {pkl_path}")

    pack = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

    # 使用已有 split
    splits = {'train': 'train', 'dev': 'valid', 'test': 'test'}
    out_data = {}

    for split_name, key in splits.items():
        src = pack[key]  # src 是 dict
        split_dict = {'raw_text': src['raw_text'],
                      'text': src['text'],
                      'text_bert': src['text_bert'],
                      'audio': src['audio'],
                      'vision': src['vision'],
                      'id': src['id'],
                      'audio_lengths': src['audio_lengths'],
                      'vision_lengths': src['vision_lengths'],
                      'annotations': src['annotations'],
                      'classification_labels': src['classification_labels'],
                      'regression_labels': src['regression_labels']}
        out_data[split_name] = split_dict

    return out_data


# def generate_missing_dataset(data, missing_rate: float, missing_modalities: str):
#     mods = set(list(missing_modalities.lower()))

#     def mask_seq(seq):
#         L = len(seq)
#         if L == 0: return seq
#         keep = np.random.binomial(1, 1.0 - missing_rate, size=L)
#         if keep.sum() == 0:
#             keep[np.random.randint(0, L)] = 1
#         out = []
#         for i, v in enumerate(seq):
#             if keep[i] == 1:
#                 out.append(v)
#             else:
#                 out.append(np.zeros_like(v))
#         return out

#     def clone_and_mask(samples):
#         out = {}
#         for key in samples:
#             if key in ['text', 'text_bert', 'audio', 'vision']:
#                 out[key] = [mask_seq(s) for s in samples[key]]
#             else:
#                 out[key] = samples[key]
#         return out

#     miss_data = {}
#     for split in ['train', 'dev', 'test']:
#         miss_data[split] = clone_and_mask(data[split])
#     return miss_data

def generate_missing_dataset(data, missing_rate: float, missing_modalities: str):
    mods = set(list(missing_modalities.lower()))
    # 定义模态顺序：text(0), audio(1), vision(2)
    mod_order = {'t': 0, 'a': 1, 'v': 2}
    target_mods = [mod_order[m] for m in mods if m in mod_order]

    def mask_seq(seq):
        L = len(seq)
        if L == 0: return seq
        keep = np.random.binomial(1, 1.0 - missing_rate, size=L)
        if keep.sum() == 0:
            keep[np.random.randint(0, L)] = 1
        out = []
        for i, v in enumerate(seq):
            if keep[i] == 1:
                out.append(v)
            else:
                out.append(np.zeros_like(v))
        return out

    def clone_and_mask(samples):
        out = {}
        # 初始化modal_mask：默认全为1（所有模态存在）
        batch_size = len(samples['text'])
        modal_mask = np.ones((batch_size, 3), dtype=int)
        
        for key in samples:
            if key in ['text', 'text_bert'] and 0 in target_mods:
                # 对文本模态进行掩码，并标记mask
                out[key] = [mask_seq(s) for s in samples[key]]
                # 这里简化处理：只要有缺失帧就认为模态缺失（可根据需求调整）
                for i in range(batch_size):
                    if np.any([np.all(v == 0) for v in out[key][i]]):
                        modal_mask[i, 0] = 0
            elif key == 'audio' and 1 in target_mods:
                out[key] = [mask_seq(s) for s in samples[key]]
                for i in range(batch_size):
                    if np.any([np.all(v == 0) for v in out[key][i]]):
                        modal_mask[i, 1] = 0
            elif key == 'vision' and 2 in target_mods:
                out[key] = [mask_seq(s) for s in samples[key]]
                for i in range(batch_size):
                    if np.any([np.all(v == 0) for v in out[key][i]]):
                        modal_mask[i, 2] = 0
            else:
                out[key] = samples[key]
        
        # 添加modal_mask字段
        out['modal_mask'] = modal_mask.tolist()
        return out

    miss_data = {}
    for split in ['train', 'dev', 'test']:
        miss_data[split] = clone_and_mask(data[split])
    return miss_data


def main(args):
    himallgg.utils.set_seed(args.seed)
    if args.dataset == 'iemocap':
        data = split_iemocap(args)
    elif args.dataset == 'mosi':
        data = split_mosi(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    himallgg.utils.save_pkl(data, args.data)
    log.info(f"Preprocessed data saved to {args.data}")

    if args.generate_missing:
        miss_data = generate_missing_dataset(data, args.missing_rate, args.missing_modalities)
        miss_out = os.path.splitext(args.data)[0] + "_missing.pkl"
        himallgg.utils.save_pkl(miss_data, miss_out)
        log.info(f"Missing-view dataset saved to {miss_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--dataset", type=str, required=True, choices=["iemocap", "mosi"], help="Dataset name.")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed.")
    parser.add_argument("--generate_missing", action="store_true", help="Also generate a missing-view dataset pkl")
    parser.add_argument("--missing_rate", type=float, default=0.0, help="Missing rate for generating missing dataset")
    parser.add_argument("--missing_modalities", type=str, default="av", help="Modalities to apply missing, from 't','a','v'")

    args = parser.parse_args()
    main(args)
