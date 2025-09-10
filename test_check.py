import pickle
from collections import Counter

# 加载数据
pkl_path = "data/iemocap/newdata.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f, encoding='latin1')  # {'train': {...}, 'dev': {...}, 'test': {...}}

def count_labels(split_data):
    """统计每个 split 中的标签数量，支持单标签或多标签"""
    all_labels = []
    for lbl in split_data['classification_labels']:
        if isinstance(lbl, list):
            all_labels.extend(lbl)
        else:
            all_labels.append(lbl)
    return Counter(all_labels)

# 打印各 split 标签分布
for split_name in ['train', 'dev', 'test']:
    counter = count_labels(data[split_name])
    print(f"=== {split_name.upper()} ===")
    for label in sorted(counter):
        print(f"标签 {label}: {counter[label]} 个样本")
    print()



import pickle
from collections import Counter

# 修改后的 PKL 文件路径
pkl_path = "data/iemocap/iemocap_features_roberta_modified.pkl"

# 读取 PKL
with open(pkl_path, "rb") as f:
    content = pickle.load(f)

# 解包
speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, train_ids, test_ids, valid_ids = content

# 统计总类别
all_labels = []
for lbls in labels.values():
    if isinstance(lbls, list):
        all_labels.extend(lbls)
    else:
        all_labels.append(lbls)

unique_classes = sorted(set(all_labels))
print(f"总共有 {len(unique_classes)} 类：{unique_classes}\n")

# 定义统计函数
def count_labels(ids, split_name):
    counter = Counter()
    for vid in ids:
        lbls = labels.get(vid, [])
        if isinstance(lbls, list):
            counter.update(lbls)
        elif lbls is not None:
            counter.update([lbls])
    print(f"{split_name} 集每类样本数量：")
    for cls in unique_classes:
        print(f"标签 {cls}: {counter.get(cls, 0)} 个样本")
    print()

# 统计各个 split
count_labels(train_ids, "Train")
count_labels(valid_ids, "Valid")
count_labels(test_ids, "Test")

import pickle
from collections import Counter

pkl_path = "data/iemocap/IEMOCAP_features_4class.pkl"

video_ids, video_speakers, video_labels, video_text, \
video_audio, video_visual, video_sentence, train_ids, test_ids = pickle.load(open(pkl_path, "rb"), encoding='latin1')

# 划分 dev（train 的 10%）
dev_size = int(len(train_ids) * 0.1)
train_vids, dev_vids = train_ids[dev_size:], train_ids[:dev_size]

splits = {"train": train_vids, "dev": dev_vids, "test": test_ids}

for split_name, vids in splits.items():
    all_labels = []
    for vid in vids:
        lbls = video_labels[vid]
        if isinstance(lbls, list):
            all_labels.extend(lbls)
        else:
            all_labels.append(lbls)
    counter = Counter(all_labels)
    print(f"\n=== {split_name} ===")
    for k in sorted(counter.keys()):
        print(f"Label {k}: {counter[k]} samples")
