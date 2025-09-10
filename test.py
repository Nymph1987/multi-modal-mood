import pickle
import os

# ===== 修改 roberta PKL =====
pkl_path = "data/iemocap/iemocap_features_roberta.pkl"
new_pkl_path = "data/iemocap/iemocap_features_roberta_modified.pkl"

with open(pkl_path, "rb") as f:
    content = pickle.load(f)

# 解包
speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, train_ids, test_ids, valid_ids = content

# 标签修改：类别4->0，类别5删除
new_labels = {}
for vid, lbl in labels.items():
    if isinstance(lbl, list):
        new_lbl = [0 if l == 4 else l for l in lbl if l != 5]  # 4 -> 0, 5 删除
        if new_lbl:
            new_labels[vid] = new_lbl
    else:
        if lbl == 4:
            new_labels[vid] = 0  # 4 -> 0
        elif lbl != 5:
            new_labels[vid] = lbl
        # lbl==5 不加入 new_labels

new_content = [
    speakers,
    new_labels,
    roberta1,
    roberta2,
    roberta3,
    roberta4,
    sentences,
    train_ids,
    test_ids,
    valid_ids
]

with open(new_pkl_path, "wb") as f:
    pickle.dump(new_content, f)

print(f"✅ 修改完成，新文件已保存为 {new_pkl_path}")


# ===== 修改原始 IEMOCAP 特征 PKL =====
pkl_path = "data/iemocap/IEMOCAP_features.pkl"
new_pkl_path = "data/iemocap/IEMOCAP_features_4class.pkl"

with open(pkl_path, "rb") as f:
    content = pickle.load(f, encoding='latin1')

# 解包
video_ids, video_speakers, video_labels, video_text, \
video_audio, video_visual, video_sentence, train_ids, test_ids = content

# 创建新 dict
new_video_ids = []
new_video_speakers = {}
new_video_labels = {}
new_video_text = {}
new_video_audio = {}
new_video_visual = {}
new_video_sentence = {}

for vid in video_ids:
    lbls = video_labels[vid]
    if isinstance(lbls, list):
        new_lbls = []
        for l in lbls:
            if l == 5:
                continue  # 删除类别5
            elif l == 4:
                new_lbls.append(0)  # 4 -> 0
            else:
                new_lbls.append(l)
        if not new_lbls:
            continue
        new_video_labels[vid] = new_lbls
    else:
        if lbls == 5:
            continue
        elif lbls == 4:
            new_video_labels[vid] = 0  # 4 -> 0
        else:
            new_video_labels[vid] = lbls

    # 保留对应特征
    new_video_ids.append(vid)
    new_video_speakers[vid] = video_speakers[vid]
    new_video_text[vid] = video_text[vid]
    new_video_audio[vid] = video_audio[vid]
    new_video_visual[vid] = video_visual[vid]
    new_video_sentence[vid] = video_sentence[vid]

# 更新 train/test 列表，只保留存在的新视频 id
new_train_ids = [vid for vid in train_ids if vid in new_video_ids]
new_test_ids = [vid for vid in test_ids if vid in new_video_ids]

# 保存新的 PKL
new_content = [
    new_video_ids,
    new_video_speakers,
    new_video_labels,
    new_video_text,
    new_video_audio,
    new_video_visual,
    new_video_sentence,
    new_train_ids,
    new_test_ids
]

os.makedirs(os.path.dirname(new_pkl_path), exist_ok=True)
with open(new_pkl_path, "wb") as f:
    pickle.dump(new_content, f)

print(f"✅ 修改完成，新文件已保存为 {new_pkl_path}")
print(f"原视频数量: {len(video_ids)}, 新视频数量: {len(new_video_ids)}")
