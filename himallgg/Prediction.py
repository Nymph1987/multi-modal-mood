# import torch
# import torch.nn.functional as F
# from sklearn import metrics
# from tqdm import tqdm
# import himallgg

# log = himallgg.utils.get_logger()


# class Prediction:
#     """
#     单独的评估类，用于评估训练好的 LGGCN 模型在任意 Dataset 上的表现。
#     支持 batch_size >1, padding mask 自动处理
#     """

#     def __init__(self, model, dataset, device="cuda:0"):
#         self.model = model
#         self.dataset = dataset
#         self.device = device

#     def evaluate(self, batch_size=None):
#         """
#         对 dataset 做评估
#         返回 weighted f1-score
#         """
#         self.model.eval()
#         golds, preds = [], []

#         with torch.no_grad():
#             for batch_idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
#                 data = self.dataset[batch_idx]

#                 # 确保 tensor 在 device 上
#                 for k, v in data.items():
#                     if k != "sentence":  # 句子文本不用转
#                         data[k] = v.to(self.device)

#                 # 模型预测
#                 y_hat = self.model(data)  # [total_nodes]

#                 # 展开 gold label 按有效长度
#                 text_len_tensor = data["text_len_tensor"]  # [B]
#                 label_tensor = data["label_tensor"]        # [B, pad_len]

#                 label_list = [label_tensor[i, :L].cpu() for i, L in enumerate(text_len_tensor)]
#                 gold_nodes = torch.cat(label_list, dim=0)

#                 golds.extend(gold_nodes.numpy().tolist())
#                 preds.extend(y_hat.cpu().numpy().tolist())

#         assert len(golds) == len(preds), f"Length mismatch: golds={len(golds)}, preds={len(preds)}"

#         print(metrics.classification_report(golds, preds, digits=4))
#         f1 = metrics.f1_score(golds, preds, average="weighted")
#         log.info(f"Weighted F1-score: {f1:.4f}")
#         return f1


import torch
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

class Prediction:
    def __init__(self, model, dataset, device="cuda"):
        self.model = model
        self.dataset = dataset
        self.device = device

    def move_to_device(self, batch, device):
        """递归搬到 GPU"""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                batch = self.dataset[idx]
                batch = self.move_to_device(batch, self.device)

                # 与训练一致：模型 forward 返回逐节点预测 y_hat
                y_hat = self.model(batch)  # 形状展平后与有效节点数一致

                # 展开 gold label（仅取有效长度部分）
                text_len_tensor = batch["text_len_tensor"]  # [B]
                label_tensor = batch["label_tensor"]        # [B, pad_len]
                label_list = [label_tensor[i, :L].cpu() for i, L in enumerate(text_len_tensor)]
                gold_nodes = torch.cat(label_list, dim=0)   # [total_valid_nodes]

                y_pred.extend(y_hat.view(-1).cpu().numpy().tolist())
                y_true.extend(gold_nodes.numpy().tolist())

        f1 = f1_score(y_true, y_pred, average="weighted")
        print(classification_report(y_true, y_pred, digits=4))
        return f1
