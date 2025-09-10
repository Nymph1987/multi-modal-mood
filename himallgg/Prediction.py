# import torch
# from torch.utils.data import DataLoader
# from sklearn.metrics import f1_score, classification_report
# from tqdm import tqdm

# class Prediction:
#     def __init__(self, model, dataset, device="cuda"):
#         self.model = model
#         self.dataset = dataset
#         self.device = device

#     def move_to_device(self, batch, device):
#         """递归搬到 GPU"""
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.to(device)
#         return batch

#     def evaluate(self):
#         """
#         基于 Dataset 顺序评估，展开有效长度后的节点标签，返回 weighted F1，并打印分类报告
#         """
#         import torch
#         from sklearn.metrics import f1_score, classification_report
#         from tqdm import tqdm

#         self.model.eval()
#         y_true, y_pred = [], []

#         with torch.no_grad():
#             for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
#                 batch = self.dataset[idx]
#                 batch = self.move_to_device(batch, self.device)

#                 # 模型输出：逐节点预测（形状已与有效节点对齐或可展平对齐）
#                 y_hat = self.model(batch)  # [total_valid_nodes_in_batch] 或可展平
#                 y_hat = y_hat.view(-1)

#                 # 取出有效长度部分的 gold labels
#                 text_len_tensor = batch["text_len_tensor"]  # [B]
#                 label_tensor = batch["label_tensor"]        # [B, pad_len]
#                 label_list = [label_tensor[i, :L].cpu() for i, L in enumerate(text_len_tensor)]
#                 gold_nodes = torch.cat(label_list, dim=0).view(-1)   # [total_valid_nodes_in_batch]

#                 assert y_hat.numel() == gold_nodes.numel(), \
#                     f"Length mismatch: preds={y_hat.numel()}, golds={gold_nodes.numel()}"

#                 y_pred.extend(y_hat.cpu().numpy().tolist())
#                 y_true.extend(gold_nodes.cpu().numpy().tolist())

#         print(classification_report(y_true, y_pred, digits=4))
#         f1 = f1_score(y_true, y_pred, average="weighted")
#         return f1



# # import torch
# # import torch.nn.functional as F
# # from sklearn import metrics
# # from tqdm import tqdm
# # import himallgg

# # log = himallgg.utils.get_logger()


# # class Prediction:
# #     """
# #     单独的评估类，用于评估训练好的 LGGCN 模型在任意 Dataset 上的表现。
# #     支持 batch_size >1, padding mask 自动处理
# #     """

# #     def __init__(self, model, dataset, device="cuda:0"):
# #         self.model = model
# #         self.dataset = dataset
# #         self.device = device

# #     def evaluate(self, batch_size=None):
# #         """
# #         对 dataset 做评估
# #         返回 weighted f1-score
# #         """
# #         self.model.eval()
# #         golds, preds = [], []

# #         with torch.no_grad():
# #             for batch_idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
# #                 data = self.dataset[batch_idx]

# #                 # 确保 tensor 在 device 上
# #                 for k, v in data.items():
# #                     if k != "sentence":  # 句子文本不用转
# #                         data[k] = v.to(self.device)

# #                 # 模型预测
# #                 y_hat = self.model(data)  # [total_nodes]

# #                 # 展开 gold label 按有效长度
# #                 text_len_tensor = data["text_len_tensor"]  # [B]
# #                 label_tensor = data["label_tensor"]        # [B, pad_len]

# #                 label_list = [label_tensor[i, :L].cpu() for i, L in enumerate(text_len_tensor)]
# #                 gold_nodes = torch.cat(label_list, dim=0)

# #                 golds.extend(gold_nodes.numpy().tolist())
# #                 preds.extend(y_hat.cpu().numpy().tolist())

# #         assert len(golds) == len(preds), f"Length mismatch: golds={len(golds)}, preds={len(preds)}"

# #         print(metrics.classification_report(golds, preds, digits=4))
# #         f1 = metrics.f1_score(golds, preds, average="weighted")
# #         log.info(f"Weighted F1-score: {f1:.4f}")
# #         return f1


# import torch
# from sklearn.metrics import f1_score, classification_report
# from tqdm import tqdm

# class Prediction:
#     def __init__(self, model, dataset, device="cuda"):
#         self.model = model
#         self.dataset = dataset
#         self.device = device

#     def move_to_device(self, batch, device):
#         """递归搬到 GPU"""
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.to(device)
#         return batch

#     def evaluate(self):
#         self.model.eval()
#         y_true, y_pred = [], []

#         with torch.no_grad():
#             for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
#                 batch = self.dataset[idx]
#                 batch = self.move_to_device(batch, self.device)

#                 # 模型预测
#                 y_hat = self.model(batch)  # [total_nodes]

#                 # 取出标签和长度
#                 text_len_tensor = batch["text_len_tensor"]  # [B]，比如 [31]
#                 label_tensor = batch["label_tensor"]        # [B, pad_len]

#                 # 遍历 batch 内每个样本
#                 for i, L in enumerate(text_len_tensor.tolist()):
#                     gold = label_tensor[i, :L].cpu()
#                     pred = y_hat[i, :L].cpu()

#                     y_true.extend(gold.numpy().tolist())
#                     y_pred.extend(pred.numpy().tolist())

#         # 计算指标
#         f1 = f1_score(y_true, y_pred, average="weighted")
#         print(classification_report(y_true, y_pred, digits=4))
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

                # 模型预测
                y_hat = self.model(batch)

                # 取出标签和长度
                text_len_tensor = batch["text_len_tensor"]  # [B]
                label_tensor = batch["label_tensor"]

                if label_tensor.dim() == 2:
                    # [B, pad_len] 情况
                    for i, L in enumerate(text_len_tensor.tolist()):
                        gold = label_tensor[i, :L].cpu()
                        pred = y_hat[i, :L].cpu()
                        y_true.extend(gold.numpy().tolist())
                        y_pred.extend(pred.numpy().tolist())
                else:
                    # [total_len] 情况（已经 flatten）
                    start = 0
                    for L in text_len_tensor.tolist():
                        end = start + L
                        gold = label_tensor[start:end].cpu()
                        pred = y_hat[start:end].cpu()
                        y_true.extend(gold.numpy().tolist())
                        y_pred.extend(pred.numpy().tolist())
                        start = end

        # 计算指标
        f1 = f1_score(y_true, y_pred, average="weighted")
        print(classification_report(y_true, y_pred, digits=4))
        return f1
