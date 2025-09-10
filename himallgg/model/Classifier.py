import torch
import torch.nn as nn
import torch.nn.functional as F
import himallgg

log = himallgg.utils.get_logger()


class MaskedEmotionAtt(nn.Module):
    """
    对每个样本节点序列做自注意力加权，返回加权后的特征。
    输入 h: [total_nodes, feature_dim]
    text_len_tensor: [B] 每个样本实际长度
    """
    def __init__(self, input_dim):
        super(MaskedEmotionAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor=None):
        # 这里不用 text_len_tensor，直接返回全量特征
        return self.lin(h)


class Classifier(nn.Module):
    """
    单节点分类器
    """
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)  # tag_size=4/6

        if getattr(args, "class_weight", False):
            # 你可以根据数据分布修改权重
            self.loss_weights = torch.tensor([1.0]*tag_size).to(args.device)
            self.nll_loss = nn.NLLLoss(weight=self.loss_weights)
        else:
            self.nll_loss = nn.NLLLoss()

    def get_prob(self, h, text_len_tensor=None):
        """
        h: [total_nodes, feat_dim]
        返回 log_softmax([total_nodes, tag_size])
        """
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=-1)  # [total_nodes, tag_size]
        return log_prob

    def forward(self, h, text_len_tensor=None):
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor=None):
        log_prob = self.get_prob(h, text_len_tensor)
        loss = self.nll_loss(log_prob, label_tensor)
        return loss


# class Classifier(nn.Module):
#     """
#     单节点分类器，四分类版本
#     """
#     def __init__(self, input_dim, hidden_size, tag_size, args):
#         super(Classifier, self).__init__()
#         self.emotion_att = MaskedEmotionAtt(input_dim)
#         self.lin1 = nn.Linear(input_dim, hidden_size)
#         self.drop = nn.Dropout(args.drop_rate)
#         self.lin2 = nn.Linear(hidden_size, tag_size)  # tag_size=4

#         if getattr(args, "class_weight", False):
#             # 四分类对应的权重，可根据四分类数据的分布调整
#             self.loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=args.device)
#             self.nll_loss = nn.NLLLoss(weight=self.loss_weights)
#         else:
#             self.nll_loss = nn.NLLLoss()

#     def get_prob(self, h, text_len_tensor=None):
#         """
#         直接对 h（[total_nodes, feat_dim]）输出 log_prob
#         """
#         h = self.emotion_att(h, text_len_tensor)
#         hidden = self.drop(F.relu(self.lin1(h)))
#         scores = self.lin2(hidden)
#         log_prob = F.log_softmax(scores, dim=-1)  # [total_nodes, 4]
#         return log_prob

#     def forward(self, h, text_len_tensor=None):
#         log_prob = self.get_prob(h, text_len_tensor)
#         y_hat = torch.argmax(log_prob, dim=-1)
#         return y_hat

#     def get_loss(self, h, label_tensor, text_len_tensor=None):
#         """
#         h: [total_nodes, feat_dim], label_tensor: [total_nodes]
#         """
#         log_prob = self.get_prob(h, text_len_tensor)
#         loss = self.nll_loss(log_prob, label_tensor)
#         return loss
