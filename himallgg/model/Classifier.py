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

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [total_nodes, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s: s + cur_len]
            z = h[s: s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H]
            ret[s: s + cur_len, :] = out
            s += cur_len
        return ret


class Classifier(nn.Module):
    """
    单节点分类器
    支持 MaskedEmotionAtt、dropout、两层全连接和 NLLLoss
    方案A：仅对有效节点计算
    """
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.emotion_att = MaskedEmotionAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)

        if getattr(args, "class_weight", False):
            self.loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                              1 / 0.160585, 1 / 0.127711, 1 / 0.252668]).to(args.device)
            self.nll_loss = nn.NLLLoss(weight=self.loss_weights)
        else:
            self.nll_loss = nn.NLLLoss()

    def get_prob(self, h, text_len_tensor):
        """
        返回 log_softmax 概率，仅保留有效节点
        h: [total_nodes, feature_dim]
        text_len_tensor: [B]
        """
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        log_prob_full = F.log_softmax(scores, dim=-1)  # [B*pad_len, num_classes]

        # 按实际长度展开有效节点
        s = 0
        log_prob_nodes = []
        for L in text_len_tensor:
            L = L.item()
            log_prob_nodes.append(log_prob_full[s: s + L])
            s += L
        log_prob_nodes = torch.cat(log_prob_nodes, dim=0)  # [total_valid_nodes, num_classes]
        return log_prob_nodes

    def forward(self, h, text_len_tensor):
        """
        返回预测标签 y_hat，仅有效节点
        """
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        """
        方案A: 仅取有效长度节点计算 loss
        h: [total_nodes, feature_dim]
        label_tensor: [B, max_len], 已padding
        text_len_tensor: [B]
        """
        log_prob = self.get_prob(h, text_len_tensor)  # [total_valid_nodes, num_classes]

        # 展开有效节点的 label
        label_list = [label_tensor[i, :L] for i, L in enumerate(text_len_tensor)]
        label_nodes = torch.cat(label_list, dim=0)  # [total_valid_nodes]

        loss = self.nll_loss(log_prob, label_nodes)
        return loss
