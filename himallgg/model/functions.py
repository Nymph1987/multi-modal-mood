import numpy as np
import torch

import himallgg

log = himallgg.utils.get_logger()

def batch_graphify1(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))
    # 小图ATT不要
    edge_weights = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = '0'
            else:
                c = '1'
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2)])
    
    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths

def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))
    # 小图ATT不要
    edge_weights = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            # print(f'speaker2 = speaker_tensor[j, item[1]].item()={speaker_tensor[j, item[1]].item()}')
            if item[0] < item[1]:
                c = '0'
            else:
                c = '1'
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])
    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def _dropout_edges(edge_index, edge_norm, edge_type, drop_ratio: float):
    if drop_ratio <= 0.0:
        return edge_index, edge_norm, edge_type
    E = edge_index.size(1)
    keep_mask = (torch.rand(E, device=edge_index.device) > drop_ratio)
    if keep_mask.sum() == 0:
        # 至少保留一条边，避免空图
        keep_mask[torch.randint(0, E, (1,), device=edge_index.device)] = True
    edge_index = edge_index[:, keep_mask]
    edge_norm = edge_norm[keep_mask]
    edge_type = edge_type[keep_mask]
    return edge_index, edge_norm, edge_type

def _mask_node_features(x: torch.Tensor, mask_ratio: float):
    if mask_ratio <= 0.0:
        return x
    x_masked = x.clone()
    mask = (torch.rand_like(x_masked) < mask_ratio)
    x_masked[mask] = 0.0
    return x_masked

# def graphcl_augment(node_features: torch.Tensor,
#                     edge_index: torch.Tensor,
#                     edge_norm: torch.Tensor,
#                     edge_type: torch.Tensor,
#                     aug_type: str = "mask_features",
#                     aug_ratio: float = 0.2):
#     """
#     轻量图增广：特征遮蔽或边丢弃，供 GraphCL 对比学习使用。
#     返回维持相同签名的（node_features, edge_index, edge_norm, edge_type）
#     """
#     if aug_type == "mask_features":
#         node_features = _mask_node_features(node_features, aug_ratio)
#     elif aug_type == "drop_edges":
#         edge_index, edge_norm, edge_type = _dropout_edges(edge_index, edge_norm, edge_type, aug_ratio)
#     else:
#         # identity
#         pass
#     return node_features, edge_index, edge_norm, edge_type
# def graphcl_augment(node_features: torch.Tensor,
#                     edge_index: torch.Tensor = None,
#                     edge_norm: torch.Tensor = None,
#                     edge_type: torch.Tensor = None,
#                     aug_type: str = "mask_features",
#                     ratio: float = 0.2):
#     """
#     轻量图增广，只返回 node_features（供 GraphCL 对比学习）。
#     如果传了边信息，也可以做边丢弃。
#     """
#     edge_index_aug, edge_norm_aug, edge_type_aug = edge_index, edge_norm, edge_type

#     if aug_type == "mask_features":
#         node_features = _mask_node_features(node_features, ratio)
#     elif aug_type == "drop_edges" and edge_index is not None:
#         edge_index_aug, edge_norm_aug, edge_type_aug = _dropout_edges(edge_index, edge_norm, edge_type, ratio)
#     else:
#         # identity
#         pass

#     # 只返回节点特征
#     return node_features


def graphcl_augment(node_features, edge_index=None, edge_norm=None, edge_type=None,
                     aug_type="mask_features", ratio=0.2):
    """
    支持节点和边增强
    """
    edge_index_aug, edge_norm_aug, edge_type_aug = edge_index, edge_norm, edge_type

    if aug_type == "mask_features":
        node_features = _mask_node_features(node_features, ratio)
    elif aug_type == "drop_edges":
        edge_index_aug, edge_norm_aug, edge_type_aug = _dropout_edges(edge_index, edge_norm, edge_type, ratio)

    return node_features, edge_index_aug, edge_norm_aug, edge_type_aug
