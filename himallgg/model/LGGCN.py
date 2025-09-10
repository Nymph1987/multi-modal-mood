# himallgg/LGGCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import himallgg
from .ContextualEncoder import ContextualEncoder
from .EdgeAtt import EdgeAtt
from .GCN import GCN, SGCN
from .Classifier import Classifier
from .functions import batch_graphify, batch_graphify1
from .Fusion import *
from .DMC import DominantModalityCorrection, ModalityReconstructor, IntegrityConstraintLoss
import copy
from .functions import graphcl_augment  # 你已有的 augmentation 函数
import math

log = himallgg.utils.get_logger()

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class CrossModalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, mask=None):
        q = self.q_layer(x)
        k = self.k_layer(y)
        v = self.v_layer(y)
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = self.dropout(attn_output)
        return attn_output + x  # residual

class LGGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        # feature dims (与你原始代码保持一致)
        u_dim, uA_dim, uV_dim = 1024, 1582, 342
        g_dim, h1_dim, h2_dim, hc_dim, tag_size = 160, 100, 100, 100, 4  # 四分类
        self.wp, self.wf, self.device = args.wp, args.wf, args.device

        # encoders
        self.rnn = ContextualEncoder(u_dim, g_dim, args)
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)

        # DMC / recon / integrity
        self.dmc = DominantModalityCorrection(d_model=g_dim, num_heads=8, dropout=args.drop_rate)
        self.recon_A = ModalityReconstructor(g_dim*3, g_dim)
        self.recon_V = ModalityReconstructor(g_dim*3, g_dim)
        self.integrity_loss = IntegrityConstraintLoss(
            alpha=getattr(args, 'integrity_alpha', 1.0),
            beta=getattr(args, 'integrity_beta', 1.0)
        )
        self.integrity_weight = getattr(args, 'integrity_weight', 1.0)

        # cross attention / edges / gcn
        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)
        self.edge_att_all = EdgeAtt(g_dim*3, args)
        self.gcn = GCN(g_dim*3, h1_dim, h2_dim, args)
        self.gcn1 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim, h1_dim, g_dim, args)

        self.att = MultiHeadedAttention(10, g_dim)
        # Classifiers
        self.clf = Classifier(h2_dim + g_dim*3, hc_dim, tag_size, args)
        self.clf_T = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_A = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_V = Classifier(g_dim, hc_dim, tag_size, args)

        # edge type map
        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[f"{j}{k}0"] = len(edge_type_to_idx)
                edge_type_to_idx[f"{j}{k}1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        self.edge_type_to_idx1 = {'00': 0, '01': 1, '10': 2, '11': 3}
        log.debug(self.edge_type_to_idx)

        # GraphCL hyperparams
        self.use_graphcl = getattr(args, "use_graphcl", False)
        self.graphcl_weight = getattr(args, "graphcl_weight", 0.0)  # 默认 0 (baseline friendly)
        self.graphcl_tau = getattr(args, "graphcl_tau", 0.1)       # 推荐较小温度
        self.graphcl_proj_dim = getattr(args, "graphcl_proj_dim", 64)
        self.graphcl_aug_type = getattr(args, "graphcl_aug_type", "mask_features")
        self.graphcl_aug_ratio = getattr(args, "graphcl_aug_ratio", 0.2)

        proj_in = g_dim
        self.proj_T = nn.Sequential(nn.Linear(proj_in, proj_in), nn.ReLU(), nn.Linear(proj_in, self.graphcl_proj_dim))
        self.proj_A = nn.Sequential(nn.Linear(proj_in, proj_in), nn.ReLU(), nn.Linear(proj_in, self.graphcl_proj_dim))
        self.proj_V = nn.Sequential(nn.Linear(proj_in, proj_in), nn.ReLU(), nn.Linear(proj_in, self.graphcl_proj_dim))

    def _sequence_mask(self, lengths, max_len=None):
        B = lengths.size(0)
        max_len = max_len or int(lengths.max())
        return torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)

    def get_rep(self, data):
        """
        返回：graph_out, features, g_out_T, g_out_A, g_out_V, extras
        extras 会包含用于 GraphCL 的增强视图 if self.use_graphcl
        """
        T_m = self.rnn(data["text_len_tensor"], data["text_tensor"])
        A_m = self.rnn_A(data["text_len_tensor"], data["audio_tensor"])
        V_m = self.rnn_V(data["text_len_tensor"], data["vision_tensor"])

        T_f = self.rnn(data["text_len_tensor"], data["text_tensor_full"])
        A_f = self.rnn_A(data["text_len_tensor"], data["audio_tensor_full"])
        V_f = self.rnn_V(data["text_len_tensor"], data["vision_tensor_full"])

        f_T, e_idx_T, e_norm_T, e_type_T, _ = batch_graphify1(T_m, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf, self.edge_type_to_idx1, self.edge_att, self.device)
        f_A, e_idx_A, e_norm_A, e_type_A, _ = batch_graphify1(A_m, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf, self.edge_type_to_idx1, self.edge_att, self.device)
        f_V, e_idx_V, e_norm_V, e_type_V, _ = batch_graphify1(V_m, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf, self.edge_type_to_idx1, self.edge_att, self.device)

        g_out_T = self.gcn1(f_T, e_idx_T, e_norm_T, e_type_T)
        g_out_A = self.gcn2(f_A, e_idx_A, e_norm_A, e_type_A)
        g_out_V = self.gcn3(f_V, e_idx_V, e_norm_V, e_type_V)

        pad_mask = self._sequence_mask(data["text_len_tensor"], max_len=T_m.size(1))
        corrected_text = self.dmc(T_m, A_m, V_m, text_mask=pad_mask)

        node_features = torch.cat([corrected_text, A_m, V_m], dim=-1)
        features, edge_index, edge_norm, edge_type, _ = batch_graphify(node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf, self.edge_type_to_idx, self.edge_att_all, self.device)

        fea1 = self.att(g_out_A, g_out_T, g_out_A)
        fea2 = self.att(g_out_V, g_out_T, g_out_V)
        features_graph = torch.cat([g_out_T, fea1.squeeze(1), fea2.squeeze(1)], dim=-1)
        graph_out = self.gcn(features_graph, edge_index, edge_norm, edge_type)

        extras = {"T_m": T_m, "A_m": A_m, "V_m": V_m, "T_f": T_f, "A_f": A_f, "V_f": V_f, "corrected_text": corrected_text}

        # --- GraphCL augment: 生成两路增强视图（若启用） ---
        # if self.use_graphcl:
        #     try:
        #         # graphcl_augment 应该接受 [N, d] 或 [B, L, d]（你实现的版本）
        #         # 我们传入 g_out_* （通常是 2D: [total_nodes, d]）
        #         gcl_T1 = graphcl_augment(g_out_T, aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio)
        #         gcl_T2 = graphcl_augment(g_out_T, aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio)
        #         gcl_A1 = graphcl_augment(g_out_A, aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio)
        #         gcl_A2 = graphcl_augment(g_out_A, aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio)
        #         gcl_V1 = graphcl_augment(g_out_V, aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio)
        #         gcl_V2 = graphcl_augment(g_out_V, aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio)

        #         extras.update({
        #             "gcl_T1": gcl_T1, "gcl_T2": gcl_T2,
        #             "gcl_A1": gcl_A1, "gcl_A2": gcl_A2,
        #             "gcl_V1": gcl_V1, "gcl_V2": gcl_V2,
        #         })
        #     except Exception as e:
        #         log.warning(f"GraphCL augment failed: {e}. Falling back to no-augment.")
        if self.use_graphcl:
            try:
                # T
                gcl_T1, e_idx_T1, e_norm_T1, e_type_T1 = graphcl_augment(
                    g_out_T, edge_index=e_idx_T, edge_norm=e_norm_T, edge_type=e_type_T,
                    aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio
                )
                gcl_T2, e_idx_T2, e_norm_T2, e_type_T2 = graphcl_augment(
                    g_out_T, edge_index=e_idx_T, edge_norm=e_norm_T, edge_type=e_type_T,
                    aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio
                )

                # A
                gcl_A1, e_idx_A1, e_norm_A1, e_type_A1 = graphcl_augment(
                    g_out_A, edge_index=e_idx_A, edge_norm=e_norm_A, edge_type=e_type_A,
                    aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio
                )
                gcl_A2, e_idx_A2, e_norm_A2, e_type_A2 = graphcl_augment(
                    g_out_A, edge_index=e_idx_A, edge_norm=e_norm_A, edge_type=e_type_A,
                    aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio
                )

                # V
                gcl_V1, e_idx_V1, e_norm_V1, e_type_V1 = graphcl_augment(
                    g_out_V, edge_index=e_idx_V, edge_norm=e_norm_V, edge_type=e_type_V,
                    aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio
                )
                gcl_V2, e_idx_V2, e_norm_V2, e_type_V2 = graphcl_augment(
                    g_out_V, edge_index=e_idx_V, edge_norm=e_norm_V, edge_type=e_type_V,
                    aug_type=self.graphcl_aug_type, ratio=self.graphcl_aug_ratio
                )

                extras.update({
                    "gcl_T1": gcl_T1, "gcl_T2": gcl_T2,
                    "gcl_A1": gcl_A1, "gcl_A2": gcl_A2,
                    "gcl_V1": gcl_V1, "gcl_V2": gcl_V2,
                    # 可选：存增强边
                    "e_idx_T1": e_idx_T1, "e_idx_T2": e_idx_T2,
                    "e_idx_A1": e_idx_A1, "e_idx_A2": e_idx_A2,
                    "e_idx_V1": e_idx_V1, "e_idx_V2": e_idx_V2,
                })
            except Exception as e:
                log.warning(f"GraphCL augment failed: {e}. Falling back to no-augment.")
        # 不抛，fallback 继续
        return graph_out, features, g_out_T, g_out_A, g_out_V, extras

    # def forward(self, data):
    #     graph_out, features, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)

    #     lengths_text = data["text_len_tensor"]       # [B]
    #     lengths_label = data["label_len_tensor"]     # [B]
    #     B = lengths_text.size(0)
    #     L_max = int(lengths_text.max().item())

    #     # helper: 将可能为 2D(总节点) 的张量重建为 3D [B,L_max,feat]
    #     def ensure_3d(x):
    #         if x.dim() == 3:
    #             return x
    #         feat_dim = x.size(1)
    #         total_nodes = x.size(0)
    #         if total_nodes == B * L_max:
    #             return x.view(B, L_max, feat_dim)
    #         x3 = torch.zeros(B, L_max, feat_dim, device=x.device, dtype=x.dtype)
    #         start = 0
    #         for i in range(B):
    #             li = int(lengths_text[i].item())
    #             if li > 0:
    #                 x3[i, :li] = x[start:start + li]
    #             start += li
    #         return x3

    #     features = ensure_3d(features)
    #     graph_out = ensure_3d(graph_out)
    #     g_out_T = ensure_3d(g_out_T)
    #     g_out_A = ensure_3d(g_out_A)
    #     g_out_V = ensure_3d(g_out_V)

    #     ar = torch.arange(L_max, device=features.device).unsqueeze(0)
    #     mask_label = ar < lengths_label.unsqueeze(1)

    #     feat_cat = torch.cat([features, graph_out], dim=-1)
    #     feat_cat_flat = feat_cat[mask_label]
    #     T_flat = g_out_T[mask_label]
    #     A_flat = g_out_A[mask_label]
    #     V_flat = g_out_V[mask_label]

    #     score = self.clf.get_prob(feat_cat_flat, lengths_label)
    #     score_T = self.clf_T.get_prob(T_flat, lengths_label)
    #     score_A = self.clf_A.get_prob(A_flat, lengths_label)
    #     score_V = self.clf_V.get_prob(V_flat, lengths_label)

    #     scores = score + 0.7*score_T + 0.2*score_V + 0.1*score_A
    #     log_prob = F.log_softmax(scores, dim=-1)
    #     y_hat = torch.argmax(log_prob, dim=-1)
    #     return y_hat

    # def forward(self, data):
    #     graph_out, features, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)

    #     lengths_text = data["text_len_tensor"]       # [B]
    #     lengths_label = data["label_len_tensor"]     # [B]
    #     B = lengths_text.size(0)
    #     L_max = int(lengths_text.max().item())

    #     def ensure_3d(x):
    #         if x.dim() == 3:
    #             return x
    #         feat_dim = x.size(1)
    #         total_nodes = x.size(0)
    #         if total_nodes == B * L_max:
    #             return x.view(B, L_max, feat_dim)
    #         x3 = torch.zeros(B, L_max, feat_dim, device=x.device, dtype=x.dtype)
    #         start = 0
    #         for i in range(B):
    #             li = int(lengths_text[i].item())
    #             if li > 0:
    #                 x3[i, :li] = x[start:start + li]
    #             start += li
    #         return x3

    #     features = ensure_3d(features)
    #     graph_out = ensure_3d(graph_out)
    #     g_out_T = ensure_3d(g_out_T)
    #     g_out_A = ensure_3d(g_out_A)
    #     g_out_V = ensure_3d(g_out_V)

    #     ar = torch.arange(L_max, device=features.device).unsqueeze(0)
    #     mask_label = ar < lengths_label.unsqueeze(1)

    #     feat_cat = torch.cat([features, graph_out], dim=-1)
    #     feat_cat_flat = feat_cat[mask_label]
    #     T_flat = g_out_T[mask_label]
    #     A_flat = g_out_A[mask_label]
    #     V_flat = g_out_V[mask_label]

    #     score = self.clf.get_prob(feat_cat_flat, lengths_label)
    #     score_T = self.clf_T.get_prob(T_flat, lengths_label)
    #     score_A = self.clf_A.get_prob(A_flat, lengths_label)
    #     score_V = self.clf_V.get_prob(V_flat, lengths_label)

    #     scores = score + 0.7*score_T + 0.2*score_V + 0.1*score_A
    #     log_prob = F.log_softmax(scores, dim=-1)
    #     y_hat = torch.argmax(log_prob, dim=-1)

    #     # ---- 新增打印 ----
    #     print("=== LGGCN Forward Debug ===")
    #     print("Combined raw scores shape:", scores.shape)
    #     print("Combined raw scores:", scores)
    #     print("Log probabilities shape:", log_prob.shape)
    #     print("Log probabilities:", log_prob)
    #     print("Predicted labels shape:", y_hat.shape)
    #     print("Predicted labels:", y_hat)
    #     print("===========================")

    #     return y_hat

    # def forward(self, data, debug_N=5):
    #     graph_out, features, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)

    #     lengths_text = data["text_len_tensor"]       # [B]
    #     lengths_label = data["label_len_tensor"]     # [B]
    #     B = lengths_text.size(0)
    #     L_max = int(lengths_text.max().item())

    #     def ensure_3d(x):
    #         if x.dim() == 3:
    #             return x
    #         feat_dim = x.size(1)
    #         total_nodes = x.size(0)
    #         if total_nodes == B * L_max:
    #             return x.view(B, L_max, feat_dim)
    #         x3 = torch.zeros(B, L_max, feat_dim, device=x.device, dtype=x.dtype)
    #         start = 0
    #         for i in range(B):
    #             li = int(lengths_text[i].item())
    #             if li > 0:
    #                 x3[i, :li] = x[start:start + li]
    #             start += li
    #         return x3

    #     features = ensure_3d(features)
    #     graph_out = ensure_3d(graph_out)
    #     g_out_T = ensure_3d(g_out_T)
    #     g_out_A = ensure_3d(g_out_A)
    #     g_out_V = ensure_3d(g_out_V)

    #     ar = torch.arange(L_max, device=features.device).unsqueeze(0)
    #     mask_label = ar < lengths_label.unsqueeze(1)

    #     feat_cat = torch.cat([features, graph_out], dim=-1)
    #     feat_cat_flat = feat_cat[mask_label]
    #     T_flat = g_out_T[mask_label]
    #     A_flat = g_out_A[mask_label]
    #     V_flat = g_out_V[mask_label]

    #     score = self.clf.get_prob(feat_cat_flat, lengths_label)
    #     score_T = self.clf_T.get_prob(T_flat, lengths_label)
    #     score_A = self.clf_A.get_prob(A_flat, lengths_label)
    #     score_V = self.clf_V.get_prob(V_flat, lengths_label)

    #     scores = score + 0.7*score_T + 0.2*score_V + 0.1*score_A
    #     log_prob = F.log_softmax(scores, dim=-1)
    #     y_hat = torch.argmax(log_prob, dim=-1)

    #     # ---- 新增打印：只显示前 N 个样本 ----
    #     N = min(debug_N, scores.size(0))
    #     print("=== LGGCN Forward Debug (first {} samples) ===".format(N))
    #     print("Combined raw scores shape:", scores.shape)
    #     print("Combined raw scores (first {}):\n".format(N), scores[:N])
    #     print("Log probabilities shape:", log_prob.shape)
    #     print("Log probabilities (first {}):\n".format(N), log_prob[:N])
    #     print("Predicted labels shape:", y_hat.shape)
    #     print("Predicted labels (first {}):".format(N), y_hat[:N])
    #     print("================================================")

    #     return y_hat
    def forward(self, data, debug_N=5):
        """
        LGGCN forward with optional debug prints.
        """
        graph_out, features, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)

        lengths_text = data["text_len_tensor"]       # [B]
        lengths_label = data["label_len_tensor"]     # [B]
        B = lengths_text.size(0)
        L_max = int(lengths_text.max().item())

        # Helper: 确保 tensor 是 [B, L_max, feat]
        def ensure_3d(x):
            if x.dim() == 3:
                return x
            feat_dim = x.size(1)
            total_nodes = x.size(0)
            if total_nodes == B * L_max:
                return x.view(B, L_max, feat_dim)
            x3 = torch.zeros(B, L_max, feat_dim, device=x.device, dtype=x.dtype)
            start = 0
            for i in range(B):
                li = int(lengths_text[i].item())
                if li > 0:
                    x3[i, :li] = x[start:start + li]
                start += li
            return x3

        features = ensure_3d(features)
        graph_out = ensure_3d(graph_out)
        g_out_T = ensure_3d(g_out_T)
        g_out_A = ensure_3d(g_out_A)
        g_out_V = ensure_3d(g_out_V)

        ar = torch.arange(L_max, device=features.device).unsqueeze(0)
        mask_label = ar < lengths_label.unsqueeze(1)

        # flatten mask
        feat_cat = torch.cat([features, graph_out], dim=-1)
        feat_cat_flat = feat_cat[mask_label]
        T_flat = g_out_T[mask_label]
        A_flat = g_out_A[mask_label]
        V_flat = g_out_V[mask_label]

        # 分类器融合
        score = self.clf.get_prob(feat_cat_flat, lengths_label)
        score_T = self.clf_T.get_prob(T_flat, lengths_label)
        score_A = self.clf_A.get_prob(A_flat, lengths_label)
        score_V = self.clf_V.get_prob(V_flat, lengths_label)

        scores = score + 0.7*score_T + 0.2*score_V + 0.1*score_A
        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)

        # --- Debug 打印 ---
        # if debug_N > 0:
        #     print("=== LGGCN Forward Debug ===")
        #     print("Combined raw scores shape:", scores.shape)
        #     print("Combined raw scores (first {} samples):".format(debug_N), scores[:debug_N])
        #     print("Log probabilities shape:", log_prob.shape)
        #     print("Log probabilities (first {} samples):".format(debug_N), log_prob[:debug_N])
        #     print("Predicted labels shape:", y_hat.shape)
        #     print("Predicted labels (first {} samples):".format(debug_N), y_hat[:debug_N])
        #     print("===========================")

        return y_hat


    def get_loss(self, data):
        """
        返回 total_loss（仍然保持与训练流程兼容）。
        同时在内部打印/记录分项 loss 以及 GraphCL 的 pos/neg cos 统计，便于监控。
        """
        graph_out, features, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)

        # 计算 B, last index
        last_idx = (data["text_len_tensor"] - 1).clamp(min=0)
        B = last_idx.size(0)
        batch_idx = torch.arange(B, device=features.device)

        # ---------- 保证 features/graph_out 为 3D ----------
        if features.dim() == 2:
            L_max = int(data["text_len_tensor"].max().item())
            total_nodes = features.size(0)
            feat_dim_features = features.size(1)
            expected_total_nodes = B * L_max
            if total_nodes != expected_total_nodes:
                features_3d = torch.zeros(B, L_max, feat_dim_features, device=features.device, dtype=features.dtype)
                start_idx = 0
                for i, length in enumerate(data["text_len_tensor"]):
                    length_i = int(length.item())
                    end_idx = start_idx + length_i
                    features_3d[i, :length_i] = features[start_idx:end_idx]
                    start_idx = end_idx
                features = features_3d
            else:
                features = features.view(B, L_max, feat_dim_features)

        if graph_out.dim() == 2:
            L_max = int(data["text_len_tensor"].max().item())
            total_nodes = graph_out.size(0)
            feat_dim_graph = graph_out.size(1)
            expected_total_nodes = B * L_max
            if total_nodes != expected_total_nodes:
                graph_out_3d = torch.zeros(B, L_max, feat_dim_graph, device=graph_out.device, dtype=graph_out.dtype)
                start_idx = 0
                for i, length in enumerate(data["text_len_tensor"]):
                    length_i = int(length.item())
                    end_idx = start_idx + length_i
                    graph_out_3d[i, :length_i] = graph_out[start_idx:end_idx]
                    start_idx = end_idx
                graph_out = graph_out_3d
            else:
                graph_out = graph_out.view(B, L_max, feat_dim_graph)

        # pooling 最后有效 token
        concat_feats_pooled = torch.cat([
            features[batch_idx, last_idx],
            graph_out[batch_idx, last_idx]
        ], dim=-1)

        # g_out_T/A/V 重建为 3D，然后取最后 token 特征
        L_max = int(data["text_len_tensor"].max().item())

        def rebuild_2d_to_3d(x, B, L_max):
            if x.dim() == 3:
                return x
            feat_dim = x.size(1)
            total_nodes = x.size(0)
            expected = B * L_max
            if total_nodes != expected:
                x3 = torch.zeros(B, L_max, feat_dim, device=x.device, dtype=x.dtype)
                start = 0
                for i, length in enumerate(data["text_len_tensor"]):
                    length_i = int(length.item())
                    end = start + length_i
                    if length_i > 0:
                        x3[i, :length_i] = x[start:end]
                    start = end
                return x3
            else:
                return x.view(B, L_max, feat_dim)

        g_out_T = rebuild_2d_to_3d(g_out_T, B, L_max)
        g_out_A = rebuild_2d_to_3d(g_out_A, B, L_max)
        g_out_V = rebuild_2d_to_3d(g_out_V, B, L_max)

        g_out_T_flat_last = g_out_T[batch_idx, last_idx]
        g_out_A_flat_last = g_out_A[batch_idx, last_idx]
        g_out_V_flat_last = g_out_V[batch_idx, last_idx]

        # labels_flat 对齐（基于 label_len_tensor）
        labels_flat_list = []
        start_idx = 0
        label_len = data["label_len_tensor"]
        for i in range(B):
            valid_len = int(data["text_len_tensor"][i].item())
            lab_len = int(label_len[i].item())
            if lab_len <= 0 or valid_len <= 0:
                labels_flat_list.append(torch.tensor(0, dtype=torch.long, device=data["label_tensor"].device))
            else:
                local_last = min(valid_len, lab_len) - 1
                labels_flat_list.append(data["label_tensor"][start_idx + local_last])
            start_idx += lab_len
        labels_flat = torch.stack(labels_flat_list).to(torch.long)

        # 计算分类 loss
        loss_main = self.clf.get_loss(concat_feats_pooled, labels_flat)
        loss_T = self.clf_T.get_loss(g_out_T_flat_last, labels_flat)
        loss_A = self.clf_A.get_loss(g_out_A_flat_last, labels_flat)
        loss_V = self.clf_V.get_loss(g_out_V_flat_last, labels_flat)

        # ---------- GraphCL 对比损失 ----------
        loss_gcl = torch.tensor(0.0, device=features.device)
        cos_pos_mean = 0.0
        cos_neg_mean = 0.0
        if self.use_graphcl and self.graphcl_weight > 0:
            # mask_label: [B,L_max] 选择与 label 对齐的节点（如 forward 所用）
            ar = torch.arange(L_max, device=features.device).unsqueeze(0)
            mask_label = ar < data["label_len_tensor"].unsqueeze(1)  # [B,L_max]

            # extras 中可能含有 gcl_T1/gcl_T2 等增强视图（2D 或 3D）
            def ensure_3d_from(src, fallback):
                x = src if src is not None else fallback
                if x.dim() == 3:
                    return x
                # 2D -> 3D
                B_local = data["text_len_tensor"].size(0)
                feat_dim = x.size(1)
                x3 = torch.zeros(B_local, L_max, feat_dim, device=x.device, dtype=x.dtype)
                start = 0
                for i, length in enumerate(data["text_len_tensor"]):
                    li = int(length.item())
                    if li > 0:
                        x3[i, :li] = x[start:start+li]
                        start += li
                return x3

            # 取增强视图（如果没有增强视图将退化为原视图）
            zT_src = ensure_3d_from(extras.get("gcl_T1", None), g_out_T)
            zT_src2 = ensure_3d_from(extras.get("gcl_T2", None), g_out_T)
            zA_src = ensure_3d_from(extras.get("gcl_A1", None), g_out_A)
            zA_src2 = ensure_3d_from(extras.get("gcl_A2", None), g_out_A)
            zV_src = ensure_3d_from(extras.get("gcl_V1", None), g_out_V)
            zV_src2 = ensure_3d_from(extras.get("gcl_V2", None), g_out_V)

            # 取有效节点并投影 -> 得到 [N, d]
            T_nodes = zT_src[mask_label]    # view1
            T_nodes2 = zT_src2[mask_label]  # view2
            A_nodes = zA_src[mask_label]
            A_nodes2 = zA_src2[mask_label]
            V_nodes = zV_src[mask_label]
            V_nodes2 = zV_src2[mask_label]

            # 有效配对掩码（bool 长度 N）
            mask_audio = data.get('mask_audio', torch.ones_like(mask_label, dtype=torch.long)).bool()
            mask_visual = data.get('mask_visual', torch.ones_like(mask_label, dtype=torch.long)).bool()
            valid_TA = mask_audio[mask_label]            # [N]
            valid_TV = mask_visual[mask_label]           # [N]
            valid_AV = (mask_audio & mask_visual)[mask_label]  # [N]

            # 投影
            pT1 = self.proj_T(T_nodes)
            pT2 = self.proj_T(T_nodes2)
            pA1 = self.proj_A(A_nodes)
            pA2 = self.proj_A(A_nodes2)
            pV1 = self.proj_V(V_nodes)
            pV2 = self.proj_V(V_nodes2)

            # 稳健 NT-Xent：保证索引对齐（使用 valid_mask）
            def pairwise_nt_xent(z1, z2, valid_mask):
                idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
                if idx.numel() < 2:
                    return torch.tensor(0.0, device=z1.device)
                z1v = z1[idx]
                z2v = z2[idx]
                z1n = F.normalize(z1v, dim=-1)
                z2n = F.normalize(z2v, dim=-1)
                logits12 = torch.matmul(z1n, z2n.t()) / self.graphcl_tau
                logits21 = torch.matmul(z2n, z1n.t()) / self.graphcl_tau
                labels = torch.arange(z1n.size(0), device=z1.device)
                loss12 = F.cross_entropy(logits12, labels)
                loss21 = F.cross_entropy(logits21, labels)
                return 0.5 * (loss12 + loss21)

            loss_TA = pairwise_nt_xent(pT1, pA1, valid_TA)
            loss_TV = pairwise_nt_xent(pT1, pV1, valid_TV)
            loss_AV = pairwise_nt_xent(pA1, pV1, valid_AV)
            loss_gcl = (loss_TA + loss_TV + loss_AV) / 3.0
            loss_gcl = loss_gcl.to(features.device)
            loss_gcl = loss_gcl * 1.0  # 保持 float tensor

            # 计算 pos/neg cos 供监控（在有数据的情况下）
            try:
                # positive cos between T and A where valid_TA
                idx_TA = valid_TA.nonzero(as_tuple=False).squeeze(1)
                if idx_TA.numel() >= 2:
                    tvec = F.normalize(pT1[idx_TA], dim=-1)
                    avec = F.normalize(pA1[idx_TA], dim=-1)
                    cos_pos = (tvec * avec).sum(-1)
                    cos_pos_mean = cos_pos.mean().item()
                    # negative: shuffle A
                    perm = torch.randperm(avec.size(0), device=avec.device)
                    cos_neg_mean = (tvec * avec[perm]).sum(-1).mean().item()
                else:
                    cos_pos_mean = 0.0
                    cos_neg_mean = 0.0
            except Exception:
                cos_pos_mean = 0.0
                cos_neg_mean = 0.0

            # assign to outer scope
            loss_gcl = loss_gcl
        else:
            loss_gcl = torch.tensor(0.0, device=features.device)

        # 总损失
        total_loss = loss_main + 0.7*loss_T + 0.2*loss_V + 0.1*loss_A + self.graphcl_weight * loss_gcl

        # 监控打印（训练时会很有用）
        try:
            log.info(f"[Loss] main={loss_main.item():.4f}, T={loss_T.item():.4f}, A={loss_A.item():.4f}, V={loss_V.item():.4f}, GCL={loss_gcl.item():.6f}, total={total_loss.item():.4f}, cos+={cos_pos_mean:.3f}, cos-={cos_neg_mean:.3f}")
        except Exception:
            # 在某些场景下 loss 非 scalar（不太可能），忽略打印失败
            pass

        return total_loss


    def get_graphcl_loss(self, data):
        """
        返回 GraphCL 对比损失。
        如果未启用 GraphCL 或 weight=0，则返回 0.
        """
        if not self.use_graphcl or self.graphcl_weight <= 0:
            return torch.tensor(0.0, device=data["text_tensor"].device)

        _, _, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)
        B = data["text_len_tensor"].size(0)
        L_max = int(data["text_len_tensor"].max().item())

        # mask 对齐
        ar = torch.arange(L_max, device=g_out_T.device).unsqueeze(0)
        mask_label = ar < data["label_len_tensor"].unsqueeze(1)  # [B,L_max]

        def ensure_3d_from(src, fallback):
            x = src if src is not None else fallback
            if x.dim() == 3:
                return x
            feat_dim = x.size(1)
            x3 = torch.zeros(B, L_max, feat_dim, device=x.device, dtype=x.dtype)
            start = 0
            for i, length in enumerate(data["text_len_tensor"]):
                li = int(length.item())
                if li > 0:
                    x3[i, :li] = x[start:start+li]
                    start += li
            return x3

        # 取增强视图或 fallback
        zT1 = ensure_3d_from(extras.get("gcl_T1", None), g_out_T)
        zT2 = ensure_3d_from(extras.get("gcl_T2", None), g_out_T)
        zA1 = ensure_3d_from(extras.get("gcl_A1", None), g_out_A)
        zA2 = ensure_3d_from(extras.get("gcl_A2", None), g_out_A)
        zV1 = ensure_3d_from(extras.get("gcl_V1", None), g_out_V)
        zV2 = ensure_3d_from(extras.get("gcl_V2", None), g_out_V)

        T_nodes = zT1[mask_label]
        T_nodes2 = zT2[mask_label]
        A_nodes = zA1[mask_label]
        A_nodes2 = zA2[mask_label]
        V_nodes = zV1[mask_label]
        V_nodes2 = zV2[mask_label]

        # 投影
        pT1 = self.proj_T(T_nodes)
        pT2 = self.proj_T(T_nodes2)
        pA1 = self.proj_A(A_nodes)
        pA2 = self.proj_A(A_nodes2)
        pV1 = self.proj_V(V_nodes)
        pV2 = self.proj_V(V_nodes2)

        # NT-Xent
        def pairwise_nt_xent(z1, z2):
            z1n = F.normalize(z1, dim=-1)
            z2n = F.normalize(z2, dim=-1)
            logits = torch.matmul(z1n, z2n.t()) / self.graphcl_tau
            labels = torch.arange(z1n.size(0), device=z1.device)
            return F.cross_entropy(logits, labels)

        loss_TA = pairwise_nt_xent(pT1, pA1)
        loss_TV = pairwise_nt_xent(pT1, pV1)
        loss_AV = pairwise_nt_xent(pA1, pV1)

        loss_gcl = (loss_TA + loss_TV + loss_AV) / 3.0
        return loss_gcl
