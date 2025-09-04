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
import copy  # 新增

log = himallgg.utils.get_logger()

def clones(module, N):
    "Produce N identical layers."
    # 修复：使用 deepcopy，避免通过 __dict__ 传入非法 kwargs
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
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
        u_dim, uA_dim, uV_dim = 1024, 1582, 342
        g_dim, h1_dim, h2_dim, hc_dim, tag_size = 160, 100, 100, 100, 6
        self.wp, self.wf, self.device = args.wp, args.wf, args.device

        self.rnn = ContextualEncoder(u_dim, g_dim, args)
        self.rnn_A = ContextualEncoder(uA_dim, g_dim, args)
        self.rnn_V = ContextualEncoder(uV_dim, g_dim, args)

        self.dmc = DominantModalityCorrection(d_model=g_dim, num_heads=8, dropout=args.drop_rate)
        self.recon_A = ModalityReconstructor(g_dim*3, g_dim)
        self.recon_V = ModalityReconstructor(g_dim*3, g_dim)
        self.integrity_loss = IntegrityConstraintLoss(
            alpha=getattr(args, 'integrity_alpha', 1.0),
            beta=getattr(args, 'integrity_beta', 1.0)
        )
        self.integrity_weight = getattr(args, 'integrity_weight', 1.0)

        self.cross_attention = CrossModalAttention(g_dim)
        self.edge_att = EdgeAtt(g_dim, args)
        self.edge_att_all = EdgeAtt(g_dim*3, args)
        self.gcn = GCN(g_dim*3, h1_dim, h2_dim, args)
        self.gcn1 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn2 = SGCN(g_dim, h1_dim, g_dim, args)
        self.gcn3 = SGCN(g_dim, h1_dim, g_dim, args)

        self.att = MultiHeadedAttention(10, g_dim)
        self.clf = Classifier(h2_dim + g_dim*3, hc_dim, tag_size, args)
        self.clf_T = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_A = Classifier(g_dim, hc_dim, tag_size, args)
        self.clf_V = Classifier(g_dim, hc_dim, tag_size, args)

        # 构建 edge_type 映射
        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[f"{j}{k}0"] = len(edge_type_to_idx)
                edge_type_to_idx[f"{j}{k}1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        self.edge_type_to_idx1 = {'00': 0, '01': 1, '10': 2, '11': 3}
        log.debug(self.edge_type_to_idx)

    def _sequence_mask(self, lengths, max_len=None):
        B = lengths.size(0)
        max_len = max_len or int(lengths.max())
        return torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)

    def get_rep(self, data):
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
        return graph_out, features, g_out_T, g_out_A, g_out_V, extras

    def forward(self, data):
        graph_out, features, g_out_T, g_out_A, g_out_V, _ = self.get_rep(data)
        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score = self.clf.get_prob(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        score_T = self.clf_T.get_prob(g_out_T, data["text_len_tensor"])
        score_A = self.clf_A.get_prob(g_out_A, data["text_len_tensor"])
        score_V = self.clf_V.get_prob(g_out_V, data["text_len_tensor"])
        scores = score + 0.7*score_T + 0.2*score_V + 0.1*score_A
        log_prob = F.log_softmax(scores, dim=-1)
        y_hat = torch.argmax(log_prob, dim=-1)
        return y_hat

    def get_loss(self, data):
        graph_out, features, g_out_T, g_out_A, g_out_V, extras = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1), data["label_tensor"], data["text_len_tensor"])
        loss_T = self.clf_T.get_loss(g_out_T, data["label_tensor"], data["text_len_tensor"])
        loss_A = self.clf_A.get_loss(g_out_A, data["label_tensor"], data["text_len_tensor"])
        loss_V = self.clf_V.get_loss(g_out_V, data["label_tensor"], data["text_len_tensor"])

        B, L, _ = extras['T_m'].shape
        concat_feats = torch.cat([extras['T_m'], extras['A_m'], extras['V_m']], dim=-1)
        rec_A = self.recon_A(concat_feats)
        rec_V = self.recon_V(concat_feats)

        mask_audio = data.get('mask_audio', torch.ones((B,L), device=concat_feats.device))
        mask_visual = data.get('mask_visual', torch.ones((B,L), device=concat_feats.device))
        missing_audio = 1 - mask_audio
        missing_visual = 1 - mask_visual

        integ_loss_A, _ = self.integrity_loss(rec_A, extras['A_f'], extras['corrected_text'], extras['T_f'], mask=missing_audio)
        integ_loss_V, _ = self.integrity_loss(rec_V, extras['V_f'], extras['corrected_text'], extras['T_f'], mask=missing_visual)

        total_loss = loss + 0.7*loss_T + 0.2*loss_V + 0.1*loss_A + self.integrity_weight*(integ_loss_A + integ_loss_V)
        return total_loss
