import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from mamba_ssm import Mamba


class ContextualEncoder(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(ContextualEncoder, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "mamba":
            # 直接用 Mamba 作为序列编码器，输出维度 = g_dim
            self.in_proj = nn.Linear(self.input_size, self.hidden_dim)  # 新增：输入投影到 g_dim
            self.mamba = Mamba(
                d_model=self.hidden_dim,  # 与原来双向RNN输出维度一致
                d_state=64,
                d_conv=4,
                expand=2
            )

    def forward(self, text_len_tensor, text_tensor):
        if hasattr(self, "mamba"):
            # Mamba 分支：先把输入从 u_dim 映射到 g_dim，再送入 Mamba
            x = self.in_proj(text_tensor)
            return self.mamba(x)
        # 直接处理 padding 后的 [B, L, D]
        # 上游会用 lengths 做 mask，无需 pack
        packed = pack_padded_sequence(
            text_tensor,
            text_len_tensor.to('cpu'),
            batch_first=True,
            enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)  # 兼容 GRU/LSTM：忽略第二个返回值
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out
