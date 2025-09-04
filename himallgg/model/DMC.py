import torch
import torch.nn as nn
import torch.nn.functional as F


class DominantModalityCorrection(nn.Module):
    """
    DMC (Dominant Modality Correction) module from LNLN.
    Ensures high-quality representation of the dominant modality (text)
    by correcting it with auxiliary modalities (audio, visual).
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(DominantModalityCorrection, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # gate on concat(T, A, V)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

    def forward(self, text_features, audio_features, visual_features, text_mask=None):
        """
        Args:
            text_features: [B, L, D]
            audio_features: [B, L, D]
            visual_features: [B, L, D]
            text_mask: [B, L] (True=valid, False=pad) or None

        Returns:
            corrected_text: [B, L, D]
        """
        B, L, D = text_features.shape

        # Stack A/V to serve as cross-modal memory
        aux = torch.stack([audio_features, visual_features], dim=1)  # [B,2,L,D]
        aux = aux.view(B, 2 * L, D)  # [B, 2L, D]

        # Cross-attention: T queries, A/V as key/value
        corrected, _ = self.multihead_attn(
            query=text_features, key=aux, value=aux, key_padding_mask=None
        )

        x = self.norm1(text_features + corrected)
        x = self.norm2(x + self.ffn(x))

        # gating
        g = self.gate(torch.cat([text_features, audio_features, visual_features], dim=-1))
        out = g * x + (1 - g) * text_features
        return out


class ModalityReconstructor(nn.Module):
    """
    Reconstruction module for missing modality data recovery.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None):
        super(ModalityReconstructor, self).__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
        self.reconstructor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, available_features):
        # available_features: [B, L, Din]
        return self.reconstructor(available_features)


class IntegrityConstraintLoss(nn.Module):
    """
    Integrity constraint loss:
    total = alpha * reconstruction_mse + beta * consistency_mse
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super(IntegrityConstraintLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, reconstructed_features, target_features,
                corrected_text, original_text, mask=None):
        """
        Args:
            reconstructed_features: [B, L, D]
            target_features: [B, L, D]
            corrected_text: [B, L, D]
            original_text: [B, L, D]
            mask: [B, L] (1=include, 0=exclude) for reconstruction terms only

        Returns:
            total_loss, dict
        """
        recon = self.mse(reconstructed_features, target_features)  # [B, L, D]
        if mask is not None:
            m = mask.unsqueeze(-1).float()  # [B, L, 1]
            recon = (recon * m).sum() / (m.sum() + 1e-6)
        else:
            recon = recon.mean()

        cons = self.mse(corrected_text, original_text).mean()
        total = self.alpha * recon + self.beta * cons
        return total, {'reconstruction_loss': recon, 'consistency_loss': cons, 'total_integrity_loss': total}