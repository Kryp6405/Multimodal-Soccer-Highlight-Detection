import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionTransformer(nn.Module):
    def __init__(self, video_dim=512, audio_dim=512, text_dim=768,
                 hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()

        # Project all modalities to same dimension
        self.proj_v = nn.Linear(video_dim, hidden_dim)
        self.proj_a = nn.Linear(audio_dim, hidden_dim)
        self.proj_t = nn.Linear(text_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1,
            activation="relu"
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classifier on [CLS] token (we'll prepend one)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))

    def forward(self, v, a, t, label=None):

        # Normalize embeddings
        # v = F.normalize(v, p=2, dim=1)
        # a = F.normalize(a, p=2, dim=1)
        # t = F.normalize(t, p=2, dim=1)

        # Project to shared space
        v = self.proj_v(v)
        a = self.proj_a(a)
        t = self.proj_t(t)

        # Stack tokens: (B, 3, D)
        tokens = torch.stack([v, a, t], dim=1)

        # Add CLS: (B, 1, D)
        B = tokens.size(0)
        cls_tok = self.cls_token.repeat(B, 1, 1)

        x = torch.cat([cls_tok, tokens], dim=1)  # → (B, 4, D)

        # Transformer layers
        x = self.transformer(x)  # (B, 4, D)

        # Use CLS output
        cls = x[:, 0, :]

        logits = self.classifier(cls).squeeze(1)

        out = {"logits": logits, "prob": torch.sigmoid(logits)}

        if label is not None:
            out["loss"] = self.loss_fn(logits, label.float())

        return out
