import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionVAT(nn.Module):
    def __init__(self, video_dim=512, audio_dim=512, text_dim=768,
                 fusion_type="concat", hidden_dim=256):
        super().__init__()
        self.fusion_type = fusion_type

        # -------------------- FUSION HEADS --------------------
        if fusion_type == "logits":
            self.video_head = nn.Linear(video_dim, 1)
            self.audio_head = nn.Linear(audio_dim, 1)
            self.text_head  = nn.Linear(text_dim, 1)

            self.video_w = nn.Parameter(torch.tensor(1/3, dtype=torch.float32))
            self.audio_w = nn.Parameter(torch.tensor(1/3, dtype=torch.float32))
            self.text_w  = nn.Parameter(torch.tensor(1/3, dtype=torch.float32))

        elif fusion_type == "concat":
            fused_dim = video_dim + audio_dim + text_dim
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        elif fusion_type == "attention":
            self.proj_v = nn.Linear(video_dim, hidden_dim)
            self.proj_a = nn.Linear(audio_dim, hidden_dim)
            self.proj_t = nn.Linear(text_dim, hidden_dim)

            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def forward(self, v, a, t, label=None):
        if self.fusion_type == "logits":
            v_logit = self.video_head(v)
            a_logit = self.audio_head(a)
            t_logit = self.text_head(t)

            logits = self.video_w * v_logit + self.audio_w * a_logit + self.text_w * t_logit

        elif self.fusion_type == "concat":
            x = torch.cat([v, a, t], dim=1)
            logits = self.fusion(x)

        elif self.fusion_type == "attention":
            x = torch.stack([
                self.proj_v(v),
                self.proj_a(a),
                self.proj_t(t)
            ], dim=1)

            attended, _ = self.attn(x, x, x)
            pooled = attended.mean(dim=1)
            logits = self.fusion(pooled)

        logits = logits.squeeze(1)
        out = {"logits": logits, "prob": torch.sigmoid(logits)}

        if label is not None:
            out["loss"] = self.loss_fn(logits, label.float())

        return out
