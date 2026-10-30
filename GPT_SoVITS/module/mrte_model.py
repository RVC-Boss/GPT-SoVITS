# This is Multi-reference timbre encoder

from torch import nn
from module.attentions import MultiHeadAttention


class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels=192,
        hidden_size=512,
        out_channels=192,
        n_heads=4,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc, ssl_mask, text, text_mask, ge):
        if ge == None:
            ge = 0
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)

        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = self.cross_attention(ssl_enc * ssl_mask, text_enc * text_mask, attn_mask) + ssl_enc + ge
        x = self.c_post(x * ssl_mask)
        return x
