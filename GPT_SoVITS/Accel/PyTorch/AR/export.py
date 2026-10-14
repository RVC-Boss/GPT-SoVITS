import enum
import os
import os.path as osp
import time
from collections.abc import MutableSequence
from pathlib import Path

import torch
import typer
from torch.export import Dim
from torch.nn import functional as F

from ...logger import logger

from .. import nn
from .t2s_model_abc import AttentionABC, FeedForward, T2SDecoderABC, TransformerBlockABC, TransformerDecoderABC


Tensor = torch.Tensor

KVCache = tuple[Tensor, Tensor]

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)


class Stage(str, enum.Enum):
    embed = "embed"
    decode = "decode"


class KVCacheONNX:
    @staticmethod
    def empty(kv_cache):
        assert len(kv_cache) == 2
        k_cache, v_cache = kv_cache

        k_cache[:] = 0
        v_cache[:] = 0

    @staticmethod
    def update_cache(
        input_pos: Tensor, k_val: Tensor, v_val: Tensor, kv_cache: tuple[Tensor, Tensor], cache_idx: Tensor
    ):
        # input_pos: [B, ], k_val: [B, H, 1, D]
        k_out, v_out = kv_cache
        ip0 = input_pos - 1

        k_out[cache_idx, :, ip0, None] = k_val
        v_out[cache_idx, :, ip0, None] = v_val

        return k_out, v_out

    @staticmethod
    def prefill_kv(k_val: Tensor, v_val: Tensor, kv_cache: tuple[Tensor, Tensor]):
        # k_val: [B, S, H, D]
        k_cache, v_cache = kv_cache

        k_cache[..., : k_val.shape[1], :] = k_val.transpose(1, 2)
        v_cache[..., : v_val.shape[1], :] = v_val.transpose(1, 2)

    @staticmethod
    def init_cache(batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype: torch.dtype):
        cache_shape = (batch_size, n_heads, max_seq_length, head_dim)

        return (torch.zeros(cache_shape, dtype=dtype), torch.zeros(cache_shape, dtype=dtype))


class AttentionONNX(AttentionABC):
    def __init__(self, n_heads: int, head_dim: int, max_seq_length: int):
        super().__init__(n_heads, head_dim, max_seq_length)

        self.in_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    def __call__(self, *args, **kwds):  # type: ignore
        pass

    def onnx_prefill(self, x: Tensor, kv_cache: KVCache, attn_mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        torch._check(attn_mask.size(-2) == x.size(-2))

        q, k, v = self.in_proj(x.unsqueeze(0)).chunk(3, dim=-1)

        q, k, v = map(lambda x: x.contiguous().view(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        KVCacheONNX.prefill_kv(k, v, kv_cache)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden_dim)

        output = self.out_proj(attn)

        return output

    def onnx_decode(self, x: Tensor, input_pos: Tensor, kv_cache: KVCache, cache_idx: Tensor, attn_mask: Tensor):
        bsz, seqlen, _ = x.shape

        torch._check(attn_mask.size(-2) == 1)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q, k, v = map(lambda x: x.reshape(bsz, seqlen, self.n_head, self.head_dim), (q, k, v))

        q, k, v = map(lambda x: x.swapaxes(1, 2), (q, k, v))

        kv_cache = KVCacheONNX.update_cache(input_pos, k, v, kv_cache, cache_idx)

        max_idx = int(input_pos.max())

        q, k, v = map(lambda x: x[..., :max_idx, :], (q, *kv_cache))

        mask = attn_mask[..., :max_idx]

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        attn = attn.swapaxes(1, 2).reshape(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj(attn)

        return attn


class TransformerBlockONNX(TransformerBlockABC):
    def __init__(self, n_head: int, ffn_dim: int, hidden_dim: int, max_seq_length: int) -> None:
        super().__init__(n_head, ffn_dim, hidden_dim, max_seq_length)

        self.attention: AttentionONNX = AttentionONNX(n_head, hidden_dim, max_seq_length)  # type: ignore
        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

    def onnx_prefill(self, x: Tensor, attn_mask: Tensor, kv_cache: KVCache):
        h = self.attention_norm(
            x
            + self.attention.onnx_prefill(
                x,
                kv_cache,
                attn_mask,
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))

        return out

    def onnx_decode(self, x: Tensor, input_pos: Tensor, kv_cache: KVCache, cache_idx: Tensor, attn_mask: Tensor):
        h = self.attention_norm(
            x
            + self.attention.onnx_decode(
                x,
                input_pos,
                kv_cache,
                cache_idx,
                attn_mask,
            )
        )
        out = self.ffn_norm(h + self.feed_forward(h))
        return out


class TransformerDecoderONNX(TransformerDecoderABC):
    def __init__(
        self,
        hidden_dim: int,
        n_layer: int,
        n_head: int,
        ffn_dim: int,
        vocab_size: int,
        max_seq_length: int,
        max_batch_size: int,
    ) -> None:
        super().__init__(hidden_dim, n_layer, n_head, ffn_dim, vocab_size, max_seq_length, max_batch_size)

        self.layers: MutableSequence[TransformerBlockONNX] = nn.ModuleList(  # type: ignore
            TransformerBlockONNX(n_head, ffn_dim, hidden_dim, max_seq_length) for _ in range(n_layer)
        )

    def onnx_prefill(self, x: Tensor, mask: Tensor, *kv_caches: KVCache):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.onnx_prefill(
                x,
                mask,
                kv_cache,
            )
        return x

    def onnx_decode(
        self,
        input_pos: Tensor,
        x: Tensor,
        cache_idx: Tensor,
        attn_mask: Tensor,
        *kv_caches: KVCache,
    ):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.onnx_decode(
                x,
                input_pos,
                kv_cache,
                cache_idx,
                attn_mask,
            )

        return x


class T2SDecoderONNX(T2SDecoderABC):
    def __init__(self, config: dict, max_seq_length: int = 1024, max_batch_size: int = 10) -> None:
        super().__init__(config, max_seq_length, max_batch_size)

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.h = TransformerDecoderONNX(
            self.hidden_dim, self.n_layer, self.n_head, self.ffn_dim, self.vocab_size, max_seq_length, max_batch_size
        )

    def pre_forward(self, session) -> tuple[list[Tensor], dict[str, Tensor]]:
        return super().pre_forward(session)

    def post_forward(self, idx: int, session) -> None:
        return super().post_forward(idx, session)

    def embed_onnx_(
        self,
        x: Tensor,
        x_len: Tensor,
        y: torch.Tensor,
        bert_features: Tensor,
    ):
        B = x.shape[0]
        D = self.embedding_dim
        T_TOTAL = 500
        xy_pos = torch.zeros((B, T_TOTAL, D)).to(bert_features[0].dtype)

        bert_features = bert_features.transpose(1, 2)

        y.shape[1]
        y_emb = self.ar_audio_embedding(y)
        self.ar_audio_position.prefill(y_emb)

        for bs, x_, len_, bert_feature in zip(torch.arange(x.shape[0]), x, x_len, bert_features):
            x_emb = self.ar_text_embedding(x_[:len_])

            bert = self.bert_proj(bert_feature[:len_])

            print(bert.shape, bert_feature[:len_])

            return bert, bert_feature[:len_].unsqueeze(0)

            return bert[:20].unsqueeze(0), None
            x_emb = x_emb + bert
            self.ar_text_position.prefill(x_emb.unsqueeze(0))

            xy_pos[None, bs, :len_] = bert
            # xy_pos[None, bs, len_ : len_ + y_len] = y_pos

        return xy_pos[:, -1], None

        return xy_pos[: x.shape[0]], x_len

    def embed_onnx(
        self,
        x: torch.Tensor,  # [B, Tx]
        x_len: torch.Tensor,  # [B]
        y: torch.Tensor,  # [1, Ty, D]
        bert_features: torch.Tensor,  # [B, 1024, Tx]
    ):
        # [B, 1024, Tx] -> [B, Tx, 1024]
        bert_features = bert_features.transpose(1, 2)

        Ty = y.shape[1]
        Tx = x.shape[1]
        B = x.shape[0]
        D = self.embedding_dim
        T_TOTAL = 500

        # mask: [B, Tx], [j] Col < x_len[i]
        col = torch.arange(Tx, device=x.device).unsqueeze(0)  # [1, Tx]
        mask_x = col < x_len.view(-1, 1)  # [B, Tx]
        mask_x3 = mask_x.unsqueeze(-1)  # [B, Tx, 1]

        torch._check((Ty >= 0) and (Ty <= 250), "y_len out of range")
        torch._check((Tx >= 0) and (Tx <= 250), "x_len out of range")

        y_emb = self.ar_audio_embedding(y)  # [1, Ty, D]
        y_pos = self.ar_audio_position.prefill(y_emb)  # [1, Ty, D]

        x_emb_full = self.ar_text_embedding(x)  # [B, Tx, D]
        bert_full = self.bert_proj(bert_features[[0], : x_len[0]])  # [B, Tx, D]

        print(bert_full[0].shape, bert_features[0, : x_len[0]])

        return bert_full[0], bert_features[0, : x_len[0]]

        x_sum_full = x_emb_full + bert_full  # [B, Tx, D]
        x_pos_full = self.ar_text_position.prefill(x_sum_full)  # [B, Tx, D]

        xy_pos = torch.zeros((B, T_TOTAL, D), dtype=x_pos_full.dtype, device=x_pos_full.device)

        xy_pos[:, :Tx, :] = torch.where(
            mask_x3,
            bert_full[:, :Tx, :].to(xy_pos.dtype),
            xy_pos[:, :Tx, :],
        )

        return xy_pos[:, -1], None

        # Start From offset=x_len, Ty
        # [Ty] Index: offsets + [0..Ty-1]
        offsets = x_len.clamp(min=0, max=T_TOTAL - Ty)  # [B]
        idx_y = offsets.unsqueeze(1) + torch.arange(Ty, device=x_pos_full.device)  # [B, Ty]
        # scatter to dim=1
        # expand index to [B, Ty, D]
        idx_y3 = idx_y.unsqueeze(-1).expand(B, Ty, D)
        y_pos_b = y_pos.expand(B, Ty, D).to(xy_pos.dtype)  # [B, Ty, D]
        xy_pos = xy_pos.scatter(1, idx_y3, y_pos_b)

        return xy_pos, x_len


def torchscript_export(model: T2SDecoderONNX, stage="embed"):
    if stage == "embed":
        x = torch.randint(1, 600, (model.max_batch_size, 50))
        x_len = torch.randint(30, 50, (model.max_batch_size,))
        y = torch.randint(1, 600, (1, 200))
        bert_features = torch.rand((model.max_batch_size, 1024, 50))

        x_len[-1] = 50

        mask = torch.arange(x_len.max().item(), device=x.device).unsqueeze(0) < x_len.unsqueeze(1)

        x = x * mask
        bert_features = bert_features * mask.unsqueeze(1)

        try:
            a, c = model.embed_onnx_(x, x_len, y, bert_features)
            b, d = model.embed_onnx(x, x_len, y, bert_features)
            print("-" * 20)
            print(a - b, (a - b).sum(), (a - b).square().mean())
            # print(c - d, (c - d).sum(), (c - d).square().mean())
            exit()
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-8), (a - b).square().mean()

            model.forward = model.embed_onnx
            scripted_model = torch.jit.script(model, example_inputs=[(x, x_len, y, bert_features)])

            onnx_program = torch.onnx.export(
                scripted_model,
                (x, x_len, y, bert_features),
                input_names=["text", "text_len", "prompt", "bert_features"],
                output_names=["xy_pos", "input_pos"],
                dynamic_axes={
                    "text": {0: "Batch_Size", 1: "Sequence_Length_X"},
                    "prompt": {0: "Batch_Size", 1: "Sequence_Length_Y"},
                    "bert_features": {0: "Batch_Size", 1: "Sequence_Length_X"},
                },
                opset_version=21,
                training=False,
                do_constant_folding=True,
                external_data=False,
            )
            assert onnx_program
            onnx_program.save("onnx_export/AR_Embedding_TorchScript.onnx")

        except Exception:
            logger.bind(show_locals=False).exception("")


def dynamo_export(model: T2SDecoderONNX, stage="embed"):
    if stage == "embed":
        x = torch.randint(1, 600, (model.max_batch_size, 50))
        x_len = torch.randint(30, 50, (model.max_batch_size,))
        y = torch.randint(1, 600, (1, 200))
        bert_features = torch.rand((model.max_batch_size, 1024, 50))

        x_len[-1] = 50

        mask = torch.arange(x_len.max().item(), device=x.device).unsqueeze(0) < x_len.unsqueeze(1)

        x = x * mask
        bert_features = (bert_features.transpose(1, 2) * mask.unsqueeze(-1)).transpose(1, 2)

        dynamic_shapes = [
            {
                0: Dim("Batch_Size", min=1, max=4),
                1: Dim("Sequence_Length_X", min=1, max=50),
            },
            {
                0: Dim("Batch_Size", min=1, max=4),
            },
            {
                1: Dim("Sequence_Length_Y", min=1, max=250),
            },
            {
                0: Dim("Batch_Size", min=1, max=4),
                2: Dim("Sequence_Length_X", min=1, max=50),
            },
        ]
        try:
            a = model.embed_onnx_(x, x_len, y, bert_features)[0]
            b = model.embed_onnx(x, x_len, y, bert_features)[0]
            print(a - b, (a - b).square().mean())
            exit()
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-8), (a - b).square().mean()

            model.forward = model.embed_onnx
            onnx_program = torch.onnx.export(
                model,
                (x, x_len, y, bert_features),
                input_names=["text", "text_len", "prompt", "bert_features"],
                output_names=["xy_pos", "input_pos"],
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                opset_version=21,
                training=False,
                do_constant_folding=True,
                external_data=False,
            )
            assert onnx_program
            onnx_program.save("onnx_export/AR_Embedding_Dynamo.onnx")
        except Exception:
            logger.bind(show_locals=False).exception("")


@app.command()
def export(
    ckpt_path: Path = typer.Option(  # noqa: B008
        ...,
        "--ckpt-path",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        show_default=False,
        help="AR Checkpoint",
    ),
    dynamo: bool = typer.Option(False, is_flag=True, flag_value=True, help="Use Torch Dynamo"),
    stages: list[Stage] = typer.Option([Stage.embed], "--stages", help="Stage to export"),  # noqa: B008
):
    os.makedirs("onnx_export", exist_ok=True)
    dict_s1 = torch.load(ckpt_path, "cpu", mmap=True)
    condig = dict_s1["config"]
    model = T2SDecoderONNX(condig, 1500, 4)
    state_dict = dict_s1["weight"]
    model.load_state_dict(state_dict)

    for stage in stages:
        if dynamo:
            dynamo_export(model, stage)
        else:
            torchscript_export(model, stage)


def get_prog_name() -> str:
    script_rel = ".".join(["GPT_SoVITS", "Accelerate", "PyTorch", osp.basename(__file__)]).strip(".py")
    return f"python -s -m {script_rel}"


if __name__ == "__main__":
    t = time.perf_counter()
    app(prog_name=get_prog_name())
    logger.info(f"Exec Time: {time.perf_counter() - t:.2f} secs")
