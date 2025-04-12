
from typing import Tuple;

import torch;
import torch.nn as nn;
import torch.nn.functional as F;

import numpy as np;

def safe_softmax(logits, dim=-1):
    logits = torch.clamp(logits, min=-10, max=10)
    return F.softmax(logits, dim=dim)


class SafeMultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super(SafeMultiheadAttention, self).__init__(embed_dim, num_heads, dropout=dropout, batch_first=batch_first);
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False):

        if self.batch_first:
            batch_size, tgt_len, embed_dim = query.size()
        else:
            tgt_len, batch_size, embed_dim = query.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q * self.scaling

        if self.batch_first:
            q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            src_len = k.size(2)
            q = q.reshape(batch_size * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(batch_size * self.num_heads, src_len, self.head_dim)
            v = v.reshape(batch_size * self.num_heads, src_len, self.head_dim)
        else:
            q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            k = k.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            v = v.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            tgt_len = q.size(2)
            src_len = k.size(2)
            q = q.reshape(batch_size * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(batch_size * self.num_heads, src_len, self.head_dim)
            v = v.reshape(batch_size * self.num_heads, src_len, self.head_dim)

        attn_scores = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_scores += attn_mask;

        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(key_padding_mask_expanded.to(torch.bool), float('-inf'))
            attn_scores = attn_scores.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_weights = safe_softmax(attn_scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_weights, v);

        if self.batch_first:
            attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        else:
            attn_output = attn_output.reshape(batch_size, self.num_heads, tgt_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(tgt_len, batch_size, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        if need_weights:
            if self.batch_first:
                attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            else:
                attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            avg_attn_weights = attn_weights.sum(dim=1) / self.num_heads
            return attn_output, avg_attn_weights
        else:
            return attn_output


class SafeTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", batch_first=True):
        super(SafeTransformerEncoderLayer, self).__init__(d_model=d_model, nhead=nhead, dropout=dropout, activation=activation, batch_first=batch_first)
        self.self_attn = SafeMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)


class MedTrans(nn.Module):

    '''def __service_hookFn(self, module, input, output):
        if torch.isnan(output[0]).any():
            print(f"NaN detected in module: {module}")
            exit(254)
        if isinstance(output, torch.Tensor):
            clamped = torch.clamp(output, min=-10, max=10)
            return torch.nn.functional.softmax(clamped, dim=-1)
        return output;'''

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 maxVecSize: int = 128,
                 dropout: float=0.1,
                 batched: bool = False,
                 medMat: np.ndarray | None = None) -> None:
        super().__init__();

        # print(f"tv::80 d_model {d_model}")

        self.inDim: int = input_dim;
        self.outDim: int = output_dim;
        self.dModel: int = d_model;
        self.nhead: int = nhead;
        self.numLayers: int = num_layers;
        self.maxVec: int = maxVecSize;
        self.dropout: float = dropout;
        self.medMat: torch.Tensor | None = torch.from_numpy(medMat.astype(np.float32)) if medMat is not None else None;

        # print("tv::90", input_dim, d_model)
        self.linear_proj = nn.Linear(input_dim, d_model);
        self.positional_encoding = nn.Parameter(torch.zeros(maxVecSize, d_model));
        self.transformer_encoder = nn.TransformerEncoder(
            SafeTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=batched),
            num_layers=num_layers
        )

        # for layer in self.transformer_encoder.layers:
        #     layer.self_attn.register_forward_hook(self.__service_hookFn);

        self.t2m1 = nn.Linear(208, self.medMat.size(-1));
        self.t2m3 = nn.Linear(208, 256);
        self.t2m4 = nn.Linear(256, 256);
        self.t2m5 = nn.Linear(256, self.medMat.size(-1));
        self.t2m2 = nn.Linear(self.medMat.size(-1), self.medMat.size(-1));

        self.relu = nn.ReLU();

        '''self.ptMedConv = nn.Sequential(
            nn.Linear(208, 256),
            nn.ReLU(),
            nn.LayerNorm(256),           # 🚨 加一个 norm 稳定训练
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),             # 🚨 可选防止过拟合
            nn.Linear(256, self.medMat.size(-1))
        )'''
        '''self.ptMedConv = nn.Sequential(
            nn.Linear(208, self.medMat.size(-1))
        )'''
        self.ptMedConv = nn.Sequential(
            # nn.LayerNorm(208),
            nn.Linear(208, 512),
            # nn.Dropout(.1),
            # nn.ReLU(),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            # nn.Dropout(.1),
            # nn.LayerNorm(208),
            nn.Linear(512, self.medMat.size(-1))
        )

    def setMedMat(self, dt: np.ndarray) -> None:
        # self.medMat = torch.nn.functional.normalize(torch.from_numpy(dt.astype(np.float32)), dim=-1);
        self.medMat = torch.from_numpy(dt.astype(np.float32));
        self.ptMedConv = nn.Sequential(
            nn.Linear(208, self.medMat.size(-1))
        )

    def forward(self, x: torch.Tensor, xm: torch.Tensor, vecSelector: torch.Tensor, dev: torch.device) -> torch.Tensor:
        assert self.medMat is not None and len(x.shape) == 3;
        x_emb = self.linear_proj(x) + self.positional_encoding[:, :self.dModel];
        assert x_emb.size(-1) == self.dModel;

        skpm: torch.Tensor = torch.zeros((x.shape[0], x.size(-2)), dtype=torch.bool).to(dev);
        skpm[xm.sum(dim=-1) == 0] = 1;
        transformer_output = self.transformer_encoder(x_emb, mask=None, src_key_padding_mask=skpm);
        assert torch.sum(torch.sum(vecSelector, dim=-1) == 1) == len(transformer_output);

        # print(transformer_output.shape)
        itmd: torch.Tensor = self.ptMedConv(transformer_output);

        itmd = itmd.sum(dim=-2);
        itmd = itmd.matmul(self.medMat.to(dev).t())

        return itmd.unsqueeze(1).expand(-1, x.size(-2), -1);

