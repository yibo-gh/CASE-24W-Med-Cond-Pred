
import torch;
import torch.nn as nn;
import torch.nn.functional as F;

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

        # 定义 q, k, v 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 定义输出投影层
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
        # 根据 batch_first 标志决定输入数据的解析方式
        if self.batch_first:
            # 输入 shape: (batch_size, tgt_len, embed_dim)
            batch_size, tgt_len, embed_dim = query.size()
        else:
            # 输入 shape: (tgt_len, batch_size, embed_dim)
            tgt_len, batch_size, embed_dim = query.size()

        # 投影 q, k, v
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 缩放 q
        q = q * self.scaling

        if self.batch_first:
            # 重塑为 (batch_size, tgt_len, num_heads, head_dim)，然后转置为 (batch_size, num_heads, tgt_len, head_dim)
            q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            src_len = k.size(2)
            # 将 batch 和头数合并，变为 (batch_size*num_heads, tgt_len, head_dim) 等
            q = q.reshape(batch_size * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(batch_size * self.num_heads, src_len, self.head_dim)
            v = v.reshape(batch_size * self.num_heads, src_len, self.head_dim)
        else:
            # 非 batch_first 模式：输入 shape: (tgt_len, batch_size, embed_dim)
            q = q.view(tgt_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            k = k.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            v = v.view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
            tgt_len = q.size(2)
            src_len = k.size(2)
            q = q.reshape(batch_size * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(batch_size * self.num_heads, src_len, self.head_dim)
            v = v.reshape(batch_size * self.num_heads, src_len, self.head_dim)

        # 计算注意力得分，形状: (batch_size*num_heads, tgt_len, src_len)
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_scores += attn_mask  # attn_mask 应能广播到此形状

        if key_padding_mask is not None:
            # key_padding_mask 应该为 (batch_size, src_len)
            # 扩展 mask 至 (batch_size, 1, 1, src_len)
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # 先将 attn_scores reshape 为 (batch_size, num_heads, tgt_len, src_len)
            attn_scores = attn_scores.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(key_padding_mask_expanded.to(torch.bool), float('-inf'))
            # 再 reshape 回 (batch_size*num_heads, tgt_len, src_len)
            attn_scores = attn_scores.view(batch_size * self.num_heads, tgt_len, src_len)

        # 应用 safe softmax 来计算注意力权重
        attn_weights = safe_softmax(attn_scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 计算注意力输出
        attn_output = torch.bmm(attn_weights, v)  # 形状: (batch_size*num_heads, tgt_len, head_dim)

        # 恢复形状
        if self.batch_first:
            attn_output = attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
            # 转置回 (batch_size, tgt_len, num_heads, head_dim) 并 reshape
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        else:
            attn_output = attn_output.reshape(batch_size, self.num_heads, tgt_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(tgt_len, batch_size, self.embed_dim)

        # 最后经过输出投影层
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # 如果需要返回 attention 权重，则平均各个头
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
        # 用 SafeMultiheadAttention 替换 self_attn
        self.self_attn = SafeMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)



class MedTrans(nn.Module):

    def __service_hookFn(self, module, input, output):
        if torch.isnan(output[0]).any():
            print(f"NaN detected in module: {module}")
            exit(254)
        if isinstance(output, torch.Tensor):
            clamped = torch.clamp(output, min=-10, max=10)
            return torch.nn.functional.softmax(clamped, dim=-1)
        return output;

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 maxVecSize: int = 128,
                 dropout: float=0.1,
                 batched: bool = False) -> None:
        super().__init__();

        # print(f"tv::80 d_model {d_model}")

        self.inDim: int = input_dim;
        self.outDim: int = output_dim;
        self.dModel: int = d_model;
        self.nhead: int = nhead;
        self.numLayers: int = num_layers;
        self.maxVec: int = maxVecSize;
        self.dropout: float = dropout;

        # print("tv::90", input_dim, d_model)
        self.linear_proj = nn.Linear(input_dim, d_model);
        self.positional_encoding = nn.Parameter(torch.zeros(maxVecSize, d_model));
        # self.transformer = nn.Transformer(
        #     d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers, dropout=dropout, batch_first=True
        # );
        self.transformer_encoder = nn.TransformerEncoder(
            SafeTransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=batched),
            num_layers=num_layers
        )

        for layer in self.transformer_encoder.layers:
            layer.self_attn.register_forward_hook(self.__service_hookFn);

        self.fc = nn.Linear(d_model, output_dim);
        self.fc1 = nn.Linear(d_model, 256);
        self.fc2 = nn.Linear(256, 256);
        self.fc3 = nn.Linear(256, output_dim);

    def forward(self, x: torch.Tensor, xm: torch.Tensor, vecSelector: torch.Tensor, dev: torch.device) -> torch.Tensor:
        # print(f"tv::45 {torch.sum(x)}")
        # print("tv::55", x.shape)
        assert len(x.shape) == 3;
        x_emb = self.linear_proj(x) + self.positional_encoding[:, :self.dModel]
        # print(f"tv::46-0 {torch.sum(self.linear_proj(x))}")
        # print(f"tv::46-1 {torch.sum(self.positional_encoding)}")
        # print(f"tv::46-2 {torch.sum(x_emb)}")
        # print("tv::101", x_emb.shape, self.dModel)
        assert x_emb.size(-1) == self.dModel;
        # print("tv::107", x_emb.shape, xm.shape)
        # print("tv::108", xm.T)
        # attn_mask = torch.zeros((x.shape[0], x.shape[-1], x.shape[-1])).to(dev)
        # attn_mask += float('-inf');
        # print("tv::67", xm.shape, attn_mask[:, :self.dModel, :self.dModel].shape)
        # attn_mask[:, :self.dModel, :self.dModel][xm == 0] = 1;
        # print(f"tv::54 {torch.sum(vecSelector)}")

        skpm: torch.Tensor = torch.zeros((x.shape[0], x.size(-2)), dtype=torch.bool).to(dev);
        # print(xm.shape)
        skpm[xm.sum(dim=-1) == 0] = 1;
        # print("tv::137", x.shape, skpm.shape)
        # exit(0)
        transformer_output = self.transformer_encoder(x_emb, mask=None, src_key_padding_mask=skpm);
        # print(f"tv::55 {torch.sum(transformer_output)}")
        # if torch.isnan(torch.sum(transformer_output)):
        #     print(f"{torch.sum(transformer_output)}")
        #     exit(255);
        # output = self.fc(transformer_output[vecSelector.to(torch.int)]);
        # print(f"tv::80 {transformer_output.shape}")
        # print(f"tv::81 {vecSelector.shape}")
        # print(vecSelector)
        assert torch.sum(torch.sum(vecSelector, dim=-1) == 1) == len(transformer_output);
        # transOut: torch.Tenor = torch.zeros((len(transformer_output), transformer_output.size(-1))).to(dev).to(transformer_output.dtype);
        # for i in range(len(vecSelector)):
        #     transOut[i] = transformer_output[i][vecSelector[i] == 1]
        output = self.fc1(transformer_output);
        # print(f"tv::58 {torch.sum(output)}");
        for i in range(5):
            output = self.fc2(output);
            # print(f"tv::61-{i} {torch.sum(output)}");
        output = self.fc3(output);
        # print(f"tv::63 {torch.sum(output)}\n");
        return output

