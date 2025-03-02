
import torch;
import torch.nn as nn;


class MedTrans(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 maxVecSize: int = 128,
                 dropout: float=0.1) -> None:
        super().__init__();

        # print(f"t::80 d_model {d_model}")

        self.inDim: int = input_dim;
        self.outDim: int = output_dim;
        self.dModel: int = d_model;
        self.nhead: int = nhead;
        self.numLayers: int = num_layers;
        self.maxVec: int = maxVecSize;
        self.dropout: float = dropout;

        print("t:90", input_dim, d_model)
        self.linear_proj = nn.Linear(input_dim, d_model);
        self.positional_encoding = nn.Parameter(torch.zeros(maxVecSize, d_model));
        # self.transformer = nn.Transformer(
        #     d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers, dropout=dropout, batch_first=True
        # );
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim);

    def forward(self, x: torch.Tensor, xm: torch.Tensor, vecSelector: torch.Tensor, dev: torch.device) -> torch.Tensor:
        x_emb = self.linear_proj(x) + self.positional_encoding[:, :self.dModel]
        # print("t::101", x_emb.shape, self.dModel)
        assert x_emb.size(1) == self.dModel;
        # print("t::107", x_emb.shape, xm.shape)
        # print("t::108", xm.T)
        attn_mask = torch.zeros((len(x), len(x))).to(dev)
        attn_mask[:, :self.dModel][xm == 0] = float('-inf');
        transformer_output = self.transformer_encoder(x_emb, mask=attn_mask, src_key_padding_mask=vecSelector);
        output = self.fc(transformer_output[vecSelector.to(torch.int)]);
        return output

