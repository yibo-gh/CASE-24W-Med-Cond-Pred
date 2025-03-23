
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
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=batched),
            num_layers=num_layers
        )

        def hook_fn(module, input, output):
            if torch.isnan(output[0]).any():
                print(f"NaN detected in module: {module}")
                exit(254)

        for layer in self.transformer_encoder.layers:
            layer.self_attn.register_forward_hook(hook_fn);

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

