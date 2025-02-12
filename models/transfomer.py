
from typing import Tuple, List, Dict;

from torch.utils.data import Dataset, DataLoader;
import torch;
import numpy as np;
import torch.nn as nn;

class PtDS(Dataset):

    x: torch.Tensor;
    y: torch.Tensor;
    xm: torch.Tensor;
    ym: torch.Tensor;
    pidEntryMap: Dict[int, List[int]];
    pidList: List[int];

    def __init__(self, X: np.ndarray, xMask: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        assert len(X) == len(y) == len(mask) and X.shape == xMask.shape and y.shape == mask.shape;
        assert y.shape == mask.shape;
        self.x = torch.from_numpy(X);
        self.xm = torch.from_numpy(xMask);
        self.y = torch.from_numpy(y);
        self.ym = torch.from_numpy(mask);
        self.pidEntryMap = dict();

        for i in range(len(X)):
            pid: int = int(X[i][0]);
            # print(pid, i);
            try:
                self.pidEntryMap[pid].append(i);
            except KeyError:
                self.pidEntryMap[pid] = [i];
        self.pidList = list(self.pidEntryMap.keys());

    def __len__(self) -> int:
        return len(self.pidList);

    def getPtListLen(self) -> int:
        return len(self.pidList);

    def getPtRows(self, idx: int | List[int] | slice) -> np.ndarray:
        if isinstance(idx, int):
            return np.array(self.pidEntryMap[self.pidList[idx]]);
        elif isinstance(idx, list):
            __tmpEntry: List[int] = [];
            for i in idx:
                __tmpEntry += self.pidEntryMap[self.pidList[i]];
            return np.array(__tmpEntry);
        elif isinstance(idx, slice):
            __tmpEntry: List[int] = [];
            for i in range(*idx.indices(len(self.pidList))):
                __tmpEntry += self.pidEntryMap[self.pidList[i]];
            return np.array(__tmpEntry);
        else:
            raise TypeError;

    def __getitem__(self, idx: int | List[int] | slice) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        __ptEntry: torch.Tensor = torch.from_numpy(self.getPtRows(idx));
        return self.x[__ptEntry], self.xm[__ptEntry], self.y[__ptEntry], self.ym[__ptEntry];


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

        self.inDim: int = input_dim;
        self.outDim: int = output_dim;
        self.dModel: int = d_model;
        self.nhead: int = nhead;
        self.numLayers: int = num_layers;
        self.maxVec: int = maxVecSize;
        self.dropout: float = dropout;

        self.linear_proj = nn.Linear(input_dim, d_model);
        self.positional_encoding = nn.Parameter(torch.zeros(maxVecSize, d_model));
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dropout=dropout, batch_first=True
        );
        self.fc = nn.Linear(d_model, output_dim);

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.linear_proj(src) + self.positional_encoding[:src.size(1)]
        tgt_emb = self.linear_proj(tgt) + self.positional_encoding[:tgt.size(1)]

        transformer_output = self.transformer(
            src_emb, tgt_emb, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask
        )

        output = self.fc(transformer_output)
        return output


def train(model: nn.Module,
          X: torch.Tensor, xm: torch.Tensor,
          y: torch.Tensor, ym: torch.Tensor,
          lossFn: nn.Module, opt: torch.optim.Optimizer) -> float:
    loss = lossFn(model(X, xm, y, ym).view(-1, model.outDim), y.view(-1));
    loss.backward();
    opt.step();
    return loss.item();


def evaluate(model: nn.Module,
             X: torch.Tensor, xm: torch.Tensor,
             y: torch.Tensor, ym: torch.Tensor,
             loss_fn: nn.Module) -> float:
    model.eval();
    with torch.no_grad():
        output = model(X, y, xm, ym);
        loss = loss_fn(output.view(-1, model.fc.out_features), y.view(-1));
    return loss.item();


def iter(model: nn.Module,
         trainDs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
         testDs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
         loss_fn: nn.Module,
         opt: torch.optim.Optimizer,
         epoch: int) -> None:
    for i in range(epoch):
        x, xm, y, ym = trainDs;
        trainLoss: float = train(model, x, xm, y, ym, loss_fn, opt);
        x, xm, y, ym = testDs;
        valLoss: float = evaluate(model, x, xm, y, ym, loss_fn);
        print(f"Epoch {i + 1}/{epoch} - Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}");


def __service_sampleTrain() -> None:
    model: MedTrans = MedTrans(input_dim, output_dim, d_model, nhead, num_layers)
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9);

