
from typing import Tuple, List, Dict;

from torch.utils.data import Dataset;
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

    def __init__(self,
                 X: np.ndarray | torch.Tensor,
                 xMask: np.ndarray | torch.Tensor,
                 y: np.ndarray | torch.Tensor,
                 mask: np.ndarray | torch.Tensor) -> None:
        assert len(X) == len(y) == len(mask) and X.shape == xMask.shape and y.shape == mask.shape;
        assert y.shape == mask.shape;
        self.x = torch.from_numpy(X) if isinstance(X, np.ndarray) else X;
        self.xm = torch.from_numpy(xMask) if isinstance(xMask, np.ndarray) else xMask;
        self.y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y;
        self.ym = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask;
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

    def getPtRows(self, idx: int | List[int] | np.ndarray | slice) -> np.ndarray:
        if isinstance(idx, int):
            return np.array(self.pidEntryMap[self.pidList[idx]]);
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
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
        # self.transformer = nn.Transformer(
        #     d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers, dropout=dropout, batch_first=True
        # );
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim);

    def forward(self, x: torch.Tensor, xm: torch.Tensor, vecSelector: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.dModel;
        x_emb = self.linear_proj(x) + self.positional_encoding[:, :self.dModel]
        transformer_output = self.transformer_encoder(x_emb, mask=xm);
        output = self.fc(transformer_output[vecSelector]);
        return output


def train(model: nn.Module,
          X: torch.Tensor, xm: torch.Tensor,
          y: torch.Tensor, ym: torch.Tensor,
          lossFn: nn.Module, opt: torch.optim.Optimizer,
          dev: torch.device) -> float:

    res: torch.Tensor = model(X.to(dev), xm.to(dev), y.to(dev), ym.to(dev));
    loss = lossFn(res.view(-1, model.outDim), y.view(-1));
    loss.backward();
    opt.step();
    return loss.item();


def evaluate(model: nn.Module,
             X: torch.Tensor, xm: torch.Tensor,
             y: torch.Tensor, ym: torch.Tensor,
             loss_fn: nn.Module,
             dev: torch.device) -> float:
    model.eval();
    with torch.no_grad():
        output = model(X.to(dev), y.to(dev), xm.to(dev), ym.to(dev));
        loss = loss_fn(output.view(-1, model.fc.out_features), y.view(-1));
    return loss.item();


def __service_prepDt4training(dt: torch.Tensor, maxVec: int, mask: bool = False) -> torch.Tensor:
    # print(dt.shape)
    ret: torch.Tensor = torch.zeros((maxVec, dt.size(1)) if not mask else maxVec, dtype=torch.float32);
    ret[:len(dt)] = dt;
    return ret;


def __service_flattenMask(dt: torch.Tensor) -> torch.Tensor:
    return torch.ones(len(dt));


def __service_batching(ds: PtDS, maxVec: int) -> List[np.ndarray]:
    ret: List[List[int]] = [];
    ptVecSize: List[int] = [len(ds[i][0]) for i in range(len(ds.pidList))];
    left: int = 0;
    while left < len(ptVecSize):
        right: int = left + 1;
        while right < len(ptVecSize) and sum(ptVecSize[left:right]) <= maxVec:
            # print("t::147", left, right, sum(ptVecSize[left:right]))
            right += 1;
        ret.append([i for i in range(left, right - 1)]);
        left = right - 1;
        if right == len(ptVecSize):
            left = len(ptVecSize) + 1;
    return [ds.getPtRows(r) for r in ret];


def iter(model: nn.Module,
         trainDs: PtDS,
         testDs: PtDS,
         loss_fn: nn.Module,
         opt: torch.optim.Optimizer,
         epoch: int,
         maxVec: int,
         dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:

    print("t::163 Batching training set")
    tBatch: List[np.ndarray] = __service_batching(trainDs, maxVec);
    print("t::163 Batching testing set")
    vBatch: List[np.ndarray] = __service_batching(testDs, maxVec);
    print("t::167 Batching complete")
    # print(tBatch[0])

    for i in range(epoch):
        print(f"t::168 training {i + 1:4d}/{epoch}");
        trainLoss: float = 0;
        valLoss: float = 0;

        for b in tBatch:
            # print("t::171", b)
            trainLoss += train(model,
                               __service_prepDt4training(trainDs.x[torch.from_numpy(b)], maxVec),
                               __service_prepDt4training(__service_flattenMask(trainDs.xm)[torch.from_numpy(b)], maxVec, mask=True),
                               __service_prepDt4training(trainDs.y[torch.from_numpy(b)], maxVec),
                               __service_prepDt4training(__service_flattenMask(trainDs.ym)[torch.from_numpy(b)], maxVec, mask=True),
                               loss_fn, opt, dev=dev);

        for b in vBatch:
            valLoss += evaluate(model,
                                __service_prepDt4training(testDs.x[torch.from_numpy(b)], maxVec),
                                __service_prepDt4training(__service_flattenMask(testDs.xm)[torch.from_numpy(b)], maxVec, mask=True),
                                __service_prepDt4training(testDs.y[torch.from_numpy(b)], maxVec),
                                __service_prepDt4training(__service_flattenMask(testDs.ym)[torch.from_numpy(b)], maxVec, mask=True),
                                  loss_fn, dev=dev);
        print(f"Epoch {i + 1:4d}/{epoch} - Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}");

