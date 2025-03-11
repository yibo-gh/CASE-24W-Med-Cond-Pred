
from typing import Tuple, List, Dict;

from torch.utils.data import Dataset;
import torch;
import numpy as np;
import torch.nn as nn;
from obj.DataProcessor import DataProcessor;

class PtDS(Dataset):

    x: torch.Tensor;
    y: torch.Tensor;
    xm: torch.Tensor;
    ym: torch.Tensor;
    ymOri: torch.Tensor;
    pidEntryMap: Dict[int, List[int]];
    pidList: List[int];

    def __init__(self,
                 X: np.ndarray | torch.Tensor,
                 xMask: np.ndarray | torch.Tensor,
                 y: np.ndarray | torch.Tensor,
                 mask: np.ndarray | torch.Tensor) -> None:
        assert len(X) == len(xMask) == len(y) == len(mask) and X.shape == xMask.shape and len(mask.shape) == 1 and len(y) == len(mask);

        self.x = torch.from_numpy(X) if isinstance(X, np.ndarray) else X;
        self.xm = torch.from_numpy(xMask) if isinstance(xMask, np.ndarray) else xMask;
        self.y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y;
        self.ym = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask;
        self.ymOri = torch.zeros(self.y.shape, dtype=torch.int);
        self.ymOri[y != 0] = 1;
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


def train(model: nn.Module,
          X: torch.Tensor, xm: torch.Tensor,
          y: torch.Tensor, ym: torch.Tensor, yOri: torch.Tensor,
          lossFn: nn.Module, opt: torch.optim.Optimizer,
          dev: torch.device) -> float:
    # print("t::124", dev)
    res: torch.Tensor = model(X.to(dev), xm.to(dev), ym.to(dev), dev);
    # print("t::128", ymOri.shape, lossFn(res, y).shape)
    ymOriAcc: torch.Tensor = yOri.to(torch.int).to(dev);
    loss = (ymOriAcc * lossFn(res, y.to(dev))).sum() / ymOriAcc.sum();
    del ymOriAcc;
    loss.backward();
    opt.step();
    return loss.item();


def evaluate(model: nn.Module,
             X: torch.Tensor, xm: torch.Tensor,
             y: torch.Tensor, ym: torch.Tensor, yOri: torch.Tensor,
             lossFn: nn.Module,
             dev: torch.device) -> Tuple[float, int, int]:
    model.eval();
    with torch.no_grad():
        res = model(X.to(dev), xm.to(dev), ym.to(dev), dev);
        ymOriAcc: torch.Tensor = yOri.to(torch.int).to(dev);
        loss = (ymOriAcc * lossFn(res, y.to(dev))).sum() / ymOriAcc.sum();
        del ymOriAcc;
        res = (torch.softmax(res.cpu(), dim=1) >= .5).float();
        com: torch.Tensor = res[ym == 1] == y[ym == 1];
        total: int = int(com.view(-1).size(0));
    return loss.item(), total, int(torch.sum(com));


def __service_prepDt4training(dt: torch.Tensor, maxVec: int, mFeature: int, mask: bool = False, dtype=torch.float32) -> torch.Tensor:
    # print(dt.shape)
    # ret: torch.Tensor = torch.zeros((maxVec, max(dt.size(1), mFeature)) if not mask else maxVec, dtype=torch.float32);
    ret: torch.Tensor = torch.zeros((maxVec, max(dt.size(1), mFeature)) if not mask else maxVec, dtype=dtype);
    if not mask:
        ret[:len(dt), :dt.size(1)] = dt;
    else:
        ret[:len(dt)] = dt;
    # print("t::143", ret.shape)
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
         dp: DataProcessor,
         loss_fn: nn.Module,
         opt: torch.optim.Optimizer,
         epoch: int,
         maxVec: int,
         mFeature: int,
         dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:

    for i in range(epoch):
        trainLoss: float = 0;
        valLoss: float = 0;
        vTotal: int = 0;
        vCorr: int = 0;

        for j in range(dp.getBatchCount()):
            print(f"t::206 Training Batch {j + 1:4d}/{dp.getBatchCount()} of Epoch {i + 1:4d}/{epoch}")
            xd, xm, xo, yd, ym, yo = dp[j, True];
            # print("t::156", xd.shape, xm.shape, xo.shape, yd.shape, ym.shape, yo.shape)
            tLoss: float = train(model,
                                 __service_prepDt4training(torch.from_numpy(xd), maxVec, mFeature=0),
                                 __service_prepDt4training(torch.from_numpy(xm), maxVec, mFeature=mFeature),
                                 __service_prepDt4training(torch.from_numpy(yd), maxVec, mFeature=0),
                                 __service_prepDt4training(torch.from_numpy(yo), maxVec, mFeature=0, mask=True),
                                 __service_prepDt4training(torch.from_numpy(ym), maxVec, mFeature=0),
                                 loss_fn, opt, dev=dev);
            trainLoss += tLoss;
            # break;

        for j in range(dp.getBatchCount(False)):
            xd, xm, xo, yd, ym, yo = dp[j, False];
            # print("t::169", xd.shape, xm.shape, xo.shape, yd.shape, ym.shape, yo.shape)
            print(f"t::206 Validating Batch {j + 1:4d}/{dp.getBatchCount(False)} of Epoch {i + 1:4d}/{epoch}")
            loss, total, corr = evaluate(model,
                                         __service_prepDt4training(torch.from_numpy(xd), maxVec, mFeature=0),
                                         __service_prepDt4training(torch.from_numpy(xm), maxVec, mFeature=mFeature),
                                         __service_prepDt4training(torch.from_numpy(yd), maxVec, mFeature=0),
                                         __service_prepDt4training(torch.from_numpy(yo), maxVec, mFeature=0, mask=True),
                                         __service_prepDt4training(torch.from_numpy(ym), maxVec, mFeature=0),
                                         loss_fn, dev=dev);
            # print("t::242", loss, total, corr)
            valLoss += loss;
            vTotal += total;
            vCorr += corr;
            # break;
        print(f"Epoch {i + 1:4d}/{epoch} - Train Loss: {trainLoss:.4f}, Validation Loss: {valLoss:.4f}, Validation Acc: {vCorr * 100 / vTotal:5.2f}%");

