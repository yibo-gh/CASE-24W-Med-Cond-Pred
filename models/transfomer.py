import math
import os.path
import pickle
import time
from typing import Tuple, List, Dict, Callable;

from torch.utils.data import Dataset;
import torch;
import torch.nn.functional as F;
import numpy as np;
import torch.nn as nn;
from torchmetrics.functional import f1_score;

from obj.DataProcessor import DataProcessor;
from util.util import npF1, auroc, auprc, mrr, histk, execCmd, plot_confidence_heatmap, plot_sorted_metrics, detect_metric_outliers;

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
          sche: torch.optim.Optimizer,
          dev: torch.device,
          outAct: Callable = F.softmax) -> float:
    # res: torch.Tensor = outAct(model(X.to(dev), xm.to(dev), ym.to(dev), dev), dim=-1);
    res: torch.Tensor = model(X.to(dev), xm.to(dev), ym.to(dev), dev);
    ymOriAcc: torch.Tensor = yOri.to(torch.int).to(dev);
    loss = lossFn(res[ym == 1], y[ym == 1].to(dev)).sum() / ymOriAcc.sum();
    del ymOriAcc;
    opt.zero_grad();
    loss.backward();
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
    opt.step();
    return loss.item();


def register_forward_hooks(model):
    hooks = []

    def hook_fn(module, input, output):
        name = module.__class__.__name__
        print(f"{name:<30} | input: {tuple(input[0].shape)} → output: {tuple(output.shape)}")

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只 hook 到底层（无子模块）层
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)

    return hooks


def evaluate(model: nn.Module,
             X: torch.Tensor, xm: torch.Tensor,
             y: torch.Tensor, ym: torch.Tensor, yOri: torch.Tensor,
             lossFn: nn.Module,
             dev: torch.device,
             outAct: Callable = F.softmax) -> Tuple[float, int, int, torch.Tensor, torch.Tensor]:
    model.eval();
    # with torch.no_grad():
    if True:
        # res = outAct(model(X.to(dev), xm.to(dev), ym.to(dev), dev).cpu(), dim=-1);
        _ymd: torch.Tensor = ym.to(dev);
        # register_forward_hooks(model);
        res: torch.Tensor = model(X.to(dev), xm.to(dev), _ymd.to(dev), dev);
        # print("t::109", X.shape, xm.shape, _ymd.shape);
        # exit(0)
        ymOriAcc: torch.Tensor = yOri.to(torch.int);
        loss = lossFn(res[ym == 1], y.to(dev)[_ymd == 1]).sum() / ymOriAcc.sum();
        del ymOriAcc;
        del _ymd;
        res = res.cpu();

        for b in range(res.size(0)):
            for s in range(res.size(1)):
                __entry: torch.Tensor = res[b][s];
                __y: torch.Tensor = y[b][s];
                if torch.sum(yOri[b][s]) == 0:
                    continue;
                _, __sortedIdx = torch.sort(__entry);
                __entry[__sortedIdx[:int(sum(__y))]] = 1;
                __entry[__sortedIdx[int(sum(__y))]:] = 0;
        # res = (res >= .5).float();
        # res[res != 1] = 0;
        # print("t::117", torch.sum(res), torch.sum(y))
        # f1: torch.Tensor = f1_score(res[ym == 1], y[ym == 1], task="multiclass", num_classes=res.size(2));
        com: torch.Tensor = res[ym == 1] == y[ym == 1];
        total: int = int(com.view(-1).size(0));
    return loss.item(), total, int(torch.sum(com)), res[ym == 1], y[ym == 1];


def __service_prepDt4training(dt: torch.Tensor, maxVec: int, mFeature: int, mask: bool = False, dtype=torch.float32) -> torch.Tensor:
    if (len(dt.shape) + (0 if not mask else 1)) == 2:
        ret: torch.Tensor = torch.zeros((maxVec, max(dt.size(1), mFeature)) if not mask else maxVec, dtype=dtype);
        if not mask:
            ret[:len(dt), :dt.size(1)] = dt;
        else:
            ret[:len(dt)] = dt;
    else:
        assert (len(dt.shape) + (0 if not mask else 1)) == 3;
        ret: torch.Tensor = torch.zeros((len(dt), maxVec, max(dt.size(2), mFeature)) if not mask else (len(dt), maxVec), dtype=dtype);
        if not mask:
            ret[:, :dt.size(1), :dt.size(2)] = dt;
        else:
            ret[:, :dt.size(1)] = dt;
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


def __service_iterTrain(model: nn.Module,
                        dp: DataProcessor,
                        loss_fn: nn.Module,
                        opt: torch.optim.Optimizer,
                        sche: torch.optim.lr_scheduler.LRScheduler,
                        maxVec: int,
                        mFeature: int,
                        dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        ) -> float:
    trainLoss: float = 0;
    for j in range(dp.getBatchCount(0)):
        # print(f"t::206 Training Batch {j + 1:4d}/{dp.getBatchCount()} of Epoch {i + 1:4d}/{epoch}")
        xd, xm, xo, yd, ym, yo = dp[j, 0];
        # print("t::156", xd.shape, xm.shape, xo.shape, yd.shape, ym.shape, yo.shape)
        tLoss: float = train(model,
                             __service_prepDt4training(torch.from_numpy(xd), maxVec, mFeature=0),
                             __service_prepDt4training(torch.from_numpy(xm), maxVec, mFeature=mFeature),
                             __service_prepDt4training(torch.from_numpy(yd), maxVec, mFeature=0),
                             __service_prepDt4training(torch.from_numpy(yo), maxVec, mFeature=0, mask=True),
                             __service_prepDt4training(torch.from_numpy(ym), maxVec, mFeature=0),
                             loss_fn, opt, sche, dev=dev);
        trainLoss += tLoss;
        # break;
    return trainLoss;


def __service_iterVali(model: nn.Module,
                       dp: DataProcessor,
                       loss_fn: nn.Module,
                       maxVec: int,
                       mFeature: int,
                       dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                       ) -> float:
    valLoss: float = 0;

    for j in range(dp.getBatchCount(1)):
        xd, xm, xo, yd, ym, yo = dp[j, 1];
        loss =(
            evaluate(model,
                     __service_prepDt4training(torch.from_numpy(xd), maxVec, mFeature=0),
                     __service_prepDt4training(torch.from_numpy(xm), maxVec, mFeature=mFeature),
                     __service_prepDt4training(torch.from_numpy(yd), maxVec, mFeature=0),
                     __service_prepDt4training(torch.from_numpy(yo), maxVec, mFeature=0, mask=True),
                     __service_prepDt4training(torch.from_numpy(ym), maxVec, mFeature=0),
                     loss_fn, dev=dev))[0];
        # print("t::242", loss, total, corr)
        valLoss += loss;
        # break;
    return valLoss;


def __service_iterEval(model: nn.Module,
                       dp: DataProcessor,
                       loss_fn: nn.Module,
                       maxVec: int,
                       mFeature: int,
                       dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                       ) -> Tuple[np.ndarray, np.ndarray, float, int, int, float | List[float], float, float, float, float, float]:
    valLoss: float = 0;
    vTotal: int = 0;
    vCorr: int = 0;
    f1List: Tuple[List[np.ndarray]] = ([], []);
    __devCPU: torch.device = torch.device("cpu");

    for j in range(dp.getBatchCount(2)):
        xd, xm, xo, yd, ym, yo = dp[j, 2];
        # print("t::169", xd.shape, xm.shape, xo.shape, yd.shape, ym.shape, yo.shape)
        # print(f"t::206 Validating Batch {j + 1:4d}/{dp.getBatchCount(False)} of Epoch {i + 1:4d}/{epoch}")
        loss, total, corr, resT, gtT =(
            evaluate(model,
                     __service_prepDt4training(torch.from_numpy(xd), maxVec, mFeature=0),
                     __service_prepDt4training(torch.from_numpy(xm), maxVec, mFeature=mFeature),
                     __service_prepDt4training(torch.from_numpy(yd), maxVec, mFeature=0),
                     __service_prepDt4training(torch.from_numpy(yo), maxVec, mFeature=0, mask=True),
                     __service_prepDt4training(torch.from_numpy(ym), maxVec, mFeature=0),
                     loss_fn, dev=dev));
        # print("t::242", loss, total, corr)
        valLoss += loss;
        vTotal += total;
        vCorr += corr;
        f1List[0].append(resT.cpu().detach().numpy());
        f1List[1].append(gtT.cpu().detach().numpy())
        del resT; del gtT;
        # break;
    gt, yh = np.concatenate(f1List[1]), np.concatenate(f1List[0]);
    f1: float = npF1(gt, yh);
    _, roc = auroc(gt, yh);
    _, prc = auprc(gt, yh);
    mrs: float = mrr(gt, yh);
    his5: float = histk(gt, yh, 5);
    his10: float = histk(gt, yh, 10);
    return gt, yh, valLoss, vTotal, vCorr, f1, roc, prc, mrs, his5, his10;


def iter(model: nn.Module,
         dp: DataProcessor,
         loss_fn: nn.Module,
         opt: torch.optim.Optimizer | None,
         sche: torch.optim.lr_scheduler.LRScheduler | None,
         epoch: int,
         maxVec: int,
         mFeature: int,
         dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         save: str | None = None,
         evalOnly: bool = False) -> None:

    __bestMet: float = 0;
    if save is not None and not os.path.exists(save):
        os.mkdir(save);
        execCmd(f"cp models/trans* {save}/");
        with open(f"{save}/dp.pkl", "wb") as f:
            pickle.dump(dp, f);
    for i in range(epoch):
        trainLoss: float = 0;
        if not evalOnly:
            assert opt is not None and sche is not None;
            trainLoss = __service_iterTrain(
                model=model,
                dp=dp,
                loss_fn=loss_fn,
                opt=opt,
                sche=sche,
                maxVec=maxVec,
                mFeature=mFeature,
                dev=dev
            );

            valiLoss = __service_iterVali(model=model,
                                          dp=dp,
                                          loss_fn=loss_fn,
                                          maxVec=maxVec,
                                          mFeature=mFeature,
                                          dev=dev);
            sche.step(int(round(valiLoss)));

        valLoss: float; vTotal: int; vCorr: int;
        gt, yh, valLoss, vTotal, vCorr, f1, roc, prc, mrr, his5, his10 = __service_iterEval(
            model=model,
            dp=dp,
            loss_fn=loss_fn,
            maxVec=maxVec,
            mFeature=mFeature,
            dev=dev
        );

        _msg: str;
        if not evalOnly:
            _msg = (f"Ep {i + 1:4d}/{epoch} - " +
                    f"LR: {sche.get_last_lr()}, "+
                    f"TrL: {trainLoss:.4f}, " +
                    f"VL: {valiLoss:.4f}, " +
                    f"TeL: {valLoss:.4f}, " +
                    f"TeA: {vCorr * 100 / vTotal:5.2f}%, " +
                    f"F-1: {f1 * 100:5.2f}%, " +
                    f"AUROC {roc * 100:5.2f}%, " +
                    f"AUPRC {prc * 100:5.2f}%, " +
                    f"MRR {mrr * 100:5.2f}%, " +
                    f"Hist-5 {his5 * 100:5.2f}%, " +
                    f"Hist-10 {his10 * 100:5.2f}%");
            print(_msg);
        else:
            _msg = (f"Ep {i + 1:4d}/{epoch} - " +
                    f"TeL: {valLoss:.4f}, " +
                    f"TeA: {vCorr * 100 / vTotal:5.2f}%, " +
                    f"F-1: {f1 * 100:5.2f}%, " +
                    f"AUROC {roc * 100:5.2f}%, " +
                    f"AUPRC {prc * 100:5.2f}%, " +
                    f"MRR {mrr * 100:5.2f}%, " +
                    f"Hist-5 {his5 * 100:5.2f}%, " +
                    f"Hist-10 {his10 * 100:5.2f}%");
            print(_msg);
            # plot_confidence_heatmap(yh, gt, fname="heatmap.png");
            _allRoc, _ = auroc(gt, yh);
            _allPrc, _ = auprc(gt, yh);
            plot_sorted_metrics(_allRoc, _allPrc, fname="metPlot.png")

            rocOl: np.ndarray = detect_metric_outliers(_allRoc)[0];
            prcOl: np.ndarray = detect_metric_outliers(_allPrc, z_thresh=1.5)[0];
            print(f"t::367 roc {np.quantile(_allRoc, [0, .25, .5, .75, .9, .95])}")
            print(np.where(_allRoc == np.min(_allRoc))[0]);
            # print(f"t::368 {rocOl}");
            print(f"t::369 prc {np.quantile(_allPrc, [0, .25, .5, .75, .9, .95])}")
            # print(f"t::370 {prcOl}");

            _testsetPtid: np.ndarray = np.zeros(0);
            for j in range(dp.getBatchCount(2)):
                _testsetPtid = np.concatenate((_testsetPtid, dp[j, 2][0][:, 0, 0]));

            roc2, roc3 = np.quantile(_allRoc, [.5, .75])
            prc2, prc3 = np.quantile(_allPrc, [.5, .75])
            # print(f"t::380 {np.where((roc2 <= _allRoc) & (_allRoc <= roc3))}")
            # print(f"t::381 {np.where((prc2 <= _allPrc) & (_allPrc <= prc3))}")
            commonPool: np.ndarray = np.intersect1d(np.where((roc2 <= _allRoc) & (_allRoc <= roc3))[0], np.where((prc2 <= _allPrc) & (_allPrc <= prc3))[0])
            # print(f"t::383 Common typical {commonPool}")

            '''
            1743 SexAtBirth.Male 1941 1 1001
1749 SexAtBirth.Male 1942 1 1001
            '''
            # print(yh[1743])
            # print(np.argsort(-yh[1743]))
            # print(np.where(gt[1743] == 1)[0])
            # print(yh[1749])
            # print(np.argsort(-yh[1749]))
            # print(np.where(gt[1749] == 1)[0])

            from obj.pt import Pt, EvtClass;
            with open("data/allPt.pkl", "rb") as f:
                allPt: Dict[str, Pt] = pickle.load(f);
            for _pid in commonPool:
                if len(np.where(gt[_pid] == 1)[0]) < 1:
                    continue;
                pt: Pt = allPt[str(int(_testsetPtid[_pid]))];
                print(_pid, pt.dem.sab, pt.dem.aYr, pt.dem.aMo, pt.dem.eth)
                _digList: List[str] = [];
                for _e in pt.evtList:
                    if _e.type != EvtClass.Dig:
                        continue;
                    _digList.append(_e.cont[0]);
                print(_digList);
                print(np.argsort(-yh[_pid]))
                print(np.where(gt[_pid] == 1)[0])
                print();

        if evalOnly or math.isnan(trainLoss) or math.isnan(valLoss):
            return;
        if save is None:
            continue;
        torch.save(model, f"{save}/{i + 1}.pt");
        with open(f"{save}/{i + 1}.log", "w") as f:
            f.write(_msg);

