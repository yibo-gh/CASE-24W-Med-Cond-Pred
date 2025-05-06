
import os;
from typing import Dict, List, Callable, Tuple;
import pickle;
import sys;

import numpy as np;

sys.path.append("..");
from obj.pt import Pt;
from obj.pt import EvtClass;
from obj.embedder import Embedder;
from obj.embKG1 import KGEmbed;


class PtIterable():
    __icd: str;
    __lineSize: Dict[str, int];
    __ukb2EmbMap: Dict[str, List[np.ndarray]];
    __ptGrouping: List[List[str]];
    __emb: Embedder;
    cwv: np.ndarray;

    def getDbi(self) -> Dict[str, int]:
        return self.__emb.dbIdx;

    def getMedMat(self) -> np.ndarray:
        return self.__emb.medMat;

    def __service_seqTotalLines(self, seq: List[str]) -> int:
        return sum([self.__lineSize[s] for s in seq]);

    def __init__(self, icd: str,
                 allPt: Dict[str, Pt],
                 batchSize: int,
                 epgPkl: str,
                 ebd: Embedder) -> None:
        self.__icd = icd;
        self.__emb = ebd;
        self.cwv = self.__emb.cwv;

        if os.path.exists(epgPkl):
            with open(epgPkl, "rb") as f:
                self.__lineSize, self.__ukb2EmbMap, self.__ptGrouping = pickle.load(f);
                return;

        self.__lineSize, self.__ukb2EmbMap = ebd.dtBatching(icd);
        self.__ptGrouping = [];
        # print("dp::58", self.__lineSize)

        # def __service_seqTotalLines(seq: List[str]) -> int:
        #     return sum([self.__lineSize[s] for s in seq]);

        def __service_isPtQualified(p: Pt) -> bool:
            for e in p.evtList:
                if e.type == EvtClass.Dig and e.cont[0].lower()[:3] == icd.lower()[:3]:
                    return True;
            return False;

        if batchSize == -1:
            for p in list(allPt.keys()):
                if __service_isPtQualified(allPt[p]):
                    self.__ptGrouping.append([p]);
        else:
            __qualifiedPt: List[str] = [];
            for p in list(allPt.keys()):
                if not __service_isPtQualified(allPt[p]):
                    continue;
                __qualifiedPt.append(p);
                __placed: bool = False;
                # print(f"dp::81 {__service_seqTotalLines([p])}")
                for i in range(len(self.__ptGrouping)):
                    if self.__service_seqTotalLines(self.__ptGrouping[i] + [p]) <= batchSize:
                        self.__ptGrouping[i].append(p);
                        __placed = True;
                        break;
                if not __placed:
                    self.__ptGrouping.append([p]);
            # print(self.__ptGrouping);
            # print([__service_seqTotalLines(pg) for pg in self.__ptGrouping]);

        with open(epgPkl, "wb") as f:
            pickle.dump((self.__lineSize, self.__ukb2EmbMap, self.__ptGrouping), f);

    def countTotalPt(self, selector: np.ndarray) -> int:
        return sum([len(self.__ptGrouping[__s]) for __s in np.array([i for i in range(len(self.__ptGrouping))])[selector]]);

    def getBatchNum(self) -> int:
        return len(self.__ptGrouping);

    def split(self, rate: float = .2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        __testIdx: np.ndarray = np.array([False for _ in range(len(self.__ptGrouping))]);
        __testSelector: int = int(round(len(self.__ptGrouping) * rate));
        __test: np.ndarray = np.random.choice(len(self.__ptGrouping), size=__testSelector, replace=False);
        __testIdx[__test] = True;

        __valiIdx: np.ndarray = np.array([False for _ in range(len(self.__ptGrouping))]);
        __valiSelector: int = int(round(len(self.__ptGrouping) * (1 - rate) * rate));
        __vali: np.ndarray = np.random.choice(len(self.__ptGrouping) - np.sum(__testIdx), size=__valiSelector, replace=False);
        # print(np.where(__testIdx != True))
        # print(__vali)
        # print(np.where(__testIdx != True)[0][__vali])
        __valiIdx[np.where(__testIdx != True)[0][__vali]] = True;

        return (__valiIdx == False) & (__testIdx == False), __valiIdx, __testIdx;

    def getBatch(self, idx: int) -> List[str]:
        return (self.__ptGrouping[idx]);

    def fetchCodifiedPt(self, pt: str, medSeqMap: Dict[str, int]) -> Tuple[List[np.ndarray], np.ndarray]:
        pid: int;
        dem: np.ndarray;
        X: List[np.ndarray];
        y: np.ndarray;

        pid, dem, X, y = self.__emb.getPtVec(pt, medSeqMap);
        for i in range(len(X)):
            X[i] = np.concatenate((np.array([pid]), dem, X[i]), dtype=float);
        assert len(X) <= self.__lineSize[pt];
        return X, y;

    def getYGtClassLen(self) -> int:
        return self.__emb.getYGtClassLen();


class DataProcessor:

    __pkl: str;
    __allPt: Dict[str, Pt];
    __tarICD:str;
    __pi: PtIterable;
    __train: np.ndarray;
    __vali: np.ndarray;
    __test: np.ndarray;
    __medSeqMap: dict[str, int];
    __maxPtId: int;
    cwv: np.ndarray;

    def getDbi(self) -> Dict[str, int]:
        return self.__pi.getDbi();

    def getMedMat(self) -> np.ndarray:
        return self.__pi.getMedMat();

    def __init__(self, pkl: str, icd: str, ebd: Embedder, medSeqMapUri: str, epgPkl: str, batchSize: int = 512) -> None:
        assert os.path.exists(pkl) and os.path.exists(medSeqMapUri);
        self.__pkl = pkl;
        self.__tarICD = icd;
        with open(pkl, "rb") as f:
            self.__allPt = pickle.load(f);
        __pids: List[str] = list(self.__allPt.keys());
        __pids.sort(reverse=True);
        self.__maxPtId = int(__pids[0]);
        self.__pi = PtIterable(icd,
                               self.__allPt,
                               batchSize,
                               epgPkl=epgPkl,
                               ebd=ebd);
        self.cwv = self.__pi.cwv;
        self.__train, self.__vali, self.__test = self.__pi.split();
        assert np.sum(self.__train | self.__test | self.__vali) == self.__pi.getBatchNum();
        print("dp::107", np.sum(self.__train), np.sum(self.__vali), np.sum(self.__test), self.__pi.getBatchNum());
        # print("dp::108", self.__pi.countTotalPt(self.__train), self.__pi.countTotalPt(self.__test));
        with open(medSeqMapUri, "rb") as f:
            self.__medSeqMap = pickle.load(f);

    def __service_validateDT(self, dt: np.ndarray, mask: np.ndarray, maskOH: np.ndarray) -> bool:
        assert dt.shape == mask.shape;
        assert len(maskOH.shape) == 1;
        assert len(maskOH) == len(dt) == len(mask);
        return True;

    def getBatchCount(self, set: int = 0) -> int:
        if set == 0:
            return np.sum(self.__train);
        elif set == 1:
            return np.sum(self.__vali);
        elif set == 2:
            return np.sum(self.__test);
        else:
            raise NotImplementedError;

    def __service_makePtMatrix(self, x: List[np.ndarray], y: np.ndarray) -> Tuple[
        np.ndarray,     # X
        np.ndarray,     # X Mask
        np.ndarray,     # X Mask One-hot
        np.ndarray,     # Y GT
        np.ndarray,     # Y Mask
        np.ndarray      # y Mask One-hot
    ]:
        maxX: int = 0;
        for __ele in x:
            if len(__ele) > maxX:
                maxX = len(__ele);
        retXd: np.ndarray = np.zeros((len(x), maxX), dtype=float);
        retXm: np.ndarray = np.zeros((len(x), maxX), dtype=int);
        for i in range(len(x)):
            retXd[i][:len(x[i])] = x[i];
            retXm[i][:len(x[i])] = 1;

        retYd: np.ndarray = np.zeros((len(x), len(y)), dtype=int);
        retYm: np.ndarray = np.zeros((len(x), len(y)), dtype=int);
        retYd[-1] = y;
        retYm[-1] = 1;
        retYo: np.ndarray = np.zeros(len(x), dtype=int);
        retYo[-1] = 1;
        return retXd, retXm, np.ones(len(x), dtype=int), retYd, retYm, retYo;

    def __service_padDt(self, dt: List[
            Tuple[
                np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray
            ]
        ]) -> Tuple[
        np.ndarray,     # X
        np.ndarray,     # X Mask
        np.ndarray,     # X Mask One-hot
        np.ndarray,     # Y GT
        np.ndarray,     # Y Mask
        np.ndarray      # y Mask One-hot
    ]:
        yLen: int = len(dt[0][-2][0])
        xMax: int = 0;
        entries: int = 0;
        for ele in dt:
            xd, xm, xo, yd, ym, yo = ele;
            assert xd.shape == xm.shape;
            assert yd.shape == ym.shape;
            # print("dp::185", len(xo), len(xd), len(xm), len(yo), len(yd), len(ym))
            assert len(xo) == len(xd) == len(xm) == len(yo) == len(yd) == len(ym);
            assert yLen == len(yd[0]) == len(ym[0]);
            if len(xd[0]) > xMax:
                xMax = len(xd[0]);
            entries += len(xd);

        retXd: np.ndarray = np.zeros((entries, xMax), dtype=float);
        retXm: np.ndarray = np.zeros((entries, xMax), dtype=int);
        retXo: np.ndarray = np.zeros(entries, dtype=int);
        retYd: np.ndarray = np.zeros((entries, yLen), dtype=int);
        retYm: np.ndarray = np.zeros((entries, yLen), dtype=int);
        retYo: np.ndarray = np.zeros(entries, dtype=int);

        __accum: int = 0;
        for xd, xm, xo, yd, ym, yo in dt:
            retXd[__accum:__accum + len(xd)][:len(xd[0])] = xd;
            retXm[__accum:__accum + len(xd)][:len(xd[0])] = xm;
            retXo[__accum:__accum + len(xd)] = xo;
            retYd[__accum:__accum + len(yd)] = yd;
            retYm[__accum:__accum + len(yd)] = ym;
            retYo[__accum:__accum + len(yd)] = yo;
            __accum += len(xd);
        return retXd, retXm, retXo, retYd, retYm, retYo;

    def __service_dtBatch(self, dt: List[
        Tuple[
            np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray
        ]
    ]) -> Tuple[
        np.ndarray,  # X
        np.ndarray,  # X Mask
        np.ndarray,  # X Mask One-hot
        np.ndarray,  # Y GT
        np.ndarray,  # Y Mask
        np.ndarray   # y Mask One-hot
    ]:
        maxShape: List[int] = [0, 0];
        for x, _, _, _, _, _ in dt:
            if len(x) > maxShape[0]:
                maxShape[0] = len(x);
            if len(x[0]) > maxShape[1]:
                maxShape[1] = len(x[0]);
        rxd: np.ndarray = np.zeros((len(dt), maxShape[0], maxShape[1]), dtype=float);
        rxm: np.ndarray = np.zeros((len(dt), maxShape[0], maxShape[1]), dtype=int);
        rxo: np.ndarray = np.zeros((len(dt), maxShape[0]), dtype=int);
        ryd: np.ndarray = np.zeros((len(dt), maxShape[0], self.__pi.getYGtClassLen()), dtype=int);
        rym: np.ndarray = np.zeros((len(dt), maxShape[0], self.__pi.getYGtClassLen()), dtype=int);
        ryo: np.ndarray = np.zeros((len(dt), maxShape[0]), dtype=int);

        for dti in range(len(dt)):
            xd, xm, xo, yd, ym, yo = dt[dti];
            rxd[dti, :len(xd), :len(xd[0])] = xd;
            rxm[dti, :len(xm), :len(xm[0])] = xm;
            rxo[dti, :len(xo)] = xo;
            ryd[dti, :len(yd), :len(xd[0])] = yd;
            rym[dti, :len(ym), :len(xm[0])] = ym;
            ryo[dti, :len(yo)] = yo;
        return rxd, rxm, rxo, ryd, rym, ryo;

    def __getitem__(self, t: Tuple[int, int]) -> Tuple[
        np.ndarray,     # X
        np.ndarray,     # X Mask
        np.ndarray,     # X Mask One-hot
        np.ndarray,     # Y GT
        np.ndarray,     # Y Mask
        np.ndarray      # y Mask One-hot
    ]:
        # print(f"dp::218 {t}")
        idx: int = t[0];
        set: int = t[1];
        tarBatch: np.ndarray;
        if set == 0:
            tarBatch = self.__train;
        elif set == 1:
            tarBatch = self.__vali;
        elif set == 2:
            tarBatch = self.__test;
        else:
            raise NotImplementedError;
        __allIdx: np.ndarray = np.array([i for i in range(len(tarBatch))], dtype=int);
        if idx >= len(__allIdx[tarBatch]):
            raise IndexError;
        ptBatch: List[str] = self.__pi.getBatch(__allIdx[tarBatch].tolist()[idx]);
        allDt: List[Tuple[List[np.ndarray], np.ndarray]] = [];
        for i in range(len(ptBatch)):
            # print(f"dp::154 {i + 1:4d}/{len(ptBatch):4d}")
            p: str = ptBatch[i];
            ptx, pty = self.__pi.fetchCodifiedPt(p, self.__medSeqMap);
            allDt.append((ptx, pty));
        # __allPtMatrix: List[
        #     Tuple[
        #         np.ndarray, np.ndarray, np.ndarray,
        #         np.ndarray, np.ndarray, np.ndarray
        #     ]
        # ] = [self.__service_makePtMatrix(__pm[0], __pm[1]) for __pm in allDt]
        # return self.__service_padDt(__allPtMatrix)
        # ret = self.__service_padDt([self.__service_makePtMatrix(__pm[0], __pm[1]) for __pm in allDt]);
        # ret[0][:, 0] /= self.__maxPtId;
        return self.__service_dtBatch([self.__service_makePtMatrix(__pm[0], __pm[1]) for __pm in allDt]);
        # return ret;

    def getYGtClassLen(self) -> int:
        return self.__pi.getYGtClassLen();


def main() -> int:
    batchSize: int = 512;
    ebd: KGEmbed = KGEmbed(allPt="../data/allPt.pkl", ukb2db="../map/ukb2db.pkl", db2emd="../data/kgEmb2.pkl", icd="E11");
    dp: DataProcessor = DataProcessor(
        pkl="../data/allPt.pkl",
        ebd=ebd,
        medSeqMapUri="../map/ukbMedTokenize.pkl",
        epgPkl=f"../data/E11EmbPtGroup512.pkl",
        icd="E11",
        batchSize=batchSize
    );
    exit(0);
    xd, xm, xo, yd, ym, yo = dp[0, True];
    print(yd.shape)
    return 0;


if __name__ == "__main__":
    main();


