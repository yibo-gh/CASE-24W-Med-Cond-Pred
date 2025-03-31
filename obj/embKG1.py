
from typing import Tuple, Dict, List;
import os;

import numpy as np;
import pickle;

import sys;
sys.path.append("..");

from obj.embedder import Embedder;
from obj.pt import Pt;
from obj.pt import EvtClass;


class KGEmbed(Embedder):

    __lineSize: Dict[str, int];
    __ukb2EmbMap: Dict[str, List[np.ndarray]];
    __ukbDbMap: Dict[str, List[str] | None];
    __icd2emb: Dict[str, np.ndarray];
    __db2emb: Dict[str, np.ndarray];
    __allPt: Dict[str, Pt] | None;
    __tarYCodeMap: Dict[str, List[int]];
    __validYGtLen: int;
    __icd: str;

    def __init__(self, ukb2db: str, db2emd: str, allPt: str, icd: str, ukbMed: str = "map/ukbMed.map") -> None:
        super().__init__(ukb2db=ukb2db, db2emd=db2emd, allPt=allPt);
        self.__icd = icd;
        with open(allPt, "rb") as f:
            self.__allPt = pickle.load(f);
        self.__tarYCodeMap = dict();
        with open(ukb2db, "rb") as f:
            self.__ukbDbMap: Dict[str, List[str] | None] = self.ukb2dbMappingCorrection(pickle.load(f));
        with open(db2emd, "rb") as f:
            self.__icd2emb, self.__db2emb = pickle.load(f);
        self.__lineSize = dict();
        self.__ukb2EmbMap = dict();
        self.__service_buildTarYMap(self.__allPt, ukbMed=ukbMed);

    def __service_buildTarYMap(self, allPt: Dict[str, Pt], ukbMed: str) -> None:
        __tmpMedMap: Dict[str, int] = dict();
        for p in list(allPt.keys()):
            pt: Pt = allPt[p];
            for e in pt.evtList:
                if e.type != EvtClass.Dig or e.cont[0][:3].lower() != self.__icd[:3].lower():
                    continue;
                for m in e.assoMed:
                    # print(f"ekg::47 {p} {m}")
                    __tmpMedMap[m] = 0;
        __medList: List[str] = list(__tmpMedMap.keys());
        # for i in range(len(__medList)):
            # print(f"ekg::52 {i} {__medList[i]}")
            # self.__tarYCodeMap[__medList[i]] = i;

        __allDBid: List[str] = []
        for am in __medList:
            __allDBid += self.__ukbDbMap[am];
        __allDBid = list(set(__allDBid));
        __allDBidIdx: Dict[str, int] = dict();
        for i in range(len(__allDBid)):
            __allDBidIdx[__allDBid[i]] = i;
        # print(__allDBidIdx)

        for ml in __medList:
            self.__tarYCodeMap[ml] = [__allDBidIdx[__uid] for __uid in self.__ukbDbMap[ml]];
        self.__validYGtLen = len(__allDBidIdx);
        return;

    def dtBatching(self, icd: str) -> Tuple[Dict[str, int], Dict[str, List[np.ndarray]]]:
        for p in list(self.__allPt.keys()):
            pt: Pt = self.__allPt[p];
            self.__lineSize[p] = 1;
            for evt in pt.evtList:
                if evt.type == EvtClass.Dig:
                    self.__lineSize[p] += 1;
                    # print(f"dp::33 {self.__lineSize[p]}")
                    if evt.cont[0][:3].lower() != icd.lower():
                        for am in evt.assoMed:
                            try:
                                self.__lineSize[p] += len(self.__ukb2EmbMap[am]);
                                # print(f"dp::38 {am} {self.__ukb2EmbMap[am]}");
                            except:
                                self.__ukb2EmbMap[am] = [];
                                db: List[str] | None = self.__ukbDbMap[am];
                                if db is None:
                                    # print(f"dp::42 {am} no drugbank code found");
                                    continue;
                                for dbc in db:
                                    try:
                                        self.__ukb2EmbMap[am].append(self.__db2emb[dbc]);
                                        # print(f"dp::47 {dbc} for ukb {am} embedding found")
                                    except:
                                        # print(f"dp::49 {dbc} no embedding found")
                                        pass;
                                self.__lineSize[p] += len(self.__ukb2EmbMap[am]);

                        # print(f"dp::50 {self.__lineSize[p]}")
                    else:
                        break;
            # print(f"dp::28 {p} {self.__lineSize[p]} lines");
        return self.__lineSize, self.__ukb2EmbMap;

    def getPtVec(self, id: str, medSeqMap: Dict[str, int]) -> Tuple[
        int, np.ndarray, List[np.ndarray], np.ndarray
        # PID, dem, X, yGT
    ]:
        pt: Pt = self.__allPt[id];
        retX: List[np.ndarray] = [];
        totalMeds: int = len(list(medSeqMap.keys()));
        for e in pt.evtList:
            if e.type == EvtClass.Med:
                retX.append(
                    np.concatenate(
                        (np.array([e.type.value], dtype=float),
                         np.array([medSeqMap[m] / totalMeds for m in e.cont], dtype=float)
                         )
                    )
                );
            elif e.type == EvtClass.Dig:
                __icdEmb: np.ndarray;
                try:
                    __icdEmb = self.__icd2emb[e.cont[0]];
                except:
                    try:
                        __icdEmb = self.__icd2emb[e.cont[0][:3]];
                    except:
                        continue;
                retX.append(
                    np.concatenate(
                        (np.array([e.type.value], dtype=float),
                         __icdEmb)
                    )
                );
                if e.cont[0][:3].lower() == self.__icd[:3].lower():
                    retY: np.ndarray = np.zeros(self.__validYGtLen, dtype=int);
                    for m in e.assoMed:
                        for __cm in self.__tarYCodeMap[m]:
                            retY[__cm] = 1;
                    # print("ekg::127", int(pt.id), pt.dem.vectorize(), retX, retY)
                    return int(pt.id), pt.dem.vectorize(), retX, retY;
                for m in e.assoMed:
                    try:
                        for ukbEmb in self.__ukb2EmbMap[m]:
                            retX.append(
                                np.concatenate(
                                    (np.array([EvtClass.Med.value], dtype=float),
                                     ukbEmb
                                     )
                                )
                            );
                    except:
                        pass

    def getYGtClassLen(self) -> int:
        return self.__validYGtLen;


def main() -> int:
    KGEmbed("../map/ukb2db.pkl", "../data/kgEmb.pkl");
    return 0;

