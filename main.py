
import sys
from typing import List, Tuple, Dict;

import pickle;
import numpy as np;

sys.path.append("ukbUtil");
from ukbUtil import fetchDB, loadMeta, readFile, UKB;

from util.util import *;
from obj.pt import *;


def __service_getNotNullDates(l: List[str]) -> List[str]:
    ret: List[str] = [];
    for ele in l:
        if ele != "":
            ret.append(ele);
    return ret;


def __service_simpEth(l: List[str]) -> str:
    if len(l) == 1:
        return l[0];
    if len(np.unique((np.array(l)))) == 1:
        return l[0];
    cpy = l.copy();
    try:
        while True:
            cpy.remove("-1");
    except:
        pass;
    try:
        while True:
            cpy.remove("-3");
    except:
        pass;
    if len(cpy) > 1:
        try:
            while True:
                cpy.remove("6");
        except:
            pass;
    cpy = np.unique(np.array(cpy)).tolist();
    if cpy == []:
        return "-1";
    if len(cpy) == 1:
        return cpy[0];
    if len(cpy) == 2:
        c1, c2 = cpy;
        if len(c1) == 1 and c2.startswith(c1):
            return c2;
        if len(c2) == 1 and c1.startswith(c2):
            return c1;
    superclass: List[str] = [];
    for c in cpy:
        superclass.append(c[0]);
    if len(np.unique(np.array(superclass))) == 1:
        return superclass[0];
    return "2";


def __service_makePtFilter(dt: UKB, ptd: dict[str, bool]) -> np.ndarray:
    return np.array([ptd[pt[0]] for pt in dt["Participant ID"]]);


def __serviceF_filterByDrugMatch(allPt: Dict[str, Pt], dt: UKB) -> np.ndarray:
    hasMatchDrug: Dict[str, bool] = dict();
    for pt in allPt.keys():
        hasMatchDrug[pt] = sum([len(ml) for ml in allPt[pt].vectorize()[1]]) > 0;
    return __service_makePtFilter(dt, hasMatchDrug)


def __serviceF_filterByICDwValidMed(allPt: Dict[str, Pt], dt: UKB, icd: str) -> np.ndarray:
    uniPtId: Dict[str, bool] = dict();
    for p in allPt.keys():
        pt: Pt = allPt[p];
        uniPtId[pt.id] = False;
        for evt in pt.evtList:
            if evt.type != EvtClass.Dig or (evt.cont[0].lower() != icd.lower() and evt.cont[0][:3].lower() != icd[:3].lower()) or len(evt.assoMed) == 0:
                continue;
            uniPtId[pt.id] = True;
            break;
    return __service_makePtFilter(dt, uniPtId);


def __service_getTrainingDt(
        allPt: Dict[str, Pt],
        dt: UKB, filter: np.ndarray,
        umt: Dict[str, int]
) -> Tuple[List[np.ndarray], List[List[str]]]:
    tarPtID: np.ndarray = dt.dt[1:, 0][filter];
    X: List[np.ndarray] = [];
    y: List[List[str]] = [];
    for tpi in tarPtID:
        pt: Pt = allPt[tpi];
        x, gt = pt.vectorize(tarICD="I20");
        print(x[-1], gt[-1])
        X.append(x);
        y.append(gt);
    assert len(X) == len(y);
    for yi in y:
        for j in range(len(yi)):
            yiEvt: np.ndarray = yi[j];
            for i in range(len(yiEvt)):
                yiEvt[i] = umt[yiEvt[i]];
            yi[j] = yiEvt.astype(int);
    return X, y;


def __service_dtFlatten(X: List[np.ndarray], y: List[List[np.ndarray]]) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    retX: np.ndarray = X[0];
    for i in range(1, len(X)):
        retX = np.concatenate((retX, X[i]));

    retYL: List[np.ndarray] = [];
    for ptY in y:
        for entry in ptY:
            retYL.append(entry);
    maxLen: int = max([len(x) for x in retYL]);
    retY: np.ndarray = np.zeros((len(retYL), maxLen), dtype=int);
    retYMask: np.ndarray = np.zeros((len(retYL), maxLen), dtype=int);
    for yi in range(len(retYL)):
        retYMask[yi][:len(retYL[yi])] = 1;
        retY[yi][:len(retYL[yi])] = retYL[yi];
    return retX, retY, retYMask;


def __service_loadDt(tarICD: str,
                            ukbPickle: str | None = None,
                            dsID: str = "record-GxF1x2QJbyfKGbP5yY9JyVZ1",
                            tarCol: List[str] = ["ICD", "Medication", "date"],
                            extTarCode: List[int] = [31, 34, 52, 21000],
                            beeline: str = "bin/spark-3.2.3-bin-hadoop2.7/bin/beeline",
                            i2cUri: str = "map/icd2cui.pkl",
                            um2Uri: str = "map/ukb2db.pkl",
                            tdbUri: str = "map/drugbankIcdPerDis.pkl",
                            umtUri: str = "map/ukbMedTokenize.pkl") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt: UKB;

    if ukbPickle is not None and os.path.exists(ukbPickle):
        with open(ukbPickle, "rb") as f:
            dt = pickle.load(f);
    else:
        token: str = "MVFsNanigw5fKGJUSbXmMPHSUa7EHR5i";
        _, _, dt, _ = fetchDB(dsID=dsID,
                              keys=tarCol,
                              db=f"{token}__project-GxBzxGQJqvQ8XgpxffxXQf13",
                              beeline=beeline,
                              colCode=extTarCode);

    i2c, um2, tdb, umt = loadCoreMap(i2cUri, um2Uri, tdbUri, umtUri);
    tdbNew: Dict[str, List[str]] = dict();
    for k in tdb.keys():
        tdbNew[i2c[k]] = tdb[k];
    tdb = tdbNew;
    # print(tdb);
    # exit(0)

    # for k in list(dt.meta.keys()):
    #     print(k, dt.meta[k])
    ptMeds: np.ndarray = dt["Treatment/medication code"];

    allDig: np.ndarray = np.concatenate(
        (
            dt["Participant ID"],
            dt[getColNameByFieldID(31)][:, np.newaxis],
            dt[getColNameByFieldID(34)][:, np.newaxis],
            dt[getColNameByFieldID(52)][:, np.newaxis],
            dt["Ethnic background"],
            dt["Diagnoses - main ICD10"],
            dt["Date of first in-patient diagnosis - main ICD10"]
        ), axis=1);
    # print(allDig.shape)
    # print(allDig)

    allPt: Dict[str, Pt] = dict();

    validIcdDict: Dict[str, int] = dict();
    for i in range(len(allDig)):
        pt: Pt = allDig[i];
        id: str = pt[0];
        sex: int = int(pt[1]);
        yob: int = int(pt[2]);
        mob: int = int(pt[3]);
        eth: str = __service_simpEth(__service_getNotNullDates(pt[4:8]));
        dig: List[str] = pt[8];
        dates: List[str] = __service_getNotNullDates(pt[9:]);
        meds: List[str] = __service_getNotNullDates(ptMeds[i].tolist());

        assert len(dig) == len(dates);
        allPt[id] = Pt(id, PtDemo(SexAtBirth(sex), mob, yob, int(eth)));
        for i in range(len(dig)):
            __dig, __date = dig[i], dates[i];
            ptmList: List[str] = [];
            for m in meds:
                if medMatch(i2c, um2, tdb, __dig, m):
                    ptmList.append(m);
            if len(ptmList) > 0:
                # if __dig[:3] == "I25":
                #     print(id, __dig, ptmList)
                try:
                    validIcdDict[__dig[:3]] += 1;
                except:
                    validIcdDict[__dig[:3]] = 1;
            allPt[id].newEvt(__date, EvtClass.Dig, [__dig], ptmList);
        # for evt in allPt[id].evtList:
        #     print(evt.time, end=" ");
        # print(allPt[id].vectorize())
    validIcdList: List[Tuple[str, int]] = [(k, validIcdDict[k]) for k in validIcdDict.keys()];
    validIcdList.sort(key=lambda x: x[0]);
    validIcdList.sort(key=lambda x: x[1], reverse=True);
    # for v in validIcdList:
    #     print(v);
    #     break;

    # for pt in allPt.keys():
    #     print(allPt[pt].id, sum([len(ml) for ml in allPt[pt].vectorize()[1]]));
    ptFilter: np.ndarray = __serviceF_filterByDrugMatch(allPt, dt);
    # print("m::170", np.sum(ptFilter))
    icdFilter: np.ndarray = __serviceF_filterByICDwValidMed(allPt, dt, tarICD);
    print("m::174", np.sum(ptFilter & icdFilter));
    X, y = __service_getTrainingDt(allPt, dt, ptFilter & icdFilter, umt);
    return __service_dtFlatten(X, y)


def main() -> int:
    X: np.ndarray; y: np.ndarray; mask: np.ndarray;
    X, y, mask = __service_loadDt(tarICD="i20", ukbPickle=f"data/1737145582028.pkl");
    print(X.shape, y.shape, mask.shape)
    return 0;


if __name__ == "__main__":
    main();

