import os.path
import sys
from typing import List, Tuple, Dict, Callable;

import pickle;
import numpy as np;

sys.path.append("ukbUtil");
from ukbUtil import fetchDB, loadMeta, readFile, UKB;

from util.util import *;
from obj.pt import *;
from models.transfomer import PtDS;


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
        hasMatchDrug[pt] = False;
        for et in allPt[pt].evtList:
            # print("m::74", pt, et.time, et.type, et.cont, et.assoMed);
            if et.type == EvtClass.Dig and et.cont[0][:4] == "E11":
                print(et.assoMed);
            if (et.type == EvtClass.Dig and len(et.assoMed) > 0) or (et.type == EvtClass.Med and not et.time == "1970-01-01" and len(et.cont) > 0):
                hasMatchDrug[pt] = True;
                break;
        # hasMatchDrug[pt] = sum([len(ml.assoMed) for ml in allPt[pt].evtList]) > 0;
    # print(hasMatchDrug)
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
        dt: UKB,
        filter: np.ndarray,
        umt: Dict[str, int],
        medMap: Dict[str, int] | None = None,
        freq: Dict[str, float] | None = None
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[List[np.ndarray | None]]
]:
    tarPtID: np.ndarray = dt.dt[1:, 0][filter];
    X: List[np.ndarray] = [];
    xMask: List[np.ndarray] = [];
    y: List[List[np.ndarray | None]] = [];
    for tpi in tarPtID:
        pt: Pt = allPt[tpi];
        x, xMaskLocal, gt = pt.vectorize(tarICD="E11", medMap=medMap, freq=freq);
        # print(x[-1], gt[-1])
        X.append(x);
        xMask.append(xMaskLocal);
        y.append(gt);
    assert len(X) == len(y) == len(xMask);
    for yi in y:
        for j in range(len(yi)):
            yiEvt: np.ndarray | None = yi[j];
            if yiEvt is None:
                yi[j] = np.array([]);
                continue;
            for i in range(len(yiEvt)):
                yiEvt[i] = umt[yiEvt[i]];
            yi[j] = yiEvt.astype(int);
    return X, xMask, y;


def __service_padArr(a: np.ndarray, tarSize: int, axis: int) -> np.ndarray:
    # print("m::118", tarSize, a.shape, axis)
    assert tarSize >= a.shape[axis];
    retShape: List[int] = list(a.shape);
    retShape[axis] = tarSize;
    ret: np.ndarray = np.zeros(tuple(retShape), dtype=a.dtype);
    ret[:a.shape[0], :a.shape[1]] = a;
    return ret;


def __service_padNCat(tarSize: int, a: np.ndarray, b: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.concatenate((__service_padArr(a, tarSize, axis + 1), __service_padArr(b, tarSize, axis + 1)), axis=axis);


def __service_dtFlatten(X: List[np.ndarray], xMaskIpt: List[np.ndarray], y: List[List[np.ndarray]]) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    maxCol: int = 0;
    for x in X:
        if len(x) == 0:
            continue;
        thisCol: int = int(x.shape[1]);
        if thisCol > maxCol:
            maxCol = thisCol;
        # print(x.shape, thisCol, maxCol)
    print("m::152", len(X))
    retX: np.ndarray = __service_padArr(X[0], maxCol, 1);
    xMask: np.ndarray = __service_padArr(xMaskIpt[0], maxCol, axis=1);
    for i in range(1, len(X)):
        retX = np.concatenate((retX, __service_padArr(X[i], maxCol, 1)));
        xMask = np.concatenate((xMask, __service_padArr(xMaskIpt[i], maxCol, 1)));
        assert retX.shape == xMask.shape;

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
    return retX, xMask, retY, retYMask;


def __service_dbParseControl(dt: np.ndarray, dtype, accum: int = 0, tarLen: int = 1, dtProc: Callable | List[Callable] | None = None) -> Tuple[int, object]:
    ret: object = dtype(dt[accum : accum + tarLen]) if tarLen > 1 else dtype(dt[accum]);
    if dtProc is not None:
        if not isinstance(dtProc, list):
            ret = dtProc(ret);
        else:
            for fn in dtProc:
                ret = fn(ret);
    return accum + tarLen, ret;


def __service_loadDt(tarICD: str,
                            ukbPickle: str | None = None,
                            dsID: str = "record-GxF1x2QJbyfKGbP5yY9JyVZ1",
                            tarCol: List[str] = ["ICD", "Medication", "date"],
                            extTarCode: List[int] = [31, 34, 52, 21000],
                            beeline: str = "bin/spark-3.2.3-bin-hadoop2.7/bin/beeline",
                            i2cUri: str = "map/icd2cui.pkl",
                            um2Uri: str = "map/ukb2db.pkl",
                            tdbUri: str = "map/drugbankIcdPerDis.pkl",
                            umtUri: str = "map/ukbMedTokenize.pkl") -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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
    freqMap: Dict[str, float];
    with open("map/ukbIcdFreq.pkl", "rb") as f:
        freqMap = pickle.load(f);
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
            # dt["Diagnoses - main ICD10"],
            # dt["Date of first in-patient diagnosis - main ICD10"]
            dt["Diagnoses - ICD10"],  # All ICD 10
            dt["Date of first in-patient diagnosis - ICD10"],  # Date of all ICD 10
            dt["Diagnoses - ICD9"],  # All ICD 9
            dt["Date of first in-patient diagnosis - ICD9"]   # Date of all ICD 9
        ), axis=1);
    # print(allDig.shape)
    # print(allDig)
    icd10dtLen: int = len(dt.meta["Date of first in-patient diagnosis - ICD10"]);
    icd9dtLen: int = len(dt.meta["Date of first in-patient diagnosis - ICD9"]);

    allPt: Dict[str, Pt] = dict();

    validIcdDict: Dict[str, int] = dict();
    for i in range(len(allDig)):
        pt: np.ndarray = allDig[i];
        __accumColNum: int; id: str;
        __accumColNum, id = __service_dbParseControl(pt, dtype=str);
        sex: int;
        __accumColNum, sex = __service_dbParseControl(pt, dtype=int, accum=__accumColNum);
        yob: int;
        __accumColNum, yob = __service_dbParseControl(pt, dtype=int, accum=__accumColNum);
        mob: int;
        __accumColNum, mob = __service_dbParseControl(pt, dtype=int, accum=__accumColNum);
        ethList: str;
        __accumColNum, eth = __service_dbParseControl(pt, dtype=list, accum=__accumColNum, tarLen=4, dtProc=[__service_getNotNullDates, __service_simpEth]);
        icdx: List[str];
        __accumColNum, icdx = __service_dbParseControl(pt, dtype=list, accum=__accumColNum, dtProc=__service_getNotNullDates);
        icdxd: List[str];
        __accumColNum, icdxd = __service_dbParseControl(pt, dtype=list, accum=__accumColNum, tarLen=icd10dtLen, dtProc=__service_getNotNullDates);
        icd9: List[str];
        __accumColNum, icd9 = __service_dbParseControl(pt, dtype=list, accum=__accumColNum, dtProc=__service_getNotNullDates);
        icd9d: List[str];
        __accumColNum, icd9d = __service_dbParseControl(pt, dtype=list, accum=__accumColNum, tarLen=icd9dtLen, dtProc=__service_getNotNullDates);
        meds: List[str] = __service_getNotNullDates(ptMeds[i].tolist());

        assert len(icdx) == len(icdxd) and len(icd9) == len(icd9d);

        allPt[id] = Pt(id, PtDemo(SexAtBirth(sex), mob, yob, int(eth)));
        dig: List[str] = icd9 + icdx;
        dates: List[str] = icd9d+ icdxd;
        __ptMed: dict[str, int] = dict();
        for m in meds:
            __ptMed[m] = 0;
        for j in range(len(dig)):
            __dig, __date = dig[j], dates[j];
            ptmList: List[str] = [];
            for m in meds:
                if medMatch(i2c, um2, tdb, __dig, m):
                    ptmList.append(m);
                    __ptMed[m] = 1;
            # if len(ptmList) > 0:
            if True:
                # if __dig[:3] == "I25":
                #     print(id, __dig, ptmList)
                try:
                    validIcdDict[__dig[:3]] += 1;
                except:
                    validIcdDict[__dig[:3]] = 1;
            if __dig[:len(tarICD)].lower() == tarICD:
                # print("m::292")
                allPt[id].newEvt(__date, EvtClass.Dig, [__dig], ptmList);
            else:
                # print("m::295")
                allPt[id].newEvt(__date, EvtClass.Dig, [__dig], []);
                allPt[id].newEvt(__date, EvtClass.Med, ptmList, []);
        # for evt in allPt[id].evtList:
        #     print(evt.time, end=" ");
        # print(allPt[id].vectorize())
        noUseMed: List[str] = [];
        for m in __ptMed.keys():
            if __ptMed[m] == 0:
                noUseMed.append(m);
        allPt[id].newEvt("1970-01-01", EvtClass.Med, noUseMed, []);
    validIcdList: List[Tuple[str, int]] = [(k, validIcdDict[k]) for k in validIcdDict.keys()];
    validIcdList.sort(key=lambda x: x[0]);
    validIcdList.sort(key=lambda x: x[1], reverse=True);
    for v in validIcdList:
        if v[0] == "E11":
            print(v);
    #     break;

    # for pt in allPt.keys():
    #     print(allPt[pt].id, sum([len(ml) for ml in allPt[pt].vectorize()[1]]));ve
    ptFilter: np.ndarray = __serviceF_filterByDrugMatch(allPt, dt);
    # print("m::170", np.sum(ptFilter))
    icdFilter: np.ndarray = __serviceF_filterByICDwValidMed(allPt, dt, tarICD);
    # print("m::174", np.sum(ptFilter & icdFilter));
    X, xMask, y = __service_getTrainingDt(allPt, dt, ptFilter & icdFilter, umt, medMap=umt, freq=freqMap);
    return __service_dtFlatten(X, xMask, y);


def __service_loadDtByICD(icd: str, ukbPkl: str) -> PtDS:
    tarF: str = f"data/{icd}.pkl";
    if os.path.exists(tarF):
        with open(tarF, "rb") as f:
            return pickle.load(f);
    X: np.ndarray;
    xMask: np.ndarray;
    y: np.ndarray;
    mask: np.ndarray;
    X, xMask, y, mask = __service_loadDt(tarICD=icd, ukbPickle=ukbPkl);
    ret: PtDS = PtDS(X, xMask, y, mask);
    # with open(tarF, "wb") as f:
    #     pickle.dump(ret, f);
    return ret;


def main() -> int:
    ptds: PtDS = __service_loadDtByICD("e11", f"data/1737145582028.pkl");
    print(ptds.x.shape, ptds.xm.shape, ptds.y.shape, ptds.ym.shape);
    print(ptds.x[:4])
    print(ptds.xm[:4])
    print(ptds.y[:4])
    print(ptds.ym[:4])
    return 0;


if __name__ == "__main__":
    main();

