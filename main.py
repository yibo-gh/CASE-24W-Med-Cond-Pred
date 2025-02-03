
import sys
from typing import List, Tuple, Dict;

import pickle;
import numpy as np;

sys.path.append("ukbUtil");
from ukbUtil import fetchDB, loadMeta, readFile;

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


def main() -> int:
    token: str = "MVFsNanigw5fKGJUSbXmMPHSUa7EHR5i";
    '''pref, _, dt, _ = fetchDB(dsID="record-GxF1x2QJbyfKGbP5yY9JyVZ1",
                             keys=["ICD", "Medication", "date"],
                             db=f"{token}__project-GxBzxGQJqvQ8XgpxffxXQf13",
                             beeline="bin/spark-3.2.3-bin-hadoop2.7/bin/beeline",
                             colCode=[31, 34, 52, 21000]);'''

    with open(f"data/1737145582028.pkl", "rb") as f:
        dt = pickle.load(f);

    i2c, um2, tdb = loadCoreMap(
        "map/icd2cui.pkl",
        "map/ukb2db.pkl",
        "map/tkbDisMedGT.pkl"
    );

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
    for v in validIcdList:
        print(v)
    return 0;


if __name__ == "__main__":
    main();

