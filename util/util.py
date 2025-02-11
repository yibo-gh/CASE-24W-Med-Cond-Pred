
import os;
from typing import List, Dict, Tuple;
import pickle;
import subprocess;

import numpy as np

__icdDrugMap: List[Dict[str, List[str]] | None] = [None];


def int2binVec(i: int, l: int = 0) -> np.ndarray:
    '''
    Small-ending encoding
    :param i:   Target value to be converted
    :param l:   Minimum required vector length
    :return:    A small-ending vector embedding with 0 padding

    Example:    int2binVec(23, 0) -> (1, 1, 1, 0, 1)
    Example 2:  int2binVec(14, 6) -> (0, 1, 1, 1, 0, 0)
    '''
    ret: List[int] = [];
    __tmp: int = i;
    while __tmp > 0:
        ret.append(__tmp % 2);
        __tmp -= ret[-1];
        __tmp /= 2;
    retnp: np.ndarray = np.zeros(max(l, len(ret)), dtype=int);
    retnp[:len(ret)] = np.array(ret);
    return retnp;


def tokenizeICD10(icdOri: str, freqMap: Dict[str, float]) -> np.ndarray:
    icd: str = icdOri.replace(".", "");
    base: np.ndarray = np.concatenate((int2binVec(ord(icd[0].upper()) - 55, 6), int2binVec(int(icd[1:3]), 7))).astype(float);
    return np.concatenate((base, np.array([1 if len(icd) == 3 else freqMap[icd]]))).astype(float);


def tokenizeICD10_old(icdOri: str) -> np.ndarray:
    icd: str = icdOri.replace(".", "");
    rem: str = "";
    for i in range(3, len(icd)):
        if icd[i].isnumeric():
            rem += icd[i];
        else:
            rem += f"{ord(icd[i]) - 65:02d}";
    return int(f"{ord(icd[0]) - 55:02d}{int(icd[1:3])}{rem}");


def checkMedUsed(icd: str, med: str) -> bool:
    return True;


def loadIcd2cui(icd: str) -> Dict[str, str]:
    with open(icd, "r") as f:
        xList: List[str] = f.read().split("\n")[:-1];
    retX: Dict[str, str] = dict();
    for x in xList:
        xp: List[str] = x.split("|");
        retX[xp[13]] = xp[0];
    return retX;


def loadDisMedMap(uri: str) -> Dict[str, List[str]]:
    assert os.path.exists(uri);
    ret: Dict[str, List[str]] = dict();
    with open(uri, "r") as f:
        for _, dis, _, med, _ in f.read().split("\n")[:-1]:
            try:
                ret[dis].append(med);
            except:
                ret[dis] = [med];
    return ret;


def loadIcdDrugMap(pkl: str, icdMap: str = "map/ICD/CUI2ICD10.txt", icdMedMap: str = "TreatKB_approved_with_Finding.counter") -> Dict[str, List[str]]:
    if __icdDrugMap[0] is not None:
        return __icdDrugMap[0];
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            __icdDrugMap[0] = pickle.load(f);
    else:
        assert icdMedMap is not None and os.path.exists(icdMap) and os.path.exists(icdMedMap);
        ixm: Dict[str, str] = loadIcd2cui(icdMap);
        mdm: Dict[str, List[str]] = loadDisMedMap(icdMedMap);
        ret: Dict[str, List[str]] = dict();
        for dis in ixm.keys():
            try:
                ret[dis] = mdm[dis];
            except:
                ret[dis] = [];
        with open(pkl, "wb") as f:
            pickle.dump(ret, f);
        __icdDrugMap[0] = ret;
    return __icdDrugMap[0];


def medMatch(icd2cui: Dict[str, str], ukbMed2db: Dict[str, List[str]], tkbDisMedGT: Dict[str, List[str]],
             icd: str, ukbMedCode: str) -> bool:
    try:
        # print(icd2cui[icd])
        gt: List[str] = tkbDisMedGT[icd2cui[icd]];
        # print(gt)
        ukbMed: List[str] | None = ukbMed2db[ukbMedCode];
        # print(ukbMed)
        if ukbMed is None:
            return False;
        for g in gt:
            if ukbMed.__contains__(g):
                return True;
    except KeyError:
        return False;
    return False;


def loadCoreMap(icd2cui: str,
                ukbMed2db: str,
                tkbDisMedGT: str,
                ukbMedTokenize: str) -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int]
]:
    assert os.path.exists(icd2cui) and os.path.exists(ukbMed2db) and os.path.exists(tkbDisMedGT);
    i2c: Dict[str, str];
    um2: Dict[str, List[str]];
    tdm: Dict[str, List[str]];
    umt: Dict[str, int];

    with open(icd2cui, "rb") as f:
        i2c = pickle.load(f);
    with open(ukbMed2db, "rb") as f:
        um2 = pickle.load(f);
    with open(tkbDisMedGT, "rb") as f:
        tdm = pickle.load(f);
    with open(ukbMedTokenize, "rb") as f:
        umt = pickle.load(f);
    return i2c, um2, tdm, umt;


def getColNameByFieldID(id: int) -> str:
    return f"participant.p{id:d}";


def __test() -> None:
    print(tokenizeICD10("Z54.W2B"));
    i2c, um2, tdb = loadCoreMap(
        "../map/icd2cui.pkl",
        "../map/ukb2db.pkl",
        "../map/tkbDisMedGT.pkl"
    );
    print(medMatch(i2c, um2, tdb, "G20", "1141169700"))
    return;


def execCmd(cmd: str) -> Tuple[str, str | None]:
    out, err = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate();
    return out.decode(), None if err is None else err.decode();


if __name__ == "__main__":
    print(int2binVec(14, 6));
    # with open("../map/ukbIcdFreq.pkl", "rb") as f:
    #     print(tokenizeICD10("I20.9", pickle.load(f)));

