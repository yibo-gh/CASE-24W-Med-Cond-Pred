
import os;
from typing import List, Dict;
import pickle;


__icdDrugMap: List[Dict[str, List[str]] | None] = [None];


def tokenizeICD10(icdOri: str) -> int:
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


def __test() -> None:
    print(tokenizeICD10("Z54.W2B"));
    return;


if __name__ == "__main__":
    __test();

