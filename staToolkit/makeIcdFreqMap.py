
import sys;
from typing import Dict, List;
import pickle;

import numpy as np;

sys.path.append("../ukbUtil");
from ukbUtil import UKB;


def __service_parseICD(icd: np.ndarray, d: Dict[str, List[str]]) -> None:
    for line in icd:
        if line[0] == "":
            continue;
        for c in line[0]:
            try:
                d[c[:3]].append(c);
            except:
                d[c[:3]] = [c];
    return;


def __service_makeFamilyStt(ukb: UKB) -> Dict[str, float]:
    icd10: np.ndarray = ukb["Diagnoses - ICD10"];
    icd9: np.ndarray = ukb["Diagnoses - ICD9"];
    ret: Dict[str, List[str]] = dict();
    __service_parseICD(icd10, ret);
    __service_parseICD(icd9, ret);

    ret2: Dict[str, float] = dict();
    epsilon: float = 1e-2;
    for k in ret.keys():
        icdList: List[str] = ret[k];
        if len(icdList) == 1:
            ret2[icdList[0]] = 0;
            continue;
        for i in range(len(icdList)):
            __icd: str = icdList[i];
            ret2[__icd] = epsilon + (1 - 2 * epsilon) * i / (len(icdList) - 1);

    print(ret2);
    return ret2;


def main() -> int:
    with open("../data/1737145582028.pkl", "rb") as f:
        ukb: UKB = pickle.load(f);
    with open("../map/ukbIcdFreq.pkl", "wb") as f:
        pickle.dump(__service_makeFamilyStt(ukb), f);
    return 0;


if __name__ == "__main__":
    main();

