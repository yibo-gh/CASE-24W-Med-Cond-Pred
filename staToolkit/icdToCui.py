
from typing import List, Dict, Tuple;
import pickle;
import os;


def __service_parseICD(uri: str, d: Dict[str, str] = dict()) -> Dict[str, str]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        icdList: List[str] = f.read().split("\n")[:-1];
    for icd in icdList:
        iParse: List[str] = icd.split("|");
        cui: str = iParse[0];
        iCode: str = iParse[10];
        d[iCode] = cui;
    return d;


def procIcd2Cui(out: str = "icd2cui.pkl") -> None:
    with open(out, "wb") as f:
        pickle.dump(
            __service_parseICD("../map/ICD/CUI2ICD9.txt",
                               __service_parseICD("../map/ICD/CUI2ICD10.txt")
                               ),
            f);
    return;


if __name__ == "__main__":
    procIcd2Cui("../map/icd2cui.pkl");

