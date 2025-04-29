
from typing import List, Dict, Tuple;
import pickle;
import os;


def __service_parseICD(uri: str, d: Dict[str, List[str]] = dict(), rvs: Dict[str, List[str]] = dict()) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        icdList: List[str] = f.read().split("\n")[:-1];
    for icd in icdList:
        iParse: List[str] = icd.split("|");
        cui: str = iParse[0];
        iCode: str = iParse[10];
        iCodeMod: str = iCode.replace(".", "");
        try:
            d[iCodeMod].append(cui);
            d[iCodeMod] = list(set(d[iCodeMod]));
        except KeyError:
            d[iCodeMod] = [cui];

        try:
            rvs[cui].append(iCodeMod);
            rvs[cui] = list(set(rvs[cui]));
        except KeyError:
            rvs[cui] = [iCodeMod];

    return d, rvs;


def procIcd2Cui(out: str = "icd2cui.pkl", rvs: str = "cui2icd.pkl") -> None:
    d, r = __service_parseICD("../map/ICD/CUI2ICD10.txt")
    _i2c, _c2i = __service_parseICD("../map/ICD/CUI2ICD9.txt", d, r);

    with open(out, "wb") as f:
        pickle.dump(_i2c, f);
    with open(rvs, "wb") as f:
        pickle.dump(_c2i, f);
    return;


if __name__ == "__main__":
    procIcd2Cui("../map/icd2cui.pkl", "../map/cui2icd.pkl");

