import pickle
from typing import List, Dict;
import os;


def __service_loadUniMed(uri: str = "map/TreatKB_approved_with_Finding.counter") -> List[str]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        allLines: List[str] = f.read().split("\n")[:-1];
    meds: Dict[str, int] = dict();
    for l in allLines:
        meds[l.split("|")[0]] = 0;
    return list(meds.keys());


def __service_loadUniCode(uri: str = "map/TreatKB_approved_with_Finding.counter") -> Dict[str, str]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        allLines: List[str] = f.read().split("\n")[:-1];
    ret: Dict[str, str] = dict();
    for l in allLines:
        name, code, _, _, _ = l.split("|");
        ret[name] = code;
    return ret;


def __service_sttTkbDB(meds: List[str], pkl: str = "drugbank.pkl") -> Dict[str, str]:
    assert os.path.exists(pkl);
    ret: Dict[str, str] = dict();
    with open(pkl, "rb") as f:
        dbDict: Dict[str, str] = pickle.load(f);
    for m in meds:
        try:
            ret[m] = dbDict[m];
        except:
            print(m)
            continue;

    # This section adds manually matched entries
    ret["manganese"] = "DB06757";

    return ret;


def __service_mapTkbMed2Db() -> None:

    tkb: str = "../map/TreatKB_approved_with_Finding.counter";
    ret: List[str] = __service_loadUniMed(tkb);
    conv: Dict[str, str] = __service_sttTkbDB(ret);
    print(f"{len(conv.keys())}/{len(ret)} matched");
    # print(conv)
    name2code: Dict[str, str] = __service_loadUniCode(tkb);
    fi: Dict[str, str] = dict();
    for c in conv.keys():
        fi[name2code[c]] = conv[c];
    # print(fi);
    with open("../data/tkbMedCui2Db.pkl", "wb") as f:
        pickle.dump(fi, f);


if __name__ == "__main__":
    __service_mapTkbMed2Db();
