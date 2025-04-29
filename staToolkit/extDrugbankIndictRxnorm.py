
import requests;
from typing import List, Dict, Tuple;
import json;
import os;
import pickle;


def __service_getWeb(uri: str) -> str:
    return requests.get(uri).text;


def __service_getRxFromDB(dbid:str) -> List[str]:
    try:
        return json.loads(__service_getWeb(f"https://rxnav.nlm.nih.gov/REST/rxcui.json?idtype=drugbank&id={dbid}"))["idGroup"]["rxnormId"];
    except:
        return [];


def __service_getApprovedDisByRcui(rcui: str) -> List[str]:
    try:
        res: str = __service_getWeb(
            f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rcui}&rela=may_treat");
        cgList: list = json.loads(res)["rxclassDrugInfoList"]["rxclassDrugInfo"]

        ret: List[str] = [];
        for _cg in cgList:
            if _cg["rela"] == "may_treat":
                ret.append(_cg["rxclassMinConceptItem"]["classId"])
        return ret;
    except:
        return [];


def __service_getCuiFromMsh(msh: str) -> str:
    return "";


def __service_loadMsh(conso: str) -> Dict[str, str]:
    assert os.path.exists(conso);
    with open(conso, "r") as f:
        lines: List[str] = f.read().split("\n");
        if lines[-1] == "":
            lines = lines[:-1];

    ret: Dict[str, str] = dict();
    for _l in lines:
        _ls: List[str] = _l.split("|");
        try:
            ret[_ls[-6]];
        except:
            ret[_ls[-6]] = _ls[0];

    return ret;


def __service_getDisCuiFromDB(m2c: Dict[str, str], dbid: str) -> List[str]:
    ret: List[str] = [];
    for _cui in __service_getRxFromDB(dbid):
        for _msh in __service_getApprovedDisByRcui(_cui):
            ret.append(m2c[_msh]);
    return ret;


def __service_getIcdFromCui(c2i: Dict[str, List[str]], cui: List[str]) -> List[str]:
    print(cui)
    ret: List[str] = [];
    for _cui in cui:
        try:
            ret += c2i[_cui];
        except:
            continue;
    return ret;


def __serivce_getUkbUniDB(d: Dict[str, List[str] | None]) -> Dict[str, int]:
    ret: Dict[str, int] = dict();
    for k in d.keys():
        if d[k] is None:
            continue;
        for id in d[k]:
            ret[id] = 0;
    return ret;


def __service_fetchPerItemICD(uri: str = "data/drugbank.xml", tarID: Dict[str, int] = dict(), pref: str = "{http://www.drugbank.ca}", outPref: str = "./DbICD/") -> List[str]:
    assert os.path.exists(uri);
    from xml.etree import ElementTree as ET;
    tree: ET = ET.parse(uri);
    ret: List[str] = [];
    if not os.path.exists(outPref):
        os.mkdir(outPref);
    allChild = tree.getroot().findall(f"{pref}drug");
    for c in allChild:
        id: str = [n.text for n in c.findall(f"{pref}drugbank-id")][0];
        try:
            tarID[id];
            ret.append(id);
        except KeyError:
            continue;
    return ret;


def main() -> int:
    # print(__service_getRxFromDB("db00030"));
    # print(__service_getApprovedDisByRcui("253182"));
    _m2c: Dict[str, str] = __service_loadMsh("../map/msh.out");
    with open("../map/cui2icd.pkl", "rb") as f:
        _c2i: Dict[str, List[str]] = pickle.load(f);
    # print(__service_getIcdFromCui(_c2i, __service_getDisCuiFromDB(_m2c, "db00030")));

    with open("../map/ukb2db.umls.pkl", "rb") as f:
        uniDB: Dict[str, List[str] | None] = pickle.load(f);

    dbList: List[str] = __service_fetchPerItemICD(uri="../data/drugbank.xml", tarID=__serivce_getUkbUniDB(uniDB));

    _d: Dict[str, List[str]] = dict();
    for _db in dbList:
        print(_db, end=" ")
        _d[_db] = __service_getIcdFromCui(_c2i, __service_getDisCuiFromDB(_m2c, _db));

    print(_d);
    with open("../map/icdPerDB.pkl", "wb") as f:
        pickle.dump(_d, f);

    return 0;


def main_sim() -> int:
    with open("../map/icdPerDB.pkl", "rb") as f:
        d: Dict[str, List[str]] = pickle.load(f);
    for k in d.keys():
        d[k] = list(set(d[k]));
    print(d)
    with open("../map/icdPerDB.pkl", "wb") as f:
        pickle.dump(d, f);
    return 0;


if __name__ == "__main__":
    main_sim();
