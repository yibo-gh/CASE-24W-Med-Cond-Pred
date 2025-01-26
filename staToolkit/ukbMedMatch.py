
import os
import pickle
import sys
from typing import List, Dict, Tuple;
import json;


def __service_loadUkbMlpJson(uri: str = "ukbOut.rcd") -> Dict[str, List[json.JSONDecoder] | None]:
    with open(uri, "r") as f:
        rcd: List[str] = f.read().split("\n")[:-1];
    medDict: Dict[str, List[json.JSONDecoder] | None] = dict();
    lastMed: str = "";
    for r in rcd:
        if r[0] != "{":
            lastMed = r;
            medDict[r] = None;
        else:
            try:
                thisJson: json = json.loads(r.replace('"', '\\"').replace("'", '"'));
                if medDict[lastMed] is None:
                    medDict[lastMed] = [thisJson];
                else:
                    medDict[lastMed] += [thisJson];
            except:
                pass;
    return medDict;


def __service_getUniAmznMeds(d: Dict[str, List[json.JSONDecoder] | None], cats: List[str] = ["MEDICATION", "TEST_TREATMENT_PROCEDURE"]) -> Tuple[List[str], List[str]]:
    ret: Dict[str, int] = dict();
    unmapped: List[str] = [];
    for k in d.keys():
        if d[k] is None:
            unmapped.append(k);
            continue;
        for ele in d[k]:
            if cats.__contains__(ele["Category"]):
                ret[ele["Text"]] = 0;
    return list(ret.keys()), unmapped;


def __service_getUkbDrugbankMap(meds: List[str], dbMap: Dict[str, str]) -> Dict[str, List[str]]:
    ret: Dict[str, List[str]] = dict();
    for m in meds:
        try:
            if not m.__contains__("+"):
                ret[m] = [dbMap[m]];
                continue;
            for m2 in m.split("+"):
                try:
                    dbid: str = dbMap[m2];
                except:
                    continue;
                try:
                    ret[m].append(dbid);
                except:
                    ret[m] = [dbid];
        except:
            # print(m);
            pass;
    return ret;


def __service_loadDbPkl(pkl: str) -> dict[str, str]:
    assert os.path.exists(pkl);
    with open(pkl, "rb") as f:
        return pickle.load(f);


def __service_appendUnmap(out: str = "ump2.rcd", unmapped: List[str] = []) -> None:
    sys.path.append("..")
    from util.awsMed import runMed;
    with open(out, "w") as f:
        batchLim: int = 500
        for i in range(int(len(unmapped) / batchLim) + (1 if len(unmapped) % batchLim != 0 else 0)):
            l: List[str] = unmapped[i * batchLim: (i + 1) * batchLim];
            # f.write(str(l));
            runMed(str(l), f);


def __service_loadUkbCodingMapd(uri: str) -> Dict[str, str]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        entry: List[str] = f.read().split("\n")[1:-1];
    ret: Dict[str, str] = dict();
    for e in entry:
        code, name = e.split("\t");
        ret[name] = code;
    return ret;


def __service_convUkbToDrugBank() -> None:
    uniCat: Dict[str, int] = dict();
    ukbNlpDict: Dict[str, List[json.JSONDecoder] | None] = __service_loadUkbMlpJson(uri="ukbOutMerge.rcd");
    uniMedIngList: List[str];
    unmapped: List[str];
    cat: List[str] = ["MEDICATION", "TEST_TREATMENT_PROCEDURE"];
    uniMedIngList, unmapped = __service_getUniAmznMeds(ukbNlpDict, cats=cat);
    ukbMappable: Dict[str, List[str]] = __service_getUkbDrugbankMap(uniMedIngList, __service_loadDbPkl("drugbank.pkl"));
    # print(f"{len(ukbMappable.keys())}/{len(uniMedIngList)}")
    # print(f"Unmapped {len(unmapped)}, {unmapped}")
    ukbCodingMap: dict[str, str] = __service_loadUkbCodingMapd("../map/ukbMed.map");

    ukbDrugbankMap: Dict[str, List[str] | None] = dict();
    for k in ukbNlpDict.keys():
        ukbDrugbankMap[ukbCodingMap[k]] = None;
        jList: List[json.JSONDecoder] | None = ukbNlpDict[k];
        if jList is None:
            continue;
        for js in jList:
            if cat.__contains__(js["Category"]):
                try:
                    ingDBId: List[str] = ukbMappable[js["Text"]];
                except:
                    continue;
                try:
                    ukbDrugbankMap[ukbCodingMap[k]] += ingDBId;
                except:
                    ukbDrugbankMap[ukbCodingMap[k]] = ingDBId;

    ukbDBUnmap: int = 0;
    for ukb in ukbDrugbankMap.keys():
        if ukbDrugbankMap[ukb] is None:
            ukbDBUnmap += 1;
    print(f"{ukbDBUnmap}/{len(ukbDrugbankMap.keys())} unmappable")

    with open("../data/ukb2db.pkl", "wb") as f:
        pickle.dump(ukbDrugbankMap, f);


if __name__ == "__main__":
    __service_convUkbToDrugBank();

