
import os;
import pickle
from typing import List, Dict;
import json;

from util.util import execCmd;


def __service_getFList(uri: str) -> List[str]:
    assert os.path.exists(uri);
    ret, _ = execCmd(f"ls {uri}");
    flist: List[str] = ret.split("\n")[:-1];
    for i in range(len(flist)):
        flist[i] = flist[i][:-5]
    return flist;


def __service_isPosDig(j: json.JSONDecoder) -> bool:
    ret: bool = False;
    for attr in j["Traits"]:
        if attr["Name"] == "NEGATION":
            return False;
        if attr["Name"] == "DIAGNOSIS":
            ret = True;
    return ret;


def __service_loadJs(uri: str) -> Dict[str, List[str]]:
    assert os.path.exists(uri);
    fList: List[str] = __service_getFList(uri);
    ret: Dict[str, List[str]] = dict();
    for i in range(len(fList)):
        dbid: str = fList[i];
        fUri: str = f"{uri}/{dbid}.json";
        assert os.path.exists(fUri);
        with open(fUri, "r") as f:
            lines: List[str] = f.read().split("\n")[:-1];
        ret[dbid] = [];
        for l in lines:
            try:
                j: json.JSONDecoder = json.loads(l.replace("'", '"'));
            except json.decoder.JSONDecodeError:
                continue;
            if not __service_isPosDig(j):
                continue;
            j["ICD10CMConcepts"].sort(key=lambda x: x["Score"], reverse=True);
            # print(j["Text"], __service_isPosDig(j), j["ICD10CMConcepts"][0]["Code"])
            # print(j["ICD10CMConcepts"])
            ret[dbid].append(j["ICD10CMConcepts"][0]["Code"]);
            ret[dbid].append(j["ICD10CMConcepts"][0]["Code"][:3]);
        __tmpUniCode: Dict[str, int] = dict();
        for c in ret[dbid]:
            __tmpUniCode[c] = 0;
        ret[dbid] = list(__tmpUniCode.keys());
        ret[dbid].sort();
        # print(dbid, ret[dbid]);
        return ret;


if __name__ == "__main__":
    d: Dict[str, List[str]] = __service_loadJs("DbICD");
    with open("../map/drugbankIcd.pkl", "wb") as f:
        pickle.dump(d, f);


