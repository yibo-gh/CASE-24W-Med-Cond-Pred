
import os;
from typing import Any, List, Dict, Tuple;
import pickle;


TTY_PRIORITY = [
        "PN",
        "PX",
        "PXQ",
        'PT',  # Preferred Term
        'MH',  # MeSH Heading
        'PN',  # Preferred Name (generic)
        'PF',  # Preferred Form
        'SY',  # Synonym
        'ET',  # Entry Term
        'HT',  # Hierarchical Term
        'RPT',  # Related Preferred Term
        'RHT',  # Related Hierarchical Term
        'RAB',  # Related Abbreviation
        'SSN',  # Short String Name
    ]


class IcdTree:
    parent: List[Any];
    child: List[Any];
    code: str;
    name: str;

    _childDict: Dict[str, int];
    _parenDict: Dict[str, int];

    def __init__(self, code: str, name: str = "") -> None:
        self.code = code;
        self.name = name;
        self.parent = [];
        self.child = [];

        self._childDict = dict();
        self._parenDict = dict();

    def setName(self, name: str) -> Any:
        self.name = name;
        return self;

    def __service_appendList(self, tar: Any, tarList: List[Any], tarDict: Dict[str, Any]) -> Any:
        try:
            tarDict[tar.code];
        except KeyError:
            tarList.append(tar);
            tarDict[tar.code] = 0;
        return self;

    def appendChild(self, tar: Any) -> Any:
        if self.code == tar.code:
            return self;
        return self.__service_appendList(tar, self.child, self._childDict);

    def appendParent(self, tar: Any) -> Any:
        if self.code == tar.code:
            return self;
        return self.__service_appendList(tar, self.parent, self._parenDict);


def parseIcdTree(uri: str) -> Dict[str, IcdTree]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        lines: List[str] = f.read().split("\n");
        if len(lines[-1]) == 0:
            lines = lines[:-1];
    ret: Dict[str, IcdTree] = dict();

    for l in lines:
        _lSplit: List[str] = l.split("|");
        _c1: str = _lSplit[0];
        _rel: str = _lSplit[3];
        _c2: str = _lSplit[4];
        _sup: str = _lSplit[14];

        if _sup != "N" or _rel not in ("PAR", "CHD"):
            continue;

        _par: str;
        _chd: str;

        if _rel == "CHD":
            _par, _chd = _c1, _c2;
        else:
            _par, _chd = _c2, _c1;

        try:
            ret[_chd];
        except:
            ret[_chd] = IcdTree(_chd);

        try:
            ret[_par];
        except:
            ret[_par] = IcdTree(_par);

        ret[_chd].appendParent(ret[_par]);
        ret[_par].appendChild(ret[_chd]);

    return ret;


def parseIcdCUIMap(uri: str, tree: Dict[str, IcdTree]) -> Dict[str, str]:
    assert os.path.exists(uri);

    with open(uri, "r") as f:
        lines: List[str] = f.read().split("\n");
        if len(lines[-1]) == 0:
            lines = lines[:-1];

    ret: Dict[str, List[str]] = dict();       # Dict[ICD, CUI];
    _dCodeName: Dict[str, List[Tuple[str, str]]] = dict();

    for l in lines:
        _lSplit: List[str] = l.split("|");
        _c: str = _lSplit[0];
        _tty: str = _lSplit[12];
        _code: str = _lSplit[13].replace(".", "");
        _des: str = _lSplit[14];
        _sup: str = _lSplit[16];

        if _sup != "N":
            continue;

        try:
            if not _c in ret[_code]:
                ret[_code].append(_c);
        except KeyError:
            ret[_code] = [_c];

        try:
            _dCodeName[_c].append((_tty, _des));
        except KeyError:
            _dCodeName[_c] = [(_tty, _des)];

    for k in _dCodeName.keys():
        _desList: List[Tuple[str, str]] = _dCodeName[k];
        _desList.sort(key=lambda  x:TTY_PRIORITY.index(x[0]) if x[0] in TTY_PRIORITY else len(TTY_PRIORITY));
        try:
            tree[k].setName(_desList[0][1]);
        except KeyError:
            if len(k) < 3 or not k[1:3].isnumeric():
                continue;
            # print(f"uit::129 {k}")
            tree[k] = IcdTree(k).setName(_desList[0][1]);

    return ret;


def __service_printRootNode(tree: Dict[str, IcdTree]) -> None:
    for k in tree.keys():
        if len(tree[k].parent) == 0:
            print(tree[k].code, tree[k].name);


def __service_getAllName(uri: str) -> Dict[str, List[Tuple[str, str]]]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        lines: List[str] = f.read().split("\n");
        if len(lines[-1]) == 0:
            lines = lines[:-1];

    desDict: Dict[str, List[Tuple[str, str]]] = dict();

    for l in lines:
        _lSplit: List[str] = l.split("|");
        _c: str = _lSplit[0];
        _lan: str = _lSplit[1];
        _tty: str = _lSplit[12];
        _code: str = _lSplit[13].replace(".", "");
        _des: str = _lSplit[14];
        _sup: str = _lSplit[16];

        if _lan != "ENG":
            continue;

        try:
            desDict[_c].append((_tty, _des));
        except KeyError:
            desDict[_c] = [(_tty, _des)];

    return desDict;


def __service_getBestDes(d: Dict[str, List[Tuple[str, str]]]) -> Dict[str, str]:
    ret: Dict[str, str] = dict();
    for cui, entries in d.items():
        if not entries:
            continue;

        sorted_entries = sorted(
            entries,
            key=lambda td: TTY_PRIORITY.index(td[0])
            if td[0] in TTY_PRIORITY
            else len(TTY_PRIORITY)
        )

        ret[cui] = sorted_entries[0][1];

    return ret;


def __service_mergeTrees(src: List[Dict[str, Any]]) -> Dict[str, Any]:
    ret: Dict[str, IcdTree] = dict();
    for _d in src:
        for k in _d.keys():
            ret[k] = _d[k];
    return ret;


def main() -> int:
    dRel9: Dict[str, IcdTree] = parseIcdTree("../data/umls/icd9.rel");
    dRel10: Dict[str, IcdTree] = parseIcdTree("../data/umls/icd10.rel");
    dRelCom: Dict[str, IcdTree] = __service_mergeTrees([dRel9, dRel10]);

    icd2cui9: Dict[str, str] = parseIcdCUIMap("../data/umls/icd9.out", dRelCom);
    icd2cui10: Dict[str, str] = parseIcdCUIMap("../data/umls/icd10.out", dRelCom);
    icd2cuiCom: Dict[str, str] = __service_mergeTrees([icd2cui9, icd2cui10]);

    dAllName: Dict[str, List[Tuple[str, str]]] = __service_getAllName("../data/umls/2024AB/META/MRCONSO.RRF");
    dBestName: Dict[str, str] = __service_getBestDes(dAllName);

    for k in dBestName.keys():
        try:
            dRelCom[k].name = dBestName[k];
        except KeyError:
            continue;

    print(f"uit::156")
    for _cui in icd2cuiCom["X58"]:
        print("X58", _cui, dRelCom[_cui].name)

    with open("../data/umlsIcdInfo.pkl", "wb") as f:
        pickle.dump((dRel9, dRel10, dRelCom, icd2cui9, icd2cui10, icd2cuiCom), f);

    return 0;


if __name__ == "__main__":
    main();
