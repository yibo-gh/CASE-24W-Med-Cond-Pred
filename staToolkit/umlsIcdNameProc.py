
import os;
from typing import Dict, List, Tuple;


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


def main() -> int:
    d: Dict[str, List[Tuple[str, str]]] = __service_getAllName("../data/umls/2024AB/META/MRCONSO.RRF");
    dBestName: Dict[str, str] = __service_getBestDes(d);
    _cuiList: List[str] = ["C0274281", "C0496512", "C4759661"];
    for _cui in _cuiList:
        print(_cui, dBestName[_cui]);
    return 0;


if __name__ == "__main__":
    main();

