
from typing import List, Dict;


def __service_loadDict(fi: Dict[str, List[str] | None] = dict(), l: List[str] = []) -> Dict[str, List[str] | None]:
    lastMed = l[0];
    for ele in l:
        if ele[0] != "{":
            lastMed = ele;
            fi[lastMed] = None;
            continue;
        try:
            fi[lastMed].append(ele);
        except:
            fi[lastMed] = [ele];
    return fi;


def __service_printMissStt(d: Dict[str, List[str] | None]) -> None:
    mis: int = 0;
    for k in d.keys():
        if d[k] is None:
            mis += 1;
    print(f"{mis}/{len(d.keys())} missing")


def __service_merge(f1: str = "ukbOut.rcd", f2: str = "ukbOutSup.rcd", out: str = "ukbOutMerge.rcd") -> None:
    with open(f1, "r") as f:
        fi: Dict[str, List[str] | None] = __service_loadDict(l=f.read().split("\n")[:-1]);
    with open(f2, "r") as f:
        fi = __service_loadDict(fi=fi, l=f.read().split("\n")[:-1]);
    with open(out, "w") as f:
        for ele in fi.keys():
            f.write(f"{ele}\n");
            if fi[ele] is not None:
                for j in fi[ele]:
                    f.write(f"{j}\n");


if __name__ == "__main__":
    __service_merge();
