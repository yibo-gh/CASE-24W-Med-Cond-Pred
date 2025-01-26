
from typing import List, Dict, Tuple;
import json;


class UkbSupplement:
    name: str;
    start: int; end: int;
    j: json.JSONDecoder | None;
    js: str | None;

    def __init__(self, s: str, lastPtr: int) -> None:
        self.start = lastPtr + 4;
        self.name = s;
        self.end = self.start + len(self.name);
        self.j = None;
        self.js = None;

    def assignJson(self, j: json) -> None:
        assert j["BeginOffset"] >= self.start and j["EndOffset"] <= self.end and len(j["Text"]) == j["EndOffset"] - j["BeginOffset"];
        self.j = j;
        try:
            assert self.name.__contains__(j["Text"]);
        except Exception as e:
            print(self.name, j["Text"])
            raise e;

    def validate(self, j: json) -> bool:
        return j["BeginOffset"] >= self.start and j["EndOffset"] <= self.end;


def __service_parseStr(s: str) -> List[UkbSupplement]:
    sl: List[str] = s[1:-1].split(", ");
    ret: List[UkbSupplement] = [];
    for ele in sl:
        ret.append(UkbSupplement(ele[1:-1], lastPtr=(-2 if len(ret) == 0 else ret[-1].end)))
    return ret;


def __service_export() -> None:
    with open("ump2.rcd", "r") as f:
        l: List[str] = f.read().split("\n")[:-1];

    usl: List[UkbSupplement] = [];
    __tmpUsl: List[UkbSupplement] = [];
    for s in l:
        if s[0] == "[":
            usl += __tmpUsl;
            __tmpUsl: List[UkbSupplement] = __service_parseStr(s);
            continue;
        try:
            j: json = json.loads(s.replace('"', '\\"').replace("'", '"'));
        except:
            continue;
        for us in __tmpUsl:
            if us.validate(j):
                us.assignJson(j);
                us.js = s;
                break;

    usl += __tmpUsl;
    nullCt: int = 0;
    for us in usl:
        if us.j is None:
            nullCt += 1;
    print(f"{nullCt}/{len(usl)} with no js matched.")

    with open("ukbOutSup.rcd", "w") as f:
        for us in usl:
            f.write(f"{us.name}\n");
            if us.j is not None:
                f.write(f"{us.js}\n");


if __name__ == "__main__":
    __service_export();

