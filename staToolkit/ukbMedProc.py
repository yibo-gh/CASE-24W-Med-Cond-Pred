
from typing import List;

from awsMed import runMed;


def parseUkbMap(uri: str = "../map/ukbMed.map", out: str = "ukbOut.rcd") -> None:
    with open(uri, "r") as f:
        allMeds: List[str] = f.read().split("\n")[1:-1];
    for i in range(len(allMeds)):
        allMeds[i] = allMeds[i].split("\t")[1];
    with open(out, "w") as f:
        for i in range(len(allMeds)):
            m: str = allMeds[i];
            print(f"Processing {i + 1:6d}/{len(allMeds)}")
            runMed(m, f);
    return;


if __name__ == "__main__":
    parseUkbMap();
