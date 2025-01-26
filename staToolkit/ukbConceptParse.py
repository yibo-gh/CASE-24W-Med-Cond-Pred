
import os;
from typing import List, TextIO;


def genOut(ukb: str = "../map/ukbMed.map", fMaxLine: int = 250, out: str = "../map/ukbMedNoSeq") -> None:
    assert os.path.exists(ukb);
    with open(ukb, "r") as f:
        allNames: List[str] = [l.split("\t")[1] for l in f.read().split("\n")[1:-1]];
    fList: List[TextIO] = [open(f"{out}-{i}.rcd", "w") for i in range(int(len(allNames) / fMaxLine + (0 if len(allNames) % fMaxLine == 0 else 1)))];

    for i in range(len(fList)):
        for n in allNames[fMaxLine * i : min(fMaxLine * (i + 1), len(allNames))]:
            fList[i].write(f"{n}\n");
        fList[i].close();
        open(f"{out}-{i}.json", "w").close();


if __name__ == "__main__":
    genOut();

