
from typing import Dict;
import os;
import pickle;


def __service_parseUkbMed(uri: str) -> Dict[str, int]:
    assert os.path.exists(uri);
    with open(uri, "r") as f:
        lines: List[str] = f.read().split("\n")[:-1];
    ret: Dict[str, int] = dict();
    for i in range(len(lines)):
        l: str = lines[i];
        ret[l.split("\t")[0]] = i;
    return ret;


if __name__ == "__main__":
    with open("../map/ukbMedTokenize.pkl", "wb") as f:
        pickle.dump(__service_parseUkbMed("../map/ukbMed.map"), f);
