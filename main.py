
import sys
from typing import List, Tuple, Dict;

import pickle;
import numpy as np;

sys.path.append("ukbUtil");
from ukbUtil import fetchDB, loadMeta, readFile;


def main() -> int:
    token: str = "MVFsNanigw5fKGJUSbXmMPHSUa7EHR5i";
    '''pref, _, dt, _ = fetchDB(dsID="record-GxF1x2QJbyfKGbP5yY9JyVZ1",
                             keys=["ICD", "Medication", "date"],
                             db=f"{token}__project-GxBzxGQJqvQ8XgpxffxXQf13",
                             beeline="bin/spark-3.2.3-bin-hadoop2.7/bin/beeline",
                             colCode=[31, 34, 52, 21000]);'''

    i9m: Dict[str, str] = loadIcd2cui("map/ICD/CUI2ICD9.txt");
    ixm: Dict[str, str] = loadIcd2cui("map/ICD/CUI2ICD10.txt");
    print(i9m["41"])
    exit(0);
    with open(f"data/1737145582028.pkl", "rb") as f:
        dt = pickle.load(f);
    print(dt.shape);
    print(dt.meta);
    return 0;


if __name__ == "__main__":
    main();

