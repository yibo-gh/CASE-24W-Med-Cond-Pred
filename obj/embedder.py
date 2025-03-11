
from typing import Dict, List, Tuple;
import sys;

import numpy as np;

sys.path.append("..");
from obj.pt import Pt;

class Embedder:

    def __init__(self, ukb2db: str, db2emd: str, allPt: str) -> None:
        super().__init__();

    def dtBatching(self, icd: str) -> Tuple[Dict[str, int], Dict[str, List[np.ndarray]]]:
        raise NotImplementedError;

    def getPtVec(self, id: str, medSeqMap: Dict[str, int]) -> Tuple[
        int, np.ndarray, List[np.ndarray], np.ndarray
        # PID, dem, X, yGT
    ]:
        raise NotImplementedError;

    def getYGtClassLen(self) -> int:
        raise NotImplementedError;

