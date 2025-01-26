
from typing import List, Tuple;
from enum import Enum;

import numpy as np;

from util.util import tokenizeICD10;


class EvtClass(Enum):
    Dig = 0;
    Lab = 1;
    Med = 2;
    Def = 127


class Evt:
    time: str;
    type: EvtClass;
    assoMed: List[str];
    cont: List[str];

    def __init__(self, time: str, ec: EvtClass, cont: List[str], medList: List[str]) -> None:
        '''
        :param time:
        :param ec:
        :param medList:     Please only to include medications associated to this event
        '''
        self.type = ec;
        __tp: str = time.replace("-", "");
        assert len(__tp) == 8;
        self.time = time;
        self.cont = cont;
        self.assoMed = medList;

    def vectorize(self) -> Tuple[int, np.ndarray, np.ndarray | None]:
        if self.type == EvtClass.Dig:
            return EvtClass.Dig.value, np.array([tokenizeICD10(c) for c in self.cont]), self.assoMed;
        raise NotImplementedError;


class SexAtBirth(Enum):
    Male = 0;
    Female = 1;
    Intersex = 2;
    Def = 127;


class PtDemo:
    sab: SexAtBirth;
    aMo: int;
    aYr: int;
    vec: np.ndarray;

    def __init__(self, sab: SexAtBirth, mo: int, yr: int) -> None:
        self.sab = sab;
        self.aMo = mo;
        self.aYr = yr;
        self.vec = np.array([self.sab.value, self.aMo, self.aYr]);

    def vectorize(self) -> np.ndarray:
        return self.vec;


class Pt:
    id: str;
    evtList: List[Evt];
    dem: PtDemo;
    __ptDem: np.ndarray;

    def __init__(self, id: str, dem: PtDemo) -> None:
        self.id = id;
        self.evtList = [];
        self.dem = dem;

    def newEvt(self, time: str, ec: EvtClass, newDrug: List[str]) -> None:
        self.evtList.append(Evt(time, ec, newDrug));
        self.evtList.sort(key=lambda x: x.time);

    def __service_matrixizeEvtEntry(self, evt: Evt) -> Tuple[int, np.ndarray, int, np.ndarray, List[str]]:
        evtClass, evt, assoMed = evt.vectorize();
        return self.id, self.dem.vec, evtClass, evt, assoMed;

    def vectorize(self, lim: int = 0) -> np.ndarray:
        raise NotImplementedError;


def __test() -> None:
    print(SexAtBirth(1))
    return;


if __name__ == "__main__":
    __test();

