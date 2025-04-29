
import os;
import pickle
from typing import List, Dict;
from xml.etree import ElementTree as ET;

from util.awsMed import getICD;


def __service_fetchPerItemICD(uri: str = "data/drugbank.xml", tarID: Dict[str, int] = dict(), pref: str = "{http://www.drugbank.ca}", outPref: str = "./DbICD/") -> None:
    assert os.path.exists(uri);
    tree: ET = ET.parse(uri);
    if not os.path.exists(outPref):
        os.mkdir(outPref);
    allChild = tree.getroot().findall(f"{pref}drug");
    counter: int = 0;
    for c in allChild:
        id: str = [n.text for n in c.findall(f"{pref}drugbank-id")][0];

        print(f"Processing {counter + 1:5d}/{len(allChild)}, {id}", end="")
        try:
            tarID[id];
            indText: str | None = c.find(f"{pref}indication").text;
            assert indText is not None;
            _out: str = f"{outPref}{id}.json";
            if os.path.exists(_out):
                print(". Exists", end="");
                raise KeyError;
            print(". Fetching demanding item.");
            getICD(indText, out=_out);
        except KeyError:
            print(". No need, pass.");
        except AssertionError:
            print(". Not available.");
        except Exception as e:
            raise e;
        counter += 1;


def __serivce_getUkbUniDB(d: Dict[str, List[str] | None]) -> Dict[str, int]:
    ret: Dict[str, int] = dict();
    for k in d.keys():
        if d[k] is None:
            continue;
        for id in d[k]:
            ret[id] = 0;
    return ret;


if __name__ == "__main__":
    with open("../map/ukb2db.umls.pkl", "rb") as f:
        uniDB: Dict[str, List[str] | None] = pickle.load(f);
    __service_fetchPerItemICD(uri="../data/drugbank.xml", tarID=__serivce_getUkbUniDB(uniDB));
    '''with open("dbApprovedICD.pkl", "wb") as f:
        import pickle;
        pickle.dump(dbDict, f);'''

