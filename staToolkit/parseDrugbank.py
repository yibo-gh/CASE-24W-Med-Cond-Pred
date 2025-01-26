
import os;
from typing import List, Dict;
from xml.etree import ElementTree as ET;


def __service_loadDrugBank(uri: str = "data/drugbank.xml", pref: str = "{http://www.drugbank.ca}") -> Dict[str, str]:
    assert os.path.exists(uri);
    tree: ET = ET.parse(uri);
    nameMap: Dict[str, str] = dict();
    for c in tree.getroot().findall(f"{pref}drug"):
        names: List[str] = [n.text for n in c.findall(f"{pref}name")]
        assert len(names) == 1;
        id: List[str] = [n.text for n in c.findall(f"{pref}drugbank-id")]
        nameMap[names[0].lower()] = id[0];
        for n in c.find(f"{pref}synonyms").findall(f"{pref}synonym"):
            # if id[0] == "DB00003":
            #     print("synonyms", id[0], n.text)
            nameMap[n.text.lower()] = id[0];
            nameMap[n.text.split(" (")[0].lower()] = id[0];
        for n in c.find(f"{pref}international-brands").findall(f"{pref}international-brand"):
            # if id[0] == "DB00003":
            #     print("intl name", id[0], n.find(f"{pref}name").text.lower())
            nameMap[n.find(f"{pref}name").text.lower()] = id[0];
        '''print(id[0])
        if id[0] == "DB00002":
            for cc in c:
                print(cc.tag)
            exit(0)'''
        for d in c.find(f"{pref}products").findall(f"{pref}product"):
            # if id[0] == "DB00003":
            #     print("brand name", id[0], d.find(f"{pref}name").text);
            nameMap[d.find(f"{pref}name").text.lower()] = id[0];
    return nameMap;


if __name__ == "__main__":
    dbDict: Dict[str, str] = __service_loadDrugBank("../data/drugbank.xml");
    with open("drugbank.pkl", "wb") as f:
        import pickle;
        pickle.dump(dbDict, f);

