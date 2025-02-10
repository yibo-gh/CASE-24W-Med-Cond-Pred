
from typing import Tuple, List, Dict;

from torch.utils.data import Dataset, DataLoader;
import torch;
import numpy as np;
import torch.nn as nn;
from transformers import BertModel, BertConfig;

class PtDS(Dataset):

    x: torch.Tensor;
    y: torch.Tensor;
    xm: torch.Tensor;
    ym: torch.Tensor;
    pidEntryMap: Dict[int, List[int]];
    pidList: List[int];

    def __init__(self, X: np.ndarray, xMask: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        assert len(X) == len(y) == len(mask) and X.shape == xMask.shape and y.shape == mask.shape;
        assert y.shape == mask.shape;
        self.x = torch.from_numpy(X);
        self.xm = torch.from_numpy(xMask);
        self.y = torch.from_numpy(y);
        self.ym = torch.from_numpy(mask);
        self.pidEntryMap = dict();

        for i in range(len(X)):
            pid: int = int(X[i][0]);
            # print(pid, i);
            try:
                self.pidEntryMap[pid].append(i);
            except KeyError:
                self.pidEntryMap[pid] = [i];
        self.pidList = list(self.pidEntryMap.keys());

    def __len__(self) -> int:
        return len(self.pidList)

    def __getitem__(self, idx: int | List[int] | slice) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        __ptEntry: torch.Tensor;
        if isinstance(idx, int):
            __ptEntry = torch.from_numpy(np.array(self.pidEntryMap[self.pidList[idx]]));
        elif isinstance(idx, list):
            __tmpEntry: List[int] = [];
            for i in idx:
                __tmpEntry += self.pidEntryMap[self.pidList[i]];
            __ptEntry = torch.from_numpy(np.array(__tmpEntry));
        elif isinstance(idx, slice):
            __tmpEntry: List[int] = [];
            for i in range(*idx.indices(len(self.pidList))):
                __tmpEntry += self.pidEntryMap[self.pidList[i]];
            __ptEntry = torch.from_numpy(np.array(__tmpEntry));
        else:
            raise TypeError;

        return self.x[__ptEntry], self.xm[__ptEntry], self.y[__ptEntry], self.ym[__ptEntry];


class MedicationTransformer(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(MedicationTransformer, self).__init__()
        self.bert_config = BertConfig(hidden_size=embedding_dim, num_hidden_layers=4, num_attention_heads=8,
                                      intermediate_size=512)
        self.bert = BertModel(self.bert_config)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token representation
        return self.classifier(cls_output)
