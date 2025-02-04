
import math
from typing import Tuple, List;

from torch.utils.data import Dataset, DataLoader;
import torch;
import numpy as np;

class PtDS(Dataset):

    xList: List[torch.Tensor];
    yList: List[torch.Tensor];
    mList: List[torch.Tensor];

    def __service_makeTensor(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray, first: int, fi: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        last: int = min(fi, len(y));
        return torch.from_numpy(X[first:last]), torch.from_numpy(y[first:last]), torch.from_numpy(mask[first:last]);

    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray, batch: int = 64) -> None:
        fullBatch: int = int(math.floor(len(X) / batch));
        self.xList = self.yList = self.mList = [];
        for i in range(fullBatch):
            __x, __y, __mask = self.__service_makeTensor(X, y, mask, i * batch, (i + 1) * batch);
            self.xList.append(__x);
            self.yList.append(__y);
            self.mList.append(__mask);
        if fullBatch * batch < len(X):
            __x, __y, __mask = self.__service_makeTensor(X, y, mask, (i + 1) * batch, len(X));
            self.xList.append(__x);
            self.yList.append(__y);
            self.mList.append(__mask);

    def __len__(self) -> int:
        return len(self.xList)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.xList[i], self.yList[i], self.mList[i];
