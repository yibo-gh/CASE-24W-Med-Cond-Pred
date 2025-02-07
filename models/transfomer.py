
import math
from typing import Tuple, List;

from torch.utils.data import Dataset, DataLoader;
import torch;
import numpy as np;

class PtDS(Dataset):

    xList: torch.Tensor;
    yList: torch.Tensor;
    mList: torch.Tensor;

    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        assert len(X) == len(y) == len(mask);
        assert y.shape == mask.shape;
        self.xList = torch.from_numpy(X);
        self.yList = torch.from_numpy(y);
        self.mList = torch.from_numpy(mask);

    def __len__(self) -> int:
        return len(self.xList)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.xList[idx], self.yList[idx], self.mList[idx];
