
import os;
from typing import List, Dict, Tuple;
import pickle;
import subprocess;

import numpy as np
import torch
from sklearn.metrics import f1_score;
import matplotlib.pyplot as plt;

__icdDrugMap: List[Dict[str, List[str]] | None] = [None];


def int2binVec(i: int, l: int = 0) -> np.ndarray:
    '''
    Small-ending encoding
    :param i:   Target value to be converted
    :param l:   Minimum required vector length
    :return:    A small-ending vector embedding with 0 padding
int2binVec
    Example:    int2binVec(23, 0) -> (1, 1, 1, 0, 1)
    Example 2:  int2binVec(14, 6) -> (0, 1, 1, 1, 0, 0)
    '''
    ret: List[int] = [];
    __tmp: int = i;
    while __tmp > 0:
        ret.append(__tmp % 2);
        __tmp -= ret[-1];
        __tmp /= 2;
    retnp: np.ndarray = np.zeros(max(l, len(ret)), dtype=int);
    retnp[:len(ret)] = np.array(ret);
    return retnp;


def tokenizeICD10(icdOri: str, freqMap: Dict[str, float]) -> np.ndarray:
    icd: str = icdOri.replace(".", "");
    base: np.ndarray = np.concatenate((int2binVec(ord(icd[0].upper()) - 55, 6), int2binVec(int(icd[1:3]), 7))).astype(float);
    return np.concatenate((base, np.array([1 if len(icd) == 3 else freqMap[icd]]))).astype(float);


def tokenizeICD10_old(icdOri: str) -> np.ndarray:
    icd: str = icdOri.replace(".", "");
    rem: str = "";
    for i in range(3, len(icd)):
        if icd[i].isnumeric():
            rem += icd[i];
        else:
            rem += f"{ord(icd[i]) - 65:02d}";
    return int(f"{ord(icd[0]) - 55:02d}{int(icd[1:3])}{rem}");


def checkMedUsed(icd: str, med: str) -> bool:
    return True;


def loadIcd2cui(icd: str) -> Dict[str, str]:
    with open(icd, "r") as f:
        xList: List[str] = f.read().split("\n")[:-1];
    retX: Dict[str, str] = dict();
    for x in xList:
        xp: List[str] = x.split("|");
        retX[xp[13]] = xp[0];
    return retX;


def loadDisMedMap(uri: str) -> Dict[str, List[str]]:
    assert os.path.exists(uri);
    ret: Dict[str, List[str]] = dict();
    with open(uri, "r") as f:
        for _, dis, _, med, _ in f.read().split("\n")[:-1]:
            try:
                ret[dis].append(med);
            except:
                ret[dis] = [med];
    return ret;


def loadIcdDrugMap(pkl: str, icdMap: str = "map/ICD/CUI2ICD10.txt", icdMedMap: str = "TreatKB_approved_with_Finding.counter") -> Dict[str, List[str]]:
    if __icdDrugMap[0] is not None:
        return __icdDrugMap[0];
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            __icdDrugMap[0] = pickle.load(f);
    else:
        assert icdMedMap is not None and os.path.exists(icdMap) and os.path.exists(icdMedMap);
        ixm: Dict[str, str] = loadIcd2cui(icdMap);
        mdm: Dict[str, List[str]] = loadDisMedMap(icdMedMap);
        ret: Dict[str, List[str]] = dict();
        for dis in ixm.keys():
            try:
                ret[dis] = mdm[dis];
            except:
                ret[dis] = [];
        with open(pkl, "wb") as f:
            pickle.dump(ret, f);
        __icdDrugMap[0] = ret;
    return __icdDrugMap[0];


def medMatch(icd2cui: Dict[str, str], ukbMed2db: Dict[str, List[str]], tkbDisMedGT: Dict[str, List[str]],
             icd: str, ukbMedCode: str) -> bool:
    try:
        # print(f"u::104 icd {icd} med code {ukbMedCode} gt {tkbDisMedGT[icd]} tar {ukbMed2db[ukbMedCode]}")
        # print(icd2cui[icd])
        # gt: List[str] = tkbDisMedGT[icd2cui[icd]];
        try:
            gt: List[str] = tkbDisMedGT[icd];
        except:
            gt: List[str] = tkbDisMedGT[icd[:3]];
        # print(gt)
        ukbMed: List[str] | None = ukbMed2db[ukbMedCode];
        # print(ukbMed)
        if ukbMed is None:
            return False;
        for g in gt:
            if ukbMed.__contains__(g):
                # print(f"u::104 icd {icd} med code {ukbMedCode} Machted")
                return True;
    except KeyError:
        return False;

    return False;


def loadCoreMap(icd2cui: str,
                ukbMed2db: str,
                tkbDisMedGT: str,
                ukbMedTokenize: str) -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int]
]:
    assert os.path.exists(icd2cui) and os.path.exists(ukbMed2db) and os.path.exists(tkbDisMedGT);
    i2c: Dict[str, str];
    um2: Dict[str, List[str]];
    tdm: Dict[str, List[str]];
    umt: Dict[str, int];

    with open(icd2cui, "rb") as f:
        i2c = pickle.load(f);
    with open(ukbMed2db, "rb") as f:
        um2 = pickle.load(f);
    with open(tkbDisMedGT, "rb") as f:
        tdm = pickle.load(f);
    with open(ukbMedTokenize, "rb") as f:
        umt = pickle.load(f);
    return i2c, um2, tdm, umt;


def getColNameByFieldID(id: int) -> str:
    return f"participant.p{id:d}";


def __test() -> None:
    print(tokenizeICD10("Z54.W2B"));
    i2c, um2, tdb = loadCoreMap(
        "../map/icd2cui.pkl",
        "../map/ukb2db.pkl",
        "../map/tkbDisMedGT.pkl"
    );
    print(medMatch(i2c, um2, tdb, "G20", "1141169700"))
    return;


def execCmd(cmd: str) -> Tuple[str, str | None]:
    out, err = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate();
    return out.decode(), None if err is None else err.decode();


def npF1Torch(gt: np.ndarray, gh: np.ndarray) -> float:
    assert gt.shape == gh.shape;
    from torchmetrics.functional import f1_score;
    return float(f1_score(torch.from_numpy(gh), torch.from_numpy(gt), task="multilabel", num_labels=gt.shape[-1]));


def npF1(gt: np.ndarray, yh: np.ndarray) -> float:
    tp: int = np.sum((gt == 1) & (yh == 1));
    fp: int = np.sum((gt == 0) & (yh == 1));
    fn: int = np.sum((gt == 1) & (yh == 0));

    pr: float = tp / (tp + fp);
    re: float = tp / (tp + fn);
    return 2 * pr * re / (pr + re + 1e-5);


def __service_npFlatten(dt: np.ndarray) -> np.ndarray:
    if len(dt.shape) < 3:
        return dt;
    return np.concatenate([dt[i] for i in range(len(dt))]);


def auroc(gt: np.ndarray, yh: np.ndarray) -> float:
    aucs = []
    for _gt, pred in zip(gt, yh):
        pos_idx = np.where(_gt == 1)[0]
        neg_idx = np.where(_gt == 0)[0]
        # 如果没有正例或负例，则跳过
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        # 计算 pairwise 差值：对于每个正例与每个负例比较
        diff = pred[pos_idx][:, None] - pred[neg_idx][None, :]
        # 正例分数大于负例的个数，加上相等的 0.5 份
        correct = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
        auc = correct / (len(pos_idx) * len(neg_idx))
        aucs.append(auc)
    # print(f"u::200 roc {np.quantile(aucs, [0, .25, .5, .75, .9, .95])}")
    return aucs, float(np.nanmean(aucs)) if aucs else float('nan')

def auprc(gt: np.ndarray, yh: np.ndarray) -> float:
    aps = []
    for _gt, pred in zip(gt, yh):
        order = np.argsort(-pred)  # 降序排列预测得分的索引
        sorted_gt = _gt[order]
        # 累计求和，得到每个位置上的 TP 数量
        cum_tp = np.cumsum(sorted_gt)
        total_pos = np.sum(sorted_gt)
        if total_pos == 0:
            continue
        precision = cum_tp / (np.arange(1, len(sorted_gt) + 1))
        ap = np.sum(precision * sorted_gt) / total_pos
        aps.append(ap)
    # print(f"u::216 prc {np.quantile(aps, [0, .25, .5, .75, .9, .95])}")
    return aps, float(np.nanmean(aps)) if aps else float('nan')


def mrr(gt: np.ndarray, yh: np.ndarray) -> float:
    mrrs = []
    for _gt, pred in zip(gt, yh):
        order = np.argsort(-pred)  # 降序排序
        found = False
        for rank, idx in enumerate(order, start=1):
            if _gt[idx] == 1:
                mrrs.append(1.0 / rank)
                found = True
                break
        if not found:
            mrrs.append(0.0)
    return float(np.mean(mrrs)) if mrrs else float('nan')


def histk(gt: np.ndarray, yh: np.ndarray, k: int) -> float:
    hits = []
    for _gt, pred in zip(gt, yh):
        topk_indices = np.argsort(-pred)[:k]
        hits.append(np.sum(_gt[topk_indices]))
    return float(np.mean(hits))


def plot_confidence_heatmap(conf: np.ndarray, gt: np.ndarray, title: str = "Confidence and Ground Truth Heatmap", fname: str | None = None):
    """
    绘制 heatmap，横轴为标签，纵轴为样本，
    上半部分显示模型 confidence scores 的热力图，
    同时用点或标记显示 ground truth 为 1 的位置。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 显示 confidence scores 的 heatmap
    cax = ax.imshow(conf, aspect='auto', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Confidence Score')

    # 在 heatmap 上叠加 ground truth 位置。用红色圆点标出位置
    # 遍历每个元素，如果 gt==1，则在对应位置画一个红点
    N, num_labels = gt.shape
    for i in range(N):
        for j in range(num_labels):
            if gt[i, j] == 1:
                ax.plot(j, i, 'ro', markersize=3)

    ax.set_xlabel("Label Index")
    ax.set_ylabel("Sample Index")
    ax.set_title(title)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)


def plot_sorted_metrics(auroc: np.ndarray, auprc: np.ndarray, fname: str | None = None):
    """
    将传进来的 auroc 和 auprc 数组（1D，每个元素代表一个样本或者类别的值）
    分别按照从高到低排序，然后绘制曲线图进行对比。
    """
    # 对指标进行排序
    sorted_auroc = np.sort(auroc)
    sorted_auprc = np.sort(auprc)

    # 生成 x 轴序号
    x = np.arange(len(sorted_auroc))

    # 绘制图形
    plt.figure(figsize=(8, 6))
    plt.plot(x, sorted_auroc, label="Sorted AUROC", marker='o')
    plt.plot(x, sorted_auprc, label="Sorted AUPRC", marker='x')
    plt.xlabel("Sorted Index")
    plt.ylabel("Metric Value")
    plt.title("Sorted AUROC and AUPRC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    if fname is not None:
        plt.savefig(fname)


def detect_metric_outliers(metric_values: np.ndarray, z_thresh: float = 2.0):
    """
    检测一个1D数组中的异常值（outlier），利用 z-score 方法。

    参数：
      metric_values: 1D NumPy 数组（例如每个类别的 AUROC 值）
      z_thresh: z-score 阈值，默认2.0（即离均值超过2个标准差的视为异常）

    返回：
      outlier_indices: 一个1D数组，包含异常值的索引
      z_scores: 对所有数值计算的 z-score 数组
    """
    mean_val = np.mean(metric_values)
    std_val = np.std(metric_values)
    if std_val == 0:
        return np.array([], dtype=int), np.zeros_like(metric_values)
    z_scores = (metric_values - mean_val) / std_val
    outlier_indices = np.where(np.abs(z_scores) > z_thresh)[0]
    return outlier_indices, z_scores


if __name__ == "__main__":
    # print(int2binVec(14, 6));
    # with open("../map/ukbIcdFreq.pkl", "rb") as f:
    #     print(tokenizeICD10("I20.9", pickle.load(f)));
    arr1: np.ndarray = (np.random.random((3, 4, 4)) > 0.5).astype(float);
    # print(arr1);
    arr2: np.ndarray = (np.random.random((3, 4, 4)) > 0.5).astype(float);
    # print(arr2)
    f1: float = npF1Torch(arr1, arr2)
    print(f1);
    print(npF1(arr1, arr2))

