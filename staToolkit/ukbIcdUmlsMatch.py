
import os;
import pickle;
from typing import List, Tuple, Dict;

from umlsIcdTree import IcdTree;


def _uri(uri: str) -> str:
    assert os.path.exists(uri);
    return uri;


def __service_makeIcdDescMap(ulmsIcd: _uri, allIcd:_uri) -> Dict[str, str]:
    with open(ulmsIcd, "rb") as f:
        _, _, dRelCom, _, _, icd2cuiCom = pickle.load(f);
    with open(allIcd, "r") as f:
        _ai: List[str] = f.read().split("\n");
        if len(_ai[-1]) == 0:
            _ai = _ai[:-1];

    ret: Dict[str, str] = dict();
    for _i in _ai:
        _des: str = "";
        try:
            for _cui in icd2cuiCom[_i]:
                _des += f"{dRelCom[_cui].name}\n";
            _des = _des[:-1];
            ret[_i] = _des;
        except:
            continue;
    return ret;


def __service_biobert(icd10_dict: Dict[str, str]) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    import os
    import pickle

    # --- 配置参数 ---
    BIOGPT_MODEL = "microsoft/BioGPT"
    EMBEDDING_OUTPUT_DIR = "icd10_embeddings"
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 准备模型 ---
    print("Loading BioGPT...")
    tokenizer = AutoTokenizer.from_pretrained(BIOGPT_MODEL)
    model = AutoModel.from_pretrained(BIOGPT_MODEL)
    model.to(DEVICE)
    model.eval()

    # --- 平均池化函数 ---
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # --- 创建输出文件夹 ---
    os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)

    # --- 批量处理 ---
    print("Generating embeddings...")
    codes = list(icd10_dict.keys())
    descriptions = list(icd10_dict.values())

    for i in tqdm(range(0, len(codes), BATCH_SIZE)):
        batch_codes = codes[i:i + BATCH_SIZE]
        batch_descs = descriptions[i:i + BATCH_SIZE]

        encoded_input = tokenizer(batch_descs, padding=True, truncation=True, max_length=256, return_tensors='pt')
        encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)

        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        batch_embeddings = batch_embeddings.cpu()

        # 保存每个code一个小文件（仿照原R脚本）
        for code, embedding in zip(batch_codes, batch_embeddings):
            out_path = os.path.join(EMBEDDING_OUTPUT_DIR, f"{code}.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(embedding.numpy(), f)

    print("All done!")


def main() -> int:
    icdNameMap: Dict[str, str] = __service_makeIcdDescMap("../data/umlsIcdInfo.pkl", "../notebook/allIcds.txt");
    print(icdNameMap["X58"], end="\n--------\n")
    print(icdNameMap["F640"], end="\n--------\n")
    print(icdNameMap["E110"])
    __service_biobert(icdNameMap)
    return 0;


if __name__ == "__main__":
    main();

