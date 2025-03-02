import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.hub

# 选择模型，例如 "bert-base-uncased"
model_name = "bert-base-uncased"

# 从 PyTorch Hub 加载模型（仅获取 feature extractor）
model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

# 添加分类层（适用于 58 维二分类任务）
num_labels = 58
classifier = nn.Linear(model.config.hidden_size, num_labels)  # 直接加一个分类层
model.classifier = classifier  # 替