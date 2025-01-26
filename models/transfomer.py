import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig

# Configuration
MAX_SEQ_LENGTH = 64  # Maximum sequence length after padding
EMBEDDING_DIM = 128  # Dimension of the embedding space
BATCH_SIZE = 32  # Batch size for training


class PatientDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_seq_length):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._load_data()

    def _load_data(self):
        """
        Load patient event data. Each file in the data_dir corresponds to one patient.
        Each line in a file represents an event with its details.
        """
        data = []
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r') as f:
                patient_events = f.readlines()
                data.append(patient_events)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        events = self.data[idx]
        tokenized_events = self.tokenizer(events, truncation=True, padding='max_length', max_length=self.max_seq_length,
                                          return_tensors="pt")

        # For demonstration, create dummy labels (medication codes)
        labels = torch.randint(0, 100, (len(events),))  # Random medication codes
        return tokenized_events, labels


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
        logits = self.classifier(cls_output)
        return logits


def collate_fn(batch):
    tokenized_events, labels = zip(*batch)
    input_ids = torch.cat([item['input_ids'] for item in tokenized_events], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in tokenized_events], dim=0)
    labels = torch.cat(labels, dim=0)
    return input_ids, attention_mask, labels


# Load data
data_dir = "path_to_patient_data"  # Path to the directory containing patient data files
tokenizer = lambda x, **kwargs: {  # Dummy tokenizer for demonstration
    'input_ids': torch.randint(0, 100, (len(x), kwargs['max_length'])),
    'attention_mask': torch.ones((len(x), kwargs['max_length']))
}

dataset = PatientDataset(data_dir, tokenizer, MAX_SEQ_LENGTH)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Initialize model
model = MedicationTransformer(embedding_dim=EMBEDDING_DIM, num_classes=100)  # Assuming 100 possible medication classes

# Training loop (example)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(5):  # Example: 5 epochs
    model.train()
    for batch in data_loader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
