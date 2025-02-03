
from torch.utils.data import Dataset, DataLoader;
import numpy as np;

class PtDS(Dataset):

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

