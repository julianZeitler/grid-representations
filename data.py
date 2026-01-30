import os
from pathlib import Path
import re
import torch
import json
import pickle
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, base_path) -> None:
        self.base_path = base_path

        with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        self.num_batches = self.metadata['num_batches']
        self.batch_size = self.metadata['batch_size']
        self.sequence_length = self.metadata['sequence_length']
        self.box_width = self.metadata['box_width']
        self.box_height = self.metadata['box_height']

    def __len__(self) -> int:
        dir = Path(self.base_path)
        pattern = re.compile(r"*.pkl")
        count = sum(1 for p in dir.iterdir() if p.is_file() and pattern.search(p.name))
        return count
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        batch_path = os.path.join(self.base_path, f'batch_{idx:05d}.pkl')
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f)
        
        return batch["normal"], batch["shift"]