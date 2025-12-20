import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import sys 
sys.path.append("/kaggle/working/recommender_CTR")

class Task2Dataset(Dataset):
    """
    Dataset for CTR prediction task.
    Handles both train/validation (with labels) and test (without labels).
    """
    def __init__(
            self, 
            data_path: str,
            is_train: bool = True,
            dataset_size_limit: int = None
    ):
        self.data = pl.read_parquet(data_path)
        self.is_train = is_train
        
        # Check if label column exists (for test vs train/valid)
        self.has_labels = "label" in self.data.columns
        
        if dataset_size_limit is not None:
            self.data = self.data.head(dataset_size_limit)
            
        print(f"Dataset loaded: {len(self.data):,} samples")
        print(f"Has labels: {self.has_labels}")
        print(f"Columns: {self.data.columns}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the row as a regular Python dict
        row_dict = self.data[idx].to_dict()
        
        item = {
            "user_id": row_dict["user_id"][0],
            "item_seq": row_dict["item_seq"][0],
            "item_id": row_dict["item_id"][0],
            "likes_level": row_dict["likes_level"][0],
            "views_level": row_dict["views_level"][0],
        }
        
        # Add label only if it exists (train/valid data)
        if self.has_labels:
            item["label"] = row_dict["label"][0]
        
        # Add ID for test data (if available)
        if "id" in row_dict:
            item["id"] = row_dict["id"][0]
        
        return item


def collate_fn(batch):
    """Collate function that handles both labeled and unlabeled data."""
    
    # Check if batch has labels
    has_labels = "label" in batch[0]
    has_ids = "id" in batch[0]
    
    collated = {
        "user_ids": torch.tensor([item["user_id"] for item in batch], dtype=torch.long),
        "item_seqs": torch.tensor([item["item_seq"] for item in batch], dtype=torch.long),
        "item_ids": torch.tensor([item["item_id"] for item in batch], dtype=torch.long),
        "likes_levels": torch.tensor([item["likes_level"] for item in batch], dtype=torch.float32),
        "views_levels": torch.tensor([item["views_level"] for item in batch], dtype=torch.float32),
    }
    
    if has_labels:
        collated["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    
    if has_ids:
        collated["id"] = torch.tensor([item["id"] for item in batch], dtype=torch.long)
    
    return collated