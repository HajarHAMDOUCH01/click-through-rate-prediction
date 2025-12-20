import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import sys 
sys.path.append("/kaggle/working/recommender_CTR")
class Task2Dataset(Dataset):
    """
    Dataset for CTR prediction task.
    Handles both train and validation (with labels).
    """

    def __init__(
            self, 
            data_path: str,
            is_train: bool = True,
            dataset_size_limit: int = None
    ):
        self.data = pl.read_parquet(data_path)
        self.is_train = is_train

        if dataset_size_limit is not None:
            self.data = self.data.head(dataset_size_limit)

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
            "label": row_dict["label"][0]
        }
        
        return item


def collate_fn_train(batch, mask_prob=0.15):
    """Training collate with augmentation."""
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    item_seqs = torch.tensor([item["item_seq"] for item in batch], dtype=torch.long)
    
    # Random masking
    keep_mask = torch.rand_like(item_seqs, dtype=torch.float32) > mask_prob
    padding_mask = (item_seqs == 0)
    keep_mask = keep_mask | padding_mask
    item_seqs = item_seqs * keep_mask.long()
    
    collated = {
        "user_ids": torch.tensor([item["user_id"] for item in batch], dtype=torch.long),
        "item_seqs": item_seqs,
        "item_ids": torch.tensor([item["item_id"] for item in batch], dtype=torch.long),
        "likes_levels": torch.tensor([item["likes_level"] for item in batch], dtype=torch.float32),
        "views_levels": torch.tensor([item["views_level"] for item in batch], dtype=torch.float32),
        "labels": labels
    }
    return collated


def collate_fn_val(batch):
    """Validation collate without augmentation."""
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    collated = {
        "user_ids": torch.tensor([item["user_id"] for item in batch], dtype=torch.long),
        "item_seqs": torch.tensor([item["item_seq"] for item in batch], dtype=torch.long),
        "item_ids": torch.tensor([item["item_id"] for item in batch], dtype=torch.long),
        "likes_levels": torch.tensor([item["likes_level"] for item in batch], dtype=torch.float32),
        "views_levels": torch.tensor([item["views_level"] for item in batch], dtype=torch.float32),
        "labels": labels
    }
    return collated