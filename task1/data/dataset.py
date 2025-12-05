from torchvision import transfoms 
from PIL import Image
import polars as pl
import pandas as pd

class Task1Dataset:
    def __init__(self,
            item_data_path: str,
            dataset_size_limit: int = None
    ):
        self.item_features = pl.read_parquet(item_data_path)

        if dataset_size_limit is not None:
            self.item_features = self.item_features.head(dataset_size_limit)
        self.df = self.item_features.to_pandas()
        print("DATA FRAME TYPE:", type(self.df))

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        is_padding = self.df.row["item_id" == 0]

        if is_padding:
            return {
                "item_id": 0,
                "item_image": None,
                "item_title": None,
                "is_pad": True
            }

        image_path = row["image_path"]
        
        if image_path and image_path != "":
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        item_id = row["item_id"]
        title = row["item_title"]
        
        return {
                "item_id": item_id,
                "item_image": image,
                "item_title": title,
                "is_pad": False
            }


