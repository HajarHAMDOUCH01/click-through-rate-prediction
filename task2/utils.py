import numpy as np
import torch
import polars as pl

def prepare_lookup_tables(item_info_df_path):
    """
    Prepare lookup tables from item_info dataframe.
    
    Args:
        item_info_df_path: path to item_info parquet/csv file
        
    Returns:
        multimodal_frozen_embeddings: (num_items, 128)
        item_tags_lookup: (num_items, 5)
    """
    
    # Load item info
    if isinstance(item_info_df_path, str):
        item_info_df = pl.read_parquet(item_info_df_path)
    else:
        item_info_df = item_info_df_path
    
    print(f"Item info shape: {item_info_df.shape}")
    
    # Sort by item_id to ensure proper indexing
    item_info_df = item_info_df.sort("item_id")
    
    # 1. Extract embeddings (already PCA'd to 128 dims)
    # item_emb_d128 is likely a list column: [0.1, 0.2, ..., 0.128]
    embeddings_list = item_info_df["item_emb_d128"].to_list()
    multimodal_frozen_embeddings = np.array(embeddings_list, dtype=np.float32)
    
    print(f"Multimodal embeddings shape: {multimodal_frozen_embeddings.shape}")
    
    # 2. Extract item tags
    # item_tags is a list column: [tag1, tag2, tag3, tag4, tag5]
    item_tags_list = item_info_df["item_tags"].to_list()
    item_tags_lookup = np.array(item_tags_list, dtype=np.int64)
    
    print(f"Item tags lookup shape: {item_tags_lookup.shape}")
    
    # 3. Verify row 0 is padding (if item_id starts from 0)
    first_item_id = item_info_df["item_id"][0]
    if first_item_id == 0:
        assert np.all(multimodal_frozen_embeddings[0] == 0), "Row 0 embeddings should be zeros"
        assert np.all(item_tags_lookup[0] == 0), "Row 0 tags should be zeros"
        print("✓ Padding row (item_id=0) verified")
    else:
        print(f"⚠ Warning: First item_id is {first_item_id}, not 0. May need padding row!")
    
    return multimodal_frozen_embeddings, item_tags_lookup
