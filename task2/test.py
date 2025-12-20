import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.append("/kaggle/working/recommender_CTR")

from task2.model import CTRModel
from task2.dataset.dataset import Task2Dataset, collate_fn
from task2.model_loader import load_item_embeddings_and_tags
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_test_predictions(model, test_loader):
    """
    Generate predictions for test set.
    
    Args:
        model: Trained CTR model
        test_loader: DataLoader for test data
    
    Returns:
        predictions: List of predicted probabilities
        ids: List of sample IDs
    """
    model.eval()
    all_preds = []
    all_ids = []
    
    print(f"\n{'='*70}")
    print("GENERATING TEST PREDICTIONS")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions'):
            batch_data = {k: v.to(device) for k, v in batch.items() 
                         if k not in ['id', 'labels']}
            
            logits = model.forward(batch_data)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy().tolist())
            
            if 'id' in batch:
                all_ids.extend(batch['id'].tolist())
            else:
                start_id = len(all_ids)
                all_ids.extend(range(start_id, start_id + len(probs)))
    
    return all_preds, all_ids


def save_predictions_to_csv(predictions, ids, output_path='submission.csv'):
    """
    Save predictions to CSV in competition format.
    """
    df = pd.DataFrame({
        'id': ids,
        'Task1&2': predictions
    })
    
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ PREDICTIONS SAVED")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"Total predictions: {len(predictions):,}")
    print(f"\nSample predictions:")
    print(df.head(10))
    print(f"\nPrediction statistics:")
    print(f"  Mean: {sum(predictions)/len(predictions):.4f}")
    print(f"  Min:  {min(predictions):.4f}")
    print(f"  Max:  {max(predictions):.4f}")
    print(f"  Predictions > 0.5: {sum(1 for p in predictions if p > 0.5):,} ({sum(1 for p in predictions if p > 0.5)/len(predictions)*100:.2f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("TEST PREDICTION SCRIPT")
    print("="*70 + "\n")
    
    print("Loading test dataset...")
    test_dataset = Task2Dataset(
        data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/test.parquet",
        is_train=False  
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256,  
        shuffle=False,   
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset):,} samples\n")
    
    embeddings, item_tags, num_items, num_tags = load_item_embeddings_and_tags(
        item_info_path="/kaggle/working/item_info_with_clip.parquet",
        embedding_source="item_clip_emb_d128"
    )
    
    print("\nInitializing model architecture...")
    
    model = CTRModel(
        num_items=num_items,
        frozen_embeddings=embeddings,
        item_tags=item_tags,
        num_tags=num_tags,
        embed_dim=64,
        tag_embed_dim=16,
        k=16,
        num_transformer_layers=2,
        num_heads=4,
        num_cross_layers=3,
        deep_layers=[1024, 512, 256],
        dropout=0.2,
        learning_rate=5e-4,  
    )
    
    checkpoint_path = "/kaggle/working/model_21.pth"  
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=False)
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_auc' in checkpoint:
        print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
    
    model.eval()
    
    predictions, ids = generate_test_predictions(model, test_loader)
    
    output_path = '/kaggle/working/submission.csv'
    save_predictions_to_csv(predictions, ids, output_path)
    
    print("✓ Done!")