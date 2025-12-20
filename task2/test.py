import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("/content/recommender_CTR")

from task2.model import CTRModel
from task2.dataset.dataset import Task2Dataset, collate_fn

def generate_test_predictions(model, test_loader, device='cuda'):
    """
    Generate predictions for test set.
    
    Args:
        model: Trained CTR model
        test_loader: DataLoader for test data
        device: Device to run inference on
    
    Returns:
        predictions: List of predicted probabilities
        ids: List of sample IDs
    """
    model.eval()
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions'):
            # Move batch to device
            batch_data = {k: v.to(device) for k, v in batch.items() if k != 'id'}
            
            # Get predictions
            logits = model.forward(batch_data)
            probs = torch.sigmoid(logits)
            
            # Store predictions and IDs
            all_preds.extend(probs.cpu().numpy().tolist())
            
            # If batch has IDs, store them
            if 'id' in batch:
                all_ids.extend(batch['id'].tolist())
            else:
                # Generate sequential IDs if not provided
                start_id = len(all_ids)
                all_ids.extend(range(start_id, start_id + len(probs)))
    
    return all_preds, all_ids


def save_predictions_to_csv(predictions, ids, output_path='submission.csv'):
    """
    Save predictions to CSV in competition format.
    
    Args:
        predictions: List of predicted probabilities
        ids: List of sample IDs
        output_path: Path to save CSV file
    """
    df = pd.DataFrame({
        'id': ids,
        'Task1&2': predictions
    })
    
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Predictions saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Sample predictions:")
    print(df.head(10))
    print(f"{'='*70}\n")


# Main execution code
if __name__ == "__main__":
    
    # Load test dataset (you'll need to modify Task2Dataset to handle test data)
    from task2.dataset.dataset import Task2Dataset, collate_fn
    
    test_dataset = Task2Dataset(
        data_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/test.parquet",
        is_train=False  # Set to False for test data
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256,  # Larger batch size for inference
        shuffle=False,  # Don't shuffle test data
        collate_fn=collate_fn
    )
    
    # Load embeddings and model
    from task2.model_loader import load_item_embeddings_and_tags
    
    embeddings, item_tags, num_items, num_tags = load_item_embeddings_and_tags(
        item_info_path="/kaggle/working/item_info_with_clip.parquet",
        embedding_source="item_clip_emb_d128"
    )
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Load best checkpoint
    checkpoint_path = "/kaggle/working/model_21.pth"  # Use your best model
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation AUC: {checkpoint.get('val_auc', 'N/A'):.4f}\n")
    
    # Generate predictions
    predictions, ids = generate_test_predictions(model, test_loader, device)
    
    # Save to CSV
    save_predictions_to_csv(predictions, ids, output_path='/kaggle/working/submission.csv')
    
    # Print statistics
    print(f"Prediction statistics:")
    print(f"  Mean: {sum(predictions)/len(predictions):.4f}")
    print(f"  Min:  {min(predictions):.4f}")
    print(f"  Max:  {max(predictions):.4f}")
    print(f"  Predictions > 0.5: {sum(1 for p in predictions if p > 0.5)} ({sum(1 for p in predictions if p > 0.5)/len(predictions)*100:.2f}%)")