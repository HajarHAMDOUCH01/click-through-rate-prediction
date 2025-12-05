import argparse
from torch.utils.data import DataLoader

import sys 
sys.path.append("/content/RS_competition")
from task2.training.train import TrainCTRPred
from task2.dataset.dataset import Task2Dataset, collate_fn

def main(args):
    """Main training function with configurable parameters."""

    print("Loading data...")
    
    # Load lookup tables
    from task2.utils import prepare_lookup_tables
    # Prepare lookup tables
    multimodal_embeddings, item_tags = prepare_lookup_tables(
        item_info_df_path=args.item_info_path
    )
    
    # Get dimensions
    num_items, frozen_dim = multimodal_embeddings.shape  # (91718, 128)
    num_tags = item_tags.max() + 1  # max tag ID + 1
    
    print(f"Number of items: {num_items}")
    print(f"Frozen embedding dim: {frozen_dim}")
    print(f"Number of unique tags: {num_tags}")
    
    # Create datasets
    train_dataset = Task2Dataset(
        data_path=args.train_data_path,
        is_train=True,
        dataset_size_limit=args.dataset_size_limit
    )
    
    valid_dataset = Task2Dataset(
        data_path=args.valid_data_path,
        is_train=True,
    )
    import numpy as np

    train_items = set(train_dataset.data['item_id'].to_numpy())
    val_items = set(valid_dataset.data['item_id'].to_numpy())

    unseen_items = val_items - train_items
    print(f"Unseen items in val: {len(unseen_items) / len(val_items) * 100:.1f}%")

    print("="*70 + "\n")
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = TrainCTRPred(
        multimodal_frozen_embeddings=multimodal_embeddings,
        item_tags_lookup=item_tags,
        num_tags=num_tags,
        tag_embed_dim=args.tag_embed_dim,
        learning_rate=args.learning_rate
    )
    
    # Train
    print("\nStarting training...")
    model.train_model(
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path
    )
    
    print("\nDone!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CTR Prediction Model')
    
    # Data paths
    parser.add_argument('--item_info_path', type=str, default='/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/item_info.parquet',
                        help='Path to item info parquet file')
    parser.add_argument('--train_data_path', type=str, default='/kaggle/working/train_with_full_seq.parquet',
                        help='Path to training data parquet file')
    parser.add_argument('--valid_data_path', type=str, default='/kaggle/working/valid_with_full_seq.parquet',
                        help='Path to validation data parquet file')
    parser.add_argument('--test_data_path', type=str, default='/kaggle/working/test_with_full_seq.parquet',
                        help='Path to test data parquet file')
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/',
                        help='Directory to save epoch checkpoints')
    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint')
    # Dataset parameters
    parser.add_argument('--dataset_size_limit', type=int, default=None,
                        help='Limit dataset size for quick experiments (None for full dataset)')
    
    # DataLoader parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Use pinned memory for faster GPU transfer')
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                        help='Disable pinned memory')
    
    # Model parameters
    parser.add_argument('--tag_embed_dim', type=int, default=16,
                        help='Dimension of tag embeddings')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate for optimizer')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    
    # Output paths
    parser.add_argument('--save_path', type=str, default='/content/best_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--output_path', type=str, default='/kaggle/working/predictions.csv',
                        help='Path to save test predictions')
    
    return parser.parse_args()


# ========== MAIN TRAINING SCRIPT ==========
if __name__ == "__main__":
    args = parse_args()
    main(args)