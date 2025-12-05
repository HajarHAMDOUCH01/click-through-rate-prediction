"""
Comprehensive diagnostic to find why your model overfits.
Run this BEFORE training to understand your data.
"""

import numpy as np
import torch
import polars as pl
from collections import Counter
import matplotlib.pyplot as plt

class CTRDataDiagnostic:
    """Analyze training and validation data for issues."""
    
    def __init__(self, train_path, valid_path, item_info_path):
        print("="*80)
        print("LOADING DATA FOR DIAGNOSTIC")
        print("="*80)
        
        self.train_df = pl.read_parquet(train_path)
        self.valid_df = pl.read_parquet(valid_path)
        self.item_info = pl.read_parquet(item_info_path)
        
        print(f"Train samples: {len(self.train_df):,}")
        print(f"Valid samples: {len(self.valid_df):,}")
        print(f"Total items: {len(self.item_info):,}")
    
    def analyze_label_distribution(self):
        """Check if train and val have similar label distributions."""
        print("\n" + "="*80)
        print("1. LABEL DISTRIBUTION")
        print("="*80)
        
        train_pos_rate = self.train_df['label'].mean()
        val_pos_rate = self.valid_df['label'].mean()
        
        print(f"Train positive rate: {train_pos_rate:.4f}")
        print(f"Val positive rate:   {val_pos_rate:.4f}")
        print(f"Difference:          {abs(train_pos_rate - val_pos_rate):.4f}")
        
        if abs(train_pos_rate - val_pos_rate) > 0.05:
            print("⚠️  WARNING: Label distributions are different!")
            print("   This suggests train/val are from different time periods")
            print("   or have different user populations")
            return "MISMATCH"
        else:
            print("✓ Label distributions are similar")
            return "OK"
    
    def analyze_item_overlap(self):
        """Check item overlap between train and val."""
        print("\n" + "="*80)
        print("2. ITEM OVERLAP")
        print("="*80)
        
        train_items = set(self.train_df['item_id'].to_numpy())
        val_items = set(self.valid_df['item_id'].to_numpy())
        
        unseen_items = val_items - train_items
        overlap_items = val_items & train_items
        
        unseen_pct = len(unseen_items) / len(val_items) * 100
        overlap_pct = len(overlap_items) / len(val_items) * 100
        
        print(f"Unique train items: {len(train_items):,}")
        print(f"Unique val items:   {len(val_items):,}")
        print(f"Overlap:            {len(overlap_items):,} ({overlap_pct:.2f}%)")
        print(f"Unseen in val:      {len(unseen_items):,} ({unseen_pct:.2f}%)")
        
        if unseen_pct > 20:
            print("⚠️  WARNING: Many unseen items in validation!")
            print("   Model struggles with new items → focus on item representations")
            return "HIGH_UNSEEN"
        else:
            print("✓ Item overlap is good")
            return "OK"
    
    def analyze_item_popularity(self):
        """Check if popular items appear in both train and val."""
        print("\n" + "="*80)
        print("3. ITEM POPULARITY DISTRIBUTION")
        print("="*80)
        
        train_items_freq = Counter(self.train_df['item_id'].to_numpy())
        val_items_freq = Counter(self.valid_df['item_id'].to_numpy())
        
        # Top 100 items
        top_100_train = set([item for item, _ in train_items_freq.most_common(100)])
        top_100_val = set([item for item, _ in val_items_freq.most_common(100)])
        
        overlap_top100 = len(top_100_train & top_100_val)
        
        print(f"Top 100 train items in top 100 val: {overlap_top100}/100")
        print(f"Top train item appears {train_items_freq.most_common(1)[0][1]} times")
        print(f"Top val item appears   {val_items_freq.most_common(1)[0][1]} times")
        
        # Check if top items are same
        top_20_train = [item for item, _ in train_items_freq.most_common(20)]
        print(f"\nTop 20 train items also in top 100 val: {len(set(top_20_train) & top_100_val)}/20")
        
        if overlap_top100 < 70:
            print("⚠️  WARNING: Top items are different in train vs val!")
            print("   Popularity distributions shifted → temporal data split likely")
            return "POPULARITY_SHIFT"
        else:
            print("✓ Top items are consistent")
            return "OK"
    
    def analyze_user_patterns(self):
        """Check if users in train and val are different."""
        print("\n" + "="*80)
        print("4. USER PATTERNS")
        print("="*80)
        
        train_users = set(self.train_df['user_id'].to_numpy())
        val_users = set(self.valid_df['user_id'].to_numpy())
        
        overlap_users = len(train_users & val_users)
        only_train_users = len(train_users - val_users)
        only_val_users = len(val_users - train_users)
        
        print(f"Unique train users: {len(train_users):,}")
        print(f"Unique val users:   {len(val_users):,}")
        print(f"Overlap:            {overlap_users:,}")
        print(f"Only in train:      {only_train_users:,}")
        print(f"Only in val:        {only_val_users:,}")
        
        if only_val_users > 0:
            print("⚠️  WARNING: Val has users not in training!")
            print("   Model must generalize to new users → hard problem")
            return "NEW_USERS"
        else:
            print("✓ All val users appear in training")
            return "OK"
    
    def analyze_sequence_patterns(self):
        """Check if item sequences are realistic."""
        print("\n" + "="*80)
        print("5. SEQUENCE PATTERNS")
        print("="*80)
        
        # Sample sequences
        train_seqs = self.train_df['item_seq'].head(100).to_list()
        
        # Check padding ratio
        total_positions = sum(len(seq) for seq in train_seqs)
        padded_positions = sum(np.sum(np.array(seq) == 0) for seq in train_seqs)
        
        padding_ratio = padded_positions / total_positions if total_positions > 0 else 0
        
        print(f"Sample sequences (first 100 samples):")
        print(f"  Average sequence length: {np.mean([len(s) for s in train_seqs]):.1f}")
        print(f"  Padding ratio: {padding_ratio:.2%}")
        print(f"  Max sequence length: {max(len(s) for s in train_seqs)}")
        print(f"  Min sequence length: {min(len(s) for s in train_seqs)}")
        
        if padding_ratio > 0.5:
            print("⚠️  WARNING: High padding ratio!")
            print("   Many sequences are very short")
            return "SHORT_SEQS"
        else:
            print("✓ Sequences are well-distributed")
            return "OK"
    
    def analyze_feature_ranges(self):
        """Check feature value ranges (likes_level, views_level)."""
        print("\n" + "="*80)
        print("6. FEATURE RANGES")
        print("="*80)
        
        print("Train set:")
        print(f"  likes_level:  min={self.train_df['likes_level'].min():.2f}, " 
              f"max={self.train_df['likes_level'].max():.2f}, "
              f"mean={self.train_df['likes_level'].mean():.2f}")
        print(f"  views_level:  min={self.train_df['views_level'].min():.2f}, "
              f"max={self.train_df['views_level'].max():.2f}, "
              f"mean={self.train_df['views_level'].mean():.2f}")
        
        print("\nVal set:")
        print(f"  likes_level:  min={self.valid_df['likes_level'].min():.2f}, "
              f"max={self.valid_df['likes_level'].max():.2f}, "
              f"mean={self.valid_df['likes_level'].mean():.2f}")
        print(f"  views_level:  min={self.valid_df['views_level'].min():.2f}, "
              f"max={self.valid_df['views_level'].max():.2f}, "
              f"mean={self.valid_df['views_level'].mean():.2f}")
        
        # Check for data leakage: are these post-click metrics?
        train_likes_range = self.train_df['likes_level'].max() - self.train_df['likes_level'].min()
        val_likes_range = self.valid_df['likes_level'].max() - self.valid_df['likes_level'].min()
        
        if train_likes_range > 0 and val_likes_range > 0:
            print("✓ Features have variation (not constant)")
            return "OK"
        else:
            print("⚠️  WARNING: Features might be constant!")
            return "CONSTANT_FEATURES"
    
    def analyze_embedding_quality(self):
        """Check if embeddings are well-formed."""
        print("\n" + "="*80)
        print("7. EMBEDDING QUALITY")
        print("="*80)
        
        try:
            embeddings = np.array(self.item_info['item_emb_d128'].to_list())
            
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Mean: {embeddings.mean():.6f}")
            print(f"Std: {embeddings.std():.6f}")
            print(f"Min: {embeddings.min():.6f}")
            print(f"Max: {embeddings.max():.6f}")
            print(f"Contains NaN: {np.isnan(embeddings).any()}")
            print(f"Contains Inf: {np.isinf(embeddings).any()}")
            
            # Check if embeddings are normalized
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"\nEmbedding norms - mean: {norms.mean():.4f}, std: {norms.std():.4f}")
            
            if embeddings.std() < 0.01:
                print("⚠️  WARNING: Embeddings have very low variance!")
                return "LOW_VARIANCE"
            elif np.isnan(embeddings).any() or np.isinf(embeddings).any():
                print("⚠️  WARNING: Embeddings contain NaN or Inf!")
                return "BAD_EMBEDDINGS"
            else:
                print("✓ Embeddings look healthy")
                return "OK"
        except Exception as e:
            print(f"Error analyzing embeddings: {e}")
            return "ERROR"
    
    def run_full_diagnostic(self):
        """Run all diagnostics and provide summary."""
        print("\n\n" + "="*80)
        print("STARTING FULL DIAGNOSTIC")
        print("="*80 + "\n")
        
        results = {
            "labels": self.analyze_label_distribution(),
            "items": self.analyze_item_overlap(),
            "popularity": self.analyze_item_popularity(),
            "users": self.analyze_user_patterns(),
            "sequences": self.analyze_sequence_patterns(),
            "features": self.analyze_feature_ranges(),
            "embeddings": self.analyze_embedding_quality(),
        }
        
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        
        issues = [k for k, v in results.items() if v != "OK"]
        
        if not issues:
            print("✓✓ All checks passed! Data looks good.")
            print("\nIf model still overfits, the issue is in:")
            print("  1. Model architecture (too large?)")
            print("  2. Learning rate too high")
            print("  3. Batch size too small (noisy gradients)")
            print("  4. Loss function weighting (pos_weight=2.33 might be wrong)")
        else:
            print(f"⚠️  Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}: {results[issue]}")
        
        print("\n" + "="*80)
        return results


# ============================================================================
# RUN DIAGNOSTIC
# ============================================================================

if __name__ == "__main__":
    diagnostic = CTRDataDiagnostic(
        train_path="/kaggle/working/train_with_full_seq.parquet",
        valid_path="/kaggle/working/valid_with_full_seq.parquet",
        item_info_path="/kaggle/input/www2025-mmctr-data/MicroLens_1M_MMCTR/MicroLens_1M_x1/item_info.parquet"
    )
    
    results = diagnostic.run_full_diagnostic()