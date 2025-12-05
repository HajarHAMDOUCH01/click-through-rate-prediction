import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import os

import sys 
sys.path.append("/content/RS_competition")
from task2.models.dcn_v2 import DCNV2Model
from task2.models.seq_transformer import SequentialFeatLearningModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainCTRPred(nn.Module):
    def __init__(self,
                multimodal_frozen_embeddings,  # numpy: (num_items, 128)
                item_tags_lookup,  # numpy: (num_items, 5)
                num_tags,
                learning_rate,
                tag_embed_dim=16
                ):
        super().__init__()
        
        self.device = device
        
        # Frozen multimodal embeddings (NOT trainable)
        self.register_buffer(
            'multimodal_frozen_embeddings', 
            torch.from_numpy(multimodal_frozen_embeddings).float()
        )
        
        # Static tags lookup (NOT trainable)
        self.register_buffer(
            'item_tags_lookup',
            torch.from_numpy(item_tags_lookup).long()
        )
        
        # Learnable tag embeddings (TRAINABLE)
        self.tag_embedding = nn.Embedding(
            num_embeddings=num_tags,  
            embedding_dim=tag_embed_dim,
            padding_idx=0
        )
        
        # Calculate dimensions
        frozen_dim = multimodal_frozen_embeddings.shape[1]  # 128
        total_item_dim = frozen_dim  # 128 + 64 = 192
        # Initialize models
        self.seq_feature_learning_model = SequentialFeatLearningModel(
            item_embed_dim=total_item_dim  # 192
        )
        
        self.feature_interaction_model = DCNV2Model(
            input_dim=self.seq_feature_learning_model.output_dim+192+32,
            target_item_dim=total_item_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        # Move everything to device
        self.to(device)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        
        # Optimizer and Scheduler
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(2.33)
        )

        print(f"Model initialized on {device}")
        print(f"Frozen embedding dim: {frozen_dim}")
        print(f"Tag embedding dim: {tag_embed_dim}")
        print(f"Total item embedding dim: {total_item_dim}")
        print(f"Learning rate: {learning_rate}")
    
    def get_item_embeddings(self, item_ids):
        """
        Get combined embeddings for items.
        
        Args:
            item_ids: (batch_size, seq_len) or (batch_size,)
        
        Returns:
            Combined embeddings: frozen (128) + learnable tags (32) = 160
        """
        # 1. Get frozen multimodal embeddings
        frozen_embeds = self.multimodal_frozen_embeddings[item_ids]
        
        # 2. Look up tags for these items
        item_tags = self.item_tags_lookup[item_ids]
        
        # 3. Embed tags (learnable) and aggregate
        tag_embeds = self.tag_embedding(item_tags)  # (..., 5, 32)
        tag_embeds = tag_embeds.mean(dim=-2)  # (..., 32)
        
        # 4. Concatenate frozen + learnable
        combined = torch.cat([frozen_embeds, tag_embeds], dim=-1)  # (..., 160)
        fused = self.fusion(combined)
        return fused
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: dict with keys: item_seqs, item_ids, likes_levels, views_levels
        
        Returns:
            predictions: (batch_size,)
        """
        # Extract batch data
        s_items_ids = batch["item_seqs"]  # (B, seq_len)
        pad_len = 100 - s_items_ids.size(1)
        s_items_padded = F.pad(s_items_ids, (pad_len, 0), value=0)
        
        target_items_ids = batch["item_ids"]  # (B,)
        likes_levels = batch["likes_levels"]  # (B,)
        views_levels = batch["views_levels"]  # (B,)
        
        # Get embeddings for sequence items
        s_items_embeds = self.get_item_embeddings(s_items_padded)  # (B, 100, 160)
        
        # Get embeddings for target items
        target_items_embeds = self.get_item_embeddings(target_items_ids)  # (B, 160)
        # print("Target embeds - mean:", target_items_embeds.mean().item(), "std:", target_items_embeds.std().item())
        # Sequential feature learning
        k = 16  # Fixed k from paper
        S_o = self.seq_feature_learning_model(
            s_items_padded, 
            s_items_embeds, 
            target_items_embeds, 
            k
        )
        # print("S_o shape : ", S_o.shape)
        # print("S_o: ", S_o)
        # Feature interaction and prediction
        predictions = self.feature_interaction_model(
            S_o, 
            target_items_embeds,
            likes_levels, 
            views_levels
        )
        # print(predictions)
        return predictions
    
    def compute_auc_on_accumulated(self, all_predictions, all_labels):
        """Helper function to compute AUC from accumulated predictions and labels."""
        if len(all_predictions) == 0 or len(all_labels) == 0:
            return None
        
        preds = torch.cat(all_predictions).numpy()
        labels = torch.cat(all_labels).numpy()
        
        # Apply sigmoid to convert logits to probabilities
        preds = torch.sigmoid(torch.from_numpy(preds)).numpy()
        
        try:
            auc = roc_auc_score(labels, preds)
            return auc
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            return None

    def train_epoch(self, train_dataloader, valid_dataloader=None, validation_interval=1000):
        """Train for one epoch and return metrics with periodic validation checks."""
        self.train()
        
        total_loss = 0.0
        num_batches = 0
        
        all_labels = []
        all_predictions = []

        for i, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            predictions = self.forward(batch)
            labels = batch["labels"]
            predictions = self.forward(batch).squeeze(-1)

            if i % 1000 == 0:
                print(f"\nPredictions - mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
                print(f"Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}")
                print(f"Unique predictions: {len(torch.unique(predictions))}")
                
                # Check if model is just predicting one class
                probs = torch.sigmoid(predictions)
                print(f"Probabilities - mean: {probs.mean():.4f}")
                print(f"Prob > 0.5: {(probs > 0.5).float().mean():.4f}")

                # Add this to your training script
                print("Train positive rate:", labels.mean())
                # criterion =  torch.nn.BCEWithLogitsLoss(
                # pos_weight=torch.tensor([pos_weight]).to(device)
                # )
            loss = self.criterion(predictions, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.detach().cpu())
            all_labels.append(labels.cpu())

            # Periodic logging with AUC calculation
            if i % validation_interval == 0 and i > 0:
                avg_loss_so_far = total_loss / num_batches
                
                # Calculate training AUC on accumulated data so far
                train_auc_so_far = self.compute_auc_on_accumulated(all_predictions, all_labels)
                
                print(f"\n{'='*70}")
                print(f"CHECKPOINT @ Batch {i}/{len(train_dataloader)}")
                print(f"{'='*70}")
                print(f"Train Loss (avg): {avg_loss_so_far:.6f}")
                if train_auc_so_far is not None:
                    print(f"Train AUC (so far): {train_auc_so_far:.4f}")
                
                print(f"{'='*70}\n")
        
        epoch_loss = total_loss / num_batches
        
        # Calculate final training AUC for the full epoch
        train_auc = self.compute_auc_on_accumulated(all_predictions, all_labels)
        
        return epoch_loss, train_auc
        
    def validate_epoch(self, valid_dataloader):
        """Validate and return metrics."""
        self.eval()
        all_labels = []
        all_predictions = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_dataloader, desc='Validation')):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]
                if i % 10 == 0:
                    print("Val positive rate:", labels.mean())
                predictions = self.forward(batch).squeeze(-1)
                # criterion =  torch.nn.BCEWithLogitsLoss(
                # pos_weight=torch.tensor([pos_weight]).to(device)
                # )
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                num_batches += 1

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                if i % 100 == 0:
                    avg_loss_so_far = total_loss / num_batches
                    print(f"Average val loss up to batch {i}: {avg_loss_so_far:.6f}")

        val_loss = total_loss / num_batches
        
        # Calculate validation AUC
        val_auc = self.compute_auc_on_accumulated(all_predictions, all_labels)
            
        return val_loss, val_auc
    
    def evaluate_epoch_performance(self, train_auc, val_auc, epoch_num):
        """
        Evaluate and print epoch performance with health checks.
        
        Tests:
        1. Training AUC should be > 0.65 (minimum acceptable)
        2. Validation AUC should be close to training AUC (gap < 0.05)
        """
        print("\n" + "="*70)
        print(f"EPOCH {epoch_num} PERFORMANCE EVALUATION")
        print("="*70)
        
        if train_auc is None or val_auc is None:
            print("⚠️  ERROR: Could not calculate AUC scores!")
            return "error"
        
        print(f"Training AUC:   {train_auc:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"AUC Gap:        {abs(train_auc - val_auc):.4f}")
        
        # Test 1: Minimum training performance
        print("\n" + "-"*70)
        print("TEST 1: Minimum Training Performance (AUC > 0.65)")
        print("-"*70)
        if train_auc > 0.65:
            print(f"✓ PASS: Training AUC ({train_auc:.4f}) > 0.65")
            test1_pass = True
        else:
            print(f"✗ FAIL: Training AUC ({train_auc:.4f}) ≤ 0.65")
            print("  Recommendation: Model is underfitting. Consider:")
            print("    - Increasing learning rate")
            print("    - Training for more epochs")
            print("    - Checking data quality")
            test1_pass = False
        
        # Test 2: Generalization check
        auc_gap = abs(train_auc - val_auc)
        print("\n" + "-"*70)
        print("TEST 2: Generalization Check (AUC gap < 0.05)")
        print("-"*70)
        if auc_gap < 0.05:
            print(f"✓ PASS: AUC gap ({auc_gap:.4f}) < 0.05")
            print("  Model generalizes well to validation data")
            test2_pass = True
        else:
            print(f"✗ FAIL: AUC gap ({auc_gap:.4f}) ≥ 0.05")
            if train_auc > val_auc:
                print("  ⚠️  Model is OVERFITTING. Recommendations:")
                print("    - Increase dropout rate")
                print("    - Add more regularization (weight_decay)")
                print("    - Reduce model complexity")
                print("    - Get more training data")
            else:
                print("  ⚠️  Validation AUC > Training AUC (unusual)")
                print("  This might indicate:")
                print("    - Training set is harder than validation")
                print("    - Data leakage issues")
            test2_pass = False
        
        # Overall assessment
        print("\n" + "="*70)
        print("OVERALL ASSESSMENT")
        print("="*70)
        
        if test1_pass and test2_pass:
            print("✓✓ EXCELLENT: Both tests passed!")
            print("   Continue training - model is learning well")
            status = "excellent"
        elif test1_pass and not test2_pass:
            print("⚠️  CAUTION: Model learns but doesn't generalize well")
            print("   Consider adding regularization")
            status = "caution"
        elif not test1_pass and test2_pass:
            print("⚠️  CAUTION: Model generalizes but performance is low")
            print("   Consider training longer or increasing capacity")
            status = "caution"
        else:
            print("✗✗ POOR: Both tests failed")
            print("   Model needs significant improvements")
            status = "poor"
        
        print("="*70 + "\n")
        
        return status
        
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if it exists in checkpoint
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        train_loss = checkpoint.get('train_loss', 0.0)
        print(f"Loaded checkpoint from epoch {epoch+1} with train_loss: {train_loss:.4f}")
        return epoch
        
    def train_model(self, train_dataloader, valid_dataloader, num_epochs=10, 
                    save_path=None, checkpoint_dir=None, checkpoint_path=None):
        """Complete training loop with validation and performance evaluation."""
        start_epoch = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            start_epoch = self.load_checkpoint(checkpoint_path)
            print(f"Resuming from epoch {start_epoch+1}")
        
        best_val_auc = 0.0
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Training with periodic validation checks
            train_loss, train_auc = self.train_epoch(train_dataloader, valid_dataloader)
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch+1} TRAINING COMPLETE")
            print(f"{'='*50}")
            print(f"Train Loss: {train_loss:.4f}")
            if train_auc is not None:
                print(f"Train AUC:  {train_auc:.4f}")
            
            # Full validation
            val_loss, val_auc = None, None
            if valid_dataloader is not None:
                print(f"\nRunning full validation...")
                val_loss, val_auc = self.validate_epoch(valid_dataloader)
                print(f"\nVal Loss: {val_loss:.4f}")
                if val_auc is not None:
                    print(f"Val AUC:  {val_auc:.4f}")
                
                # Performance evaluation with tests
                status = self.evaluate_epoch_performance(train_auc, val_auc, epoch+1)
                
                # Step the scheduler based on validation loss
                if val_auc is not None:
                    self.scheduler.step(val_auc)
                
                # Save best model based on validation AUC
                if val_auc is not None and val_auc > best_val_auc:
                    best_val_auc = val_auc
                    if save_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'train_loss': train_loss,
                            'train_auc': train_auc,
                            'val_loss': val_loss,
                            'val_auc': val_auc,
                        }, save_path)
                        print(f"✓ Saved best model (Val AUC: {val_auc:.4f}): {save_path}")
            
            # Save epoch checkpoint
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'train_auc': train_auc if train_auc is not None else 0.0,
                'val_loss': val_loss if val_loss is not None else 0.0,
                'val_auc': val_auc if val_auc is not None else 0.0,
            }, epoch_checkpoint_path)
            print(f"✓ Saved checkpoint: {epoch_checkpoint_path}")
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        if best_val_auc > 0:
            print(f"Best Validation AUC: {best_val_auc:.4f}")
        print("="*50)