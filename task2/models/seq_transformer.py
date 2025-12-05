import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class SequentialFeatLearningModel(nn.Module):
    def __init__(self, item_embed_dim):
        super().__init__()
        self.output_dim = 16*256
        self.encoder_layer = TransformerEncoderLayer(
            d_model=item_embed_dim*2, 
            nhead=4, 
            dim_feedforward=256, 
            batch_first=True,
            dropout=0.2,  
            
        )
        self.compressed_dim = 256
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer, 
            num_layers=1
        )
        self.output_norm = nn.LayerNorm(self.output_dim)
        self._init_weights()

        self.compression = nn.Sequential(
            nn.Linear(256, self.compressed_dim),
            nn.LayerNorm(self.compressed_dim),
            nn.GELU("tanh")
        )

    def _init_weights(self):
      """Initialize transformer weights with proper scaling."""
      for name, p in self.named_parameters():
          if p.dim() > 1:
              if 'weight' in name:
                  if 'self_attn' in name:
                      nn.init.xavier_uniform_(p, gain=1.0)
                  elif 'linear1' in name:
                      nn.init.xavier_uniform_(p, gain=1.0)
                  elif 'linear2' in name:
                      nn.init.xavier_uniform_(p, gain=1.0)
              elif 'bias' in name:
                  nn.init.zeros_(p)

    def forward(self, s_item_ids, s_items_embeds, target_items_embeddings, k):
        """
        Args 
            s : (N, s_i) and N is 100 (with padding)
            target_id : (batch_size,) - target item_id
            s_i : item embeddings in the sequence || target item embeddings
            k : int - number of recent items
        outputs
            y : 𝑆𝑜 = Flatten(𝑠1,𝑠2,. . .,𝑠𝑘,MaxPool (𝑆))
        """
        target_embeds = target_items_embeddings.unsqueeze(1)
        target_embeds = target_embeds.expand(-1, 100, -1)
        s = torch.cat([s_items_embeds, target_embeds], dim=-1) 
        padding_mask = (s_item_ids == 0)
        S = self.transformer_encoder(s, src_key_padding_mask=padding_mask)
        s_k = S[:, -k:, :].flatten(1)
        # s_max = S.max(dim=1)[0]
        # S_o = torch.cat([s_k, s_max], dim=1)
        S_o = self.compression(s_k)
        # S_o = self.output_norm(S_o)
        # print("S_o stats - mean:", S_o.mean().item(), "std:", S_o.std().item(), "min:", S_o.min().item(), "max:", S_o.max().item())
        return S_o
