import torch
import torch.nn as nn 
import torch.nn.functional as F

class DCNV2Model(nn.Module):
    def __init__(self, input_dim, target_item_dim, e_side_proj_dim=16, deep_sizes=[256, 128], num_cross_layers=3, dropout_rate=0.2):
        super().__init__()
        self.item_embed_dim = target_item_dim
        self.e_side_proj_dim = e_side_proj_dim
        self.s_o_dim = 256
        self.x0_dim = self.item_embed_dim+self.s_o_dim+self.e_side_proj_dim

        self.dense_proj = nn.Linear(2, e_side_proj_dim, bias=False)
        
        # Cross layers with dropout
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.x0_dim, self.x0_dim, bias=True),
                nn.Dropout(p=dropout_rate)
            )
            for _ in range(num_cross_layers)
        ])

        # Deep network with dropout after each activation
        layers = []
        input_dim = self.x0_dim
        for h in deep_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = h
        self.deep_network = nn.Sequential(*layers)

        final_dim = self.x0_dim + deep_sizes[-1]
        self.prediction = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Kaiming initialization for ReLU networks."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'weight' in name:
                    # Kaiming initialization for ReLU
                    nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.zeros_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)  # Bias terms
            
            # For cross layers specifically, you might want different initialization
            if 'cross_layers' in name and 'weight' in name:
                # Cross layers benefit from smaller initialization
                nn.init.normal_(p, mean=0.0, std=0.01)

    def cross_forward(self, x0, x):
        for layer in self.cross_layers:
            cross = x0 * layer(x)        
            x = cross + x                # Residual connection
        return x
    

    def forward(self, S_o, target_items_embeds, target_items_likes, target_items_views):
        """
        Args : 
            S_o : (batch_size , S_o)
            target_items_ids : (batch_size, target_items_ids)
            target_items_likes : (batch_size, target_items_likes)
            target_items_views : (batch_size, target_items_views)

        outputs :
            prediction : the probability of the target item being clicked
        """
        normalized_items_likes = (target_items_likes - 1.0) / 9.0  
        normalized_items_views = (target_items_views - 1.0) / 9.0
        e_dense = torch.stack([normalized_items_likes, normalized_items_views], dim=-1)
        
        # print("S_o : ", S_o)


        e_side = self.dense_proj(e_dense) 

        # for every entry (user_id) in the training set , there is a target_id , and label , wether that user clicked on the target_id item or not
        e_target = target_items_embeds

        # this is x_0 at the input of the cross layers
        x0 = torch.cat([e_target,e_side, S_o], dim=-1)
        # x0 = F.normalize(x0, p=2, dim=1)
        # parallel 
        x_cross = self.cross_forward(x0, x0)
        x_deep = self.deep_network(x0)

        f_combined = torch.cat([x_cross, x_deep], dim=-1)

        y = self.prediction(f_combined)
        return y