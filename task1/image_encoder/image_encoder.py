from transformers import CLIPProcessor, CLIPModel
import torch

class ItemEncoder:
    def __init__(self,
                device,
                encoder_model_name: str = "openai/clip-vit-base-patch32",
    ):
        self.device = device
        self.model = CLIPModel.from_pretrained(encoder_model_name)  
        self.processor = CLIPProcessor.from_pretrained(encoder_model_name)  

        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    @torch.no_grad()
    def encode_items(self, images, titles):
        """
        Args:
            images: PIL Image or list of PIL Images
            titles: str or list of str
        Returns:
            torch.Tensor: item embeddings [batch_size, hidden_dim]
        """
        inputs = self.processor(text=titles, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds  # [1, 512]
            text_embeds = outputs.text_embeds     # [1, 512]
            
            # Combine them (concatenate or average)
            item_embedding = torch.cat([image_embeds, text_embeds], dim=1)
        
        
        item_embedding = torch.cat([image_embeds, text_embeds], dim=1)