import torch
from transformers import BertTokenizer, BertModel

class TextEncoder:
    def __init__(self,
                device,
                encoder_model_name:str,
    ):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(encoder_model_name)
        self.model = BertModel.from_pretrained(encoder_model_name)

        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    @torch.no_grad()
    def encode_titles(self, titles):
        """
        Args:
            image: PIL Image or list of PIL Images
        Returns:
            torch.Tensor: Image embeddings [batch_size, hidden_dim]
        """
        inputs = self.tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=128, return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        return outputs.pooler_output