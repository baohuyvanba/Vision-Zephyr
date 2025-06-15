"""
Utility functions: tokenization helpers, stopping criteria, etc.
"""
import torch
from transformers import StoppingCriteria


class KeywordsStoppingCriteria(StoppingCriteria):
    """Terminate generation upon encountering specified keywords in output tokens."""
    def __init__(self, keywords, tokenizer, input_ids):
        super().__init__()
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        for kw in self.keywords:
            if kw in text:
                return True
        return False

def preprocess_image(image_path, image_processor):
    """Load and preprocess image using given image_processor."""
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    processed = image_processor(images=image, return_tensors='pt')
    return processed['pixel_values']