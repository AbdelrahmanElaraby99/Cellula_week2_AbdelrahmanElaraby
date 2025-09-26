# imagecaption.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


class ImageCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the BLIP model and processor.
        Args:
            model_name: Name of the BLIP model from Hugging Face Hub.
        """
        print("Loading processor...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        print("Processor loaded!")

        print("Loading model...")
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        print("Model loaded!")

    def caption(self, image_path: str, prompt: str = None) -> str:
        """
        Generate a caption for an image.
        Args:
            image_path: Path to the image file.
            prompt: (Optional) A text prompt to guide captioning.
        Returns:
            str: Generated caption.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=120)

        return self.processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

