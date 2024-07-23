import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

def index_images(images):
    print("Indexing images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_features = []
    with torch.no_grad():
        for img in images:
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            features = model(input_tensor)
            image_features.append(features.squeeze().cpu().numpy())

    print(f"Indexed {len(image_features)} images")
    return np.array(image_features)

# Example usage
if __name__ == "__main__":
    # This is just for demonstration. In your actual script, you'll pass the real images.
    sample_images = [Image.new('RGB', (100, 100)) for _ in range(3)]  # Creating dummy images
    features = index_images(sample_images)
    print(f"Feature shape for each image: {features[0].shape}")