# Model Inference Instructions

To perform inference using the trained model, follow these steps:

## 1. Load the Necessary Libraries

Make sure you have the required libraries installed. You can do this via pip if you haven't already:

```bash
pip install torch torchvision
```

## 2. Load the Trained Model

First, you'll need to load your trained model. Replace `model_path` with the path to your saved model file.

```python
import torch
from torchvision import models

# Specify the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.resnet50(pretrained=False)  
model.fc = torch.nn.Linear(model.fc.in_features, 37) 
model.load_state_dict(torch.load("model_path.pth")) 
model.to(device)
model.eval() 
```

## 3. Preprocess the Input Data

```python
from torchvision import transforms
from PIL import Image

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load and preprocess the image
image_path = "path_to_your_image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Open and convert the image
image = transform(image).unsqueeze(0)  # Add batch dimension
image = image.to(device)  # Move to the appropriate device
```

## 4. Perform Inference

Now you can perform inference on the preprocessed image.

```python
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(image)  # Forward pass
    _, predicted = torch.max(output, 1)  # Get the predicted class
    print(f"Predicted class: {predicted.item()}")
```
