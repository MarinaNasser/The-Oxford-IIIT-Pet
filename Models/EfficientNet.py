import torch
import torch.nn as nn
from torchvision import models

# Define a function to load the pretrained EfficientNetB0 model
def get_pretrained_efficientnet_b0(num_classes=37):
    # Load pretrained EfficientNetB0 model
    efficientnet_b0 = models.efficientnet_b0(pretrained=True)
    
    # Modify the classifier to match the number of classes
    efficientnet_b0.classifier[1] = nn.Linear(in_features=efficientnet_b0.classifier[1].in_features, out_features=num_classes)
    
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    efficientnet_b0 = efficientnet_b0.to(device)
    
    # Print to confirm the model architecture
    print(efficientnet_b0)
    
    return efficientnet_b0

# Define a function to create the custom EfficientNetB0 model
def get_custom_efficientnet_b0(num_classes=37):
    # Load pretrained EfficientNetB0 model
    efficientnet = models.efficientnet_b0(pretrained=True)

    # Remove the classifier to keep only the feature extraction layers
    efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-2])

    # Pooling layer to adapt the output to a fixed size (e.g., (1, 1))
    efficientnet_pool = nn.AdaptiveAvgPool2d((1, 1))

    # Initialize a dummy input to get the output size of the feature extractor
    dummy_input = torch.zeros(1, 3, 224, 224)  # Adjust input size if necessary
    with torch.no_grad():
        x = efficientnet_features(dummy_input)
        x = efficientnet_pool(x)
        out_features = x.numel()  # Get the number of features after pooling

    print(f"EfficientNetB0 output features size: {out_features}")

    # Define the custom layers
    efficientnet_custom = nn.Sequential(
        nn.Flatten(),  # Flatten the output from the pooling layer
        nn.Linear(out_features, 512),  # First custom fully connected layer
        nn.ReLU(),  # Activation function
        nn.Dropout(0.5),  # Dropout layer for regularization
        nn.Linear(512, num_classes)  # Final output layer
    )

    # Combine the feature extractor and custom layers into the final model
    efficientnet_b0_model = nn.Sequential(
        efficientnet_features,
        efficientnet_pool,
        efficientnet_custom
    )

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    efficientnet_b0_model.to(device)

    # Print to confirm the model architecture
    print(efficientnet_b0_model)
    
    return efficientnet_b0_model
