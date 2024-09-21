import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        
        # Load pretrained ResNet50
        resnet50_base = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layer and keep the feature extraction layers
        self.resnet50_features = nn.Sequential(*list(resnet50_base.children())[:-2])
        self.resnet50_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize a dummy input to get the output size of the feature extractor
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            x = self.resnet50_features(dummy_input)
            x = self.resnet50_pool(x)
            self.resnet50_out_features = x.numel()
        
        # Define the custom layers
        self.resnet50_custom = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.resnet50_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet50_features(x)
        x = self.resnet50_pool(x)
        x = self.resnet50_custom(x)
        return x

class CustomResNet34(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet34, self).__init__()
        
        # Load pretrained ResNet34
        resnet34_base = models.resnet34(pretrained=True)
        
        # Remove the final fully connected layer and keep the feature extraction layers
        self.resnet34_features = nn.Sequential(*list(resnet34_base.children())[:-2])
        self.resnet34_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize a dummy input to get the output size of the feature extractor
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            x = self.resnet34_features(dummy_input)
            x = self.resnet34_pool(x)
            self.resnet34_out_features = x.numel()
        
        # Define the custom layers
        self.resnet34_custom = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.resnet34_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet34_features(x)
        x = self.resnet34_pool(x)
        x = self.resnet34_custom(x)
        return x
