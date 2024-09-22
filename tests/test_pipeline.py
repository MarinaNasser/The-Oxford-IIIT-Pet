# test_pipeline.py
import unittest
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
from Models.ResNet import CustomResNet50 , CustomResNet34
from Models.train import train_model
from Models.test import test_model
from preprocessing import load_image_class_mapping
from Models.PetDataset import PetDataset  
from Models.EfficientNet import get_pretrained_efficientnet_b0, get_custom_efficientnet_b0


class TestPetBreedClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Executing setUpClass...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.model = get_pretrained_efficientnet_b0(num_classes=37)  # Replace with your actual model class
        weights_path = os.path.join(base_dir, '../weights/EfficientNet_best_model.pth')
        
        # Load weights with CPU mapping
        if os.path.exists(weights_path):
            print(f"Loading model from: {weights_path}")
            cls.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            cls.model.eval()  # Set the model to evaluation mode
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        # Define image paths and class IDs
        images_folder = os.path.join(base_dir, 'images')
        list_file = os.path.join(base_dir, 'list.txt')
        image_class_mapping = load_image_class_mapping(list_file, images_folder)
        image_paths = list(image_class_mapping.keys())
        class_ids = list(image_class_mapping.values())

        # Create a DataFrame for the dataset
        data = pd.DataFrame({
            'image_path': image_paths,
            'class_id': class_ids
        })

        # Load pretrained weights for ResNet34
        ResNet34_weights = models.resnet.ResNet34_Weights.DEFAULT

        # Get the preprocessing transforms required by the model
        cls.transform = ResNet34_weights.transforms()
        print("Transform defined.")

        # Create dataset and DataLoader
        dataset = PetDataset(data, img_dir=images_folder, transform=cls.transform)
        cls.train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        cls.test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Initialize criterion and optimizer
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=0.001)

    def setUp(self):
        # Ensure that setUpClass has been called and attributes are set
        if not hasattr(self, 'train_loader'):
            self.fail("setUpClass did not set train_loader attribute.")
        if not hasattr(self, 'test_loader'):
            self.fail("setUpClass did not set test_loader attribute.")
        if not hasattr(self, 'criterion'):
            self.fail("setUpClass did not set criterion attribute.")
        if not hasattr(self, 'optimizer'):
            self.fail("setUpClass did not set optimizer attribute.")

    def test_train_model(self):
        """Test the training function."""
        print("Executing test_train_model...")
        train_losses, train_accuracies = train_model(
            self.model, self.criterion, self.optimizer, self.train_loader, num_epochs=15
        )
        self.assertIsInstance(train_losses, list, "train_losses should be a list.")
        self.assertIsInstance(train_accuracies, list, "train_accuracies should be a list.")
        self.assertGreater(len(train_losses), 0, "train_losses should not be empty.")
        self.assertGreater(len(train_accuracies), 0, "train_accuracies should not be empty.")

    def test_test_model(self):
        """Test the testing function."""
        print("Executing test_test_model...")
        loss, test_accuracy, all_test_preds, all_test_labels = test_model(self.model, self.test_loader)
        
        # Print the test loss and accuracy
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.2f}")
        
        self.assertIsInstance(loss, float, "Loss should be a float.")
        self.assertIsInstance(test_accuracy, float, "Test accuracy should be a float.")
        self.assertIsInstance(all_test_preds, np.ndarray, "all_test_preds should be a numpy array.")
        self.assertIsInstance(all_test_labels, np.ndarray, "all_test_labels should be a numpy array.")
        self.assertEqual(len(all_test_preds), len(all_test_labels), "Predictions and labels should have the same length.")


if __name__ == '__main__':
    unittest.main()
