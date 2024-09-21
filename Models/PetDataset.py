import os
from PIL import Image
from torch.utils.data import Dataset


images_file =  'D:/Jobs/Intixel/The-Oxford-IIIT-Pet/Tests'  # Example path

class PetDataset(Dataset):
    # Constructor to initialize the dataset
    def __init__(self, data, transform=None):
        """
        Args:
        - data: A pandas DataFrame containing image names, class IDs, species, and breed IDs.
        - img_dir: Directory where the images are stored.
        - transform]: A function/transform to apply to the images.
        """
        self.data = data  # Store the dataset metadata (image names, class IDs, etc.)
        self.img_dir = images_file  # Store the directory where the images are located
        self.transform = transform  # Store any transformations to be applied to the images

    # Method to return the total number of items in the dataset
    def __len__(self):
        """
        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.data)  # Return the number of rows in the dataframe (number of images)

    # Method to load and return a single sample from the dataset
    def __getitem__(self, idx):
        """
        Args:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - tuple: (image, label), where:
            - image (PIL Image or Tensor): The loaded image (with transformations applied if given).
            - label (int): The corresponding class label (adjusted to be 0-based).
        """
        # Retrieve the image file name (without extension) from the dataset and construct the full path
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')

        # Open the image and convert it to RGB format (in case it's grayscale or other formats)
        image = Image.open(img_name).convert('RGB')

        # Retrieve the class ID from the dataset and adjust it to be zero-indexed (subtract 1)
        # The dataset class IDs are 1 to 37, so I subtract 1 to make it 0 to 36
        label = self.data.iloc[idx, 1] - 1

        # If a transformation is provided, apply it to the image
        if self.transform:
            image = self.transform(image)

        # Return the transformed image and the corresponding label
        return image, label
