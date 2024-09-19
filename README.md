# The-Oxford-IIIT-Pet
## Overview
This project uses the Oxford-IIIT Pet Dataset to build a machine learning model for multiclass classification of pet breeds. The original paper titled ["Cats and Dogs"](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/) (Parkhi et al., 2012) reports an average accuracy of about 59% on this challenging task.

However, through the application of deep learning techniques and fine-tuning of pretrained models such as ResNet34 and ResNet50, I have achieved a significantly higher accuracy of 93%, which greatly surpasses the original paper's results.



## Dataset description
The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200 images for each class created by the Visual Geometry Group at Oxford. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation. [[kaggle.com](https://www.robots.ox.ac.uk/~vgg/data/pets/)]

Original dataset was downloaded from [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/).

![image](https://github.com/user-attachments/assets/55957e1c-adb4-4309-a561-63b77d5860fe)

The following annotations are available for every image in the dataset: (a) species and breed name; (b) a tight bounding box (ROI) around the head of the animal; and (c) a pixel level foreground-background segmentation (Trimap).

![image](https://github.com/user-attachments/assets/ae31a3b0-805f-405e-8add-dd1c21b2f43a)



## Models Used
The project leverages pretrained models, including ResNet50 and ResNet34, and customizes them with additional fully connected layers to adapt to a specific classification task.

### ResNet-34
The [ResNet-34](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html#torchvision.models.resnet34) model is a deep convolutional neural network that is part of the ResNet (Residual Network) family. Pre-trained on the ImageNet-1K dataset, ResNet-34 features 34 layers and employs residual connections to address the vanishing gradient problem often encountered in deep networks. This architecture enables effective training of deeper networks by facilitating the flow of gradients through residual connections, which helps in learning more complex features and improves overall performance on image classification tasks.

### ResNet-50
The [ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) model extends the ResNet architecture to 50 layers, incorporating residual blocks and bottleneck layers for enhanced computational efficiency. This deeper configuration leverages residual connections to mitigate gradient issues, while the bottleneck design reduces computational overhead by compressing the feature maps. Pre-trained on the ImageNet-1K dataset, ResNet-50 is well-suited for transfer learning and fine-tuning, making it highly effective for complex classification tasks like those encountered in the Oxford-IIIT Pet Dataset.


## Optimizers and Learning Rates used: 
Adam: An adaptive learning rate optimizer that combines the advantages of two other extensions of stochastic gradient descent. Adam adjusts the learning rate based on the average of recent gradients, making it well-suited for handling sparse gradients and varying learning rates.

SGD (Stochastic Gradient Descent): A classic optimizer that updates the model parameters using the gradient of the loss function with respect to the parameters. When used with momentum, it helps accelerate convergence and navigate through local minima more effectively.


## Performance Metrics used: 
The performance is evaluated based on accuracy, loss, and validation metrics to ensure that it effectively classifies pet breeds and generalizes well to unseen data. Additionally, a confusion matrix is used to assess the model's ability to correctly classify each pet breed and identify any common misclassifications. 


## Folder Structure

### `models/`

This folder contains the code for defining and implementing the ResNet models used in this project.

- **`resnet_models.py`**: Defines custom ResNet models, including modifications to ResNet34 and ResNet50 architectures. The models include feature extraction layers from pre-trained ResNet architectures followed by custom classification layers.

### `weights/`

This folder stores the trained model weights.

- **`best_model.pth`**: The file where the weights of the best-performing model are saved.
- 




## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/MarinaNasser/The-Oxford-IIIT-Pet.git
    ```
2. Install the required packages:

- Python 3.6 or higher
- PyTorch 1.7.0 or higher
- torchvision 0.8.0 or higher
- matplotlib
- NumPy
- Pandas
- PIL (Pillow)


You can install the necessary packages using pip:

```
pip install torch torchvision matplotlib numpy pandas pillow
```

### Usage

 **Load and Use a Model**: Load a saved model from the `weights/` folder and use it for inference.

    
    from models.resnet_models import CustomResNet34, CustomResNet50
    import torch

    model = CustomResNet50(num_classes=10)  # Adjust the number of classes as needed
    model.load_state_dict(torch.load('weights/best_model.pth'))
    





#### Contact
For any questions or contributions, please open an issue or contact me at marinanasser8@gmail.com.

