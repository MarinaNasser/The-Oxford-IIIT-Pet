# Multiclass Classification of Pet Breeds

## 1. Project Overview
This project aims to develop a machine learning model that performs multiclass classification of pet breeds using the Oxford-IIIT Pet Dataset. The dataset contains images of various cat and dog breeds, labeled with their respective classes. The goal is to accurately classify the breeds based on input images.

The original paper titled "Cats and Dogs" (Parkhi et al., 2012) reports an average accuracy of about 59% on this challenging task.
However, through the application of deep learning techniques and fine-tuning of pretrained models such as ResNet34, ResNet50 and EfficientNetB0, I have achieved a significantly higher accuracy of 93.13%,, which greatly surpasses the original paper's results.

## 2. Data Preprocessing Steps
### 2.1 Dataset Description
The Oxford-IIIT Pet Dataset consists of 37 breeds (including both cats and dogs) with a total of 7,349 images. Each image is annotated with a class ID, species, and breed ID.
- ID: 1:37 Class ids
- SPECIES: 1:Cat 2:Dog
- BREED ID: 1-25:Cat 1:12:Dog
- All images with 1st letter as captial are cat images 
- images with small first letter are dog images


### 2.2 Data Loading
- The dataset was loaded from [[kaggle.com](https://www.robots.ox.ac.uk/~vgg/data/pets/)]
- A DataFrame was created to manage the data effectively, containing columns for Image, Class ID, Species, and Breed ID extracted from annotations/list.txt file.
- Data types for Class ID, Species, and Breed ID were converted to integers.

### 2.3 Dataset Splitting
- The dataset was split into training, validation, and test sets using an 80-10-10 ratio to ensure a representative evaluation of the model.

## 3. Model Architecture Decisions
### 3.1 Literature Review

The primary paper associated with the Oxford-IIIT Pet Dataset describes a machine learning model that classifies pet breeds from images, combining two main components: 

- **Shape Model:** 
  - A deformable part model captures the pet's shape by representing it with a root part connected to smaller parts (like the pet's face) using springs. This model employs HOG (Histogram of Oriented Gradients) filters to effectively capture the local distribution of image edges, enabling robust shape detection.

- **Appearance Model:**
  - The appearance of the pet's fur is characterized using a bag-of-words model, which aids in describing the visual features of the fur.

Additionally, the model automatically segments the pet from the background to enhance classification accuracy. Two classification approaches are compared:
- A hierarchical approach, where the pet is first classified into a family (cat or dog) and then into a breed.
- A flat approach, where the breed is predicted directly from the image.

#### Rationale for Deep Learning Models

The original paper achieved an accuracy of approximately 59%. Given the advancements in deep learning, I hypothesized that utilizing modern architectures like ResNet would yield significantly better results. Deep learning models, particularly convolutional neural networks (CNNs), are designed to automatically learn complex features and hierarchical representations from images, allowing them to outperform traditional methods. 

In preparation for my multiclass classification project on the Oxford-IIIT Pet Dataset, I identified two relevant notebooks:
  

1. **ResNet34 Model**  
   - **Source:** [GitHub Repository](https://github.com/limalkasadith/OxfordIIITPet-classification/blob/main/Fine-tune-ResNet34.ipynb)  
   - **Configuration:**  
        - **Model:** ResNet34 +  Flatten layer + 2 Dense layer
        - **Optimizer:** SGD
        - **Learning Rate:** 0.015
        - **Loss Function:** Cross-Entropy
        - **Epochs:** 500

    
   - **Results:**  
     - Training Accuracy: 0.99932  
     - Validation Accuracy: 0.92391  
     - Test Accuracy: 90.379%



2. **EfficientNetB0 Model**  
    - **Source:** [Kaggle Notebook](https://www.kaggle.com/code/lizhensheng/cat-dog)  
    - **Configuration:**  
        - **Model:** EfficientNetB0 pretrained model
        - **Optimizer:** Adam
        - **Learning Rate:** 0.00005
        - **Loss Function:** Cross-Entropy
        - **Epochs:** 40
    
     - **Results:**  
       - Training Accuracy: 0.9953  
       - Validation Accuracy: 0.9341  
       - Test Accuracy: not specified
     

### 3.2. Models Used
The project leverages pretrained models, including ResNet50, ResNet34 and EfficientNet and customizes them with additional fully connected layers to adapt to a specific classification task.

#### ResNet-34
The [ResNet-34](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html#torchvision.models.resnet34) model is a deep convolutional neural network that is part of the ResNet (Residual Network) family. Pre-trained on the ImageNet-1K dataset, ResNet-34 features 34 layers and employs residual connections to address the vanishing gradient problem often encountered in deep networks. This architecture enables effective training of deeper networks by facilitating the flow of gradients through residual connections, which helps in learning more complex features and improves overall performance on image classification tasks.

#### ResNet-50
The [ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) model extends the ResNet architecture to 50 layers, incorporating residual blocks and bottleneck layers for enhanced computational efficiency. This deeper configuration leverages residual connections to mitigate gradient issues, while the bottleneck design reduces computational overhead by compressing the feature maps. Pre-trained on the ImageNet-1K dataset, ResNet-50 is well-suited for transfer learning and fine-tuning, making it highly effective for complex classification tasks like those encountered in the Oxford-IIIT Pet Dataset.

#### EfficientNetB0
The [EfficientNetB0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0) model is part of the EfficientNet family, which is designed using a compound scaling method that uniformly scales network depth, width, and resolution. This balanced scaling approach allows EfficientNetB0 to achieve high performance with fewer parameters and computational resources compared to traditional architectures. Pre-trained on the ImageNet-1K dataset, EfficientNetB0 leverages advanced training techniques and regularization methods to enhance generalization and robustness. Its efficient architecture and pretraining make it highly effective for transfer learning and fine-tuning, enabling superior performance on complex classification tasks like those in the Oxford-IIIT Pet Dataset.

  
 
### 3.3. Trials Overview

Throughout my trials, I aimed to explore various models and hyperparameters to identify the most effective configuration for this classification task. 

- **Initial Trials:**
  - I began with ResNet34 based on the promising results from the literature.
  - My experiments revealed that although ResNet34 showed good initial performance, I wanted to push for higher accuracy by also testing ResNet50 and EfficientNetB0.
  - The final layers of the pretrained models were modified to match the number of classes in the dataset (37 breeds).
  - Based on literature, A custom fully connected layers was added to adapt the output features from the pretrained models.


- **Further Trials:**
  - I varied learning rates and optimizers (Adam and SGD) to assess their impact on model performance.
  - I also Tried the EffiecientNetB0 Model although the notebook found does not specify the test accuracy to compare my results with, i wanted to give a try.
  

### 3.3 Training Configuration
- The models were trained using the Adam optimizer and SGD, as indicated in the literature, along with Cross-Entropy Loss as the loss function, given their effectiveness in multiclass classification tasks.
- Various learning rates were tested, including 0.01, 0.001, 0.0001, 0.00001, and 0.00005, to determine the most suitable value for the training process.

## 4. Training Process & Validation
### 4.1 Training Loop
- The training process consisted of 10 epochs (if i had an ulimited colab GPU units i would have increased this number), with each epoch involving both training and validation phases.
- Training involved forward passes, loss calculation, backpropagation, and optimization steps.
- Validation was conducted after each epoch to monitor the accuracy.

### 4.2 Performance Metrics
- Metrics were collected after each validation phase to evaluate model performance and guide hyperparameter tuning.
- Metrics included training and validation losses, training and validation accuracies.

### 4.3 Trials overview
## Model Trials and Results

### 1st Trial 
I tried to replicate the results of the notebook first, the notebook trained on 500 epochs but i trained on only 10 epochs
- **Model:** ResNet34
- **Epochs:** 10
- **Optimizer:** Adam
- **Learning Rate:** 0.015
- **Train Loss:** 0.45376
- **Train Accuracy:** 0.85477
- **Validation Loss:** 0.31677
- **Validation Accuracy:** 0.89796
- **Test Accuracy:** 89.864%

### 2nd Trial 
I wanted to push for higher accuracy so i tried the ResNet50 Pretrained model.

### Using Adam Optimizer

| Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------|--------|---------------|-------------------|-----------------|---------------------|
| 0.01          | 10     | 3.1376        | 0.1180            | 3.1836          | 0.1241              |
| 0.001         | 10     | 1.8657        | 0.4410            | 2.5026          | 0.3053              |
| 0.0001        | 10     | 0.9964        | 0.7078            | 2.4638          | 0.3571              |
| 5e-05         | 10     | 0.7344        | 0.8029            | 2.6909          | 0.3350              |
| 1e-05         | 10     | 0.6357        | 0.8237            | 2.7004          | 0.3401              |

### Using SGD Optimizer

| Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------|--------|---------------|-------------------|-----------------|---------------------|
| 0.01          | 10     | 2.9434        | 0.1754            | 3.0361          | 0.1641              |
| 0.001         | 10     | 1.5218        | 0.5637            | 2.3295          | 0.3401              |
| 0.0001        | 10     | 0.9237        | 0.7346            | 2.4748          | 0.3512              |
| 5e-05         | 10     | 0.6836        | 0.8080            | 2.6580          | 0.3393              |
| 1e-05         | 10     | 0.6363        | 0.8288            | 2.6932          | 0.3350              |

---

### 3rd Trial  (Custom Model - ResNet34)

#### Using Adam Optimizer

| Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------|--------|---------------|-------------------|-----------------|---------------------|
| 5e-05         | 10     | 0.0562        | 0.9862            | 0.4123          | 0.8895              |
| 1e-05         | 10     | 0.0039        | 1.0000            | 0.3556          | 0.9252              |

#### Using SGD Optimizer

| Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------|--------|---------------|-------------------|-----------------|---------------------|
| 5e-05         | 10     | 0.0356        | 0.9921            | 0.3749          | 0.9039              |
| 1e-05         | 10     | 0.0056        | 0.9991            | 0.3478          | 0.9286              |
 
### (Custom Model - ResNet50)

#### Using Adam Optimizer

| Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------|--------|---------------|-------------------|-----------------|---------------------|
| 5e-05         | 10     | 0.0611        | 0.9866            | 0.3612          | 0.9022              |
| 1e-05         | 10     | 0.0072        | 0.9983            | 0.3062          | 0.9294              |

#### Using SGD Optimizer

| Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------|--------|---------------|-------------------|-----------------|---------------------|
| 5e-05         | 10     | 0.0413        | 0.9911            | 0.3292          | 0.9099              |
| 1e-05         | 10     | 0.0050        | 0.9994            | 0.2930          | 0.9252              |


### 4th Trial (EfficientNetB0)

#### Using Adam Optimizer
| Model               | Learning Rate | Epochs | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|---------------------|---------------|--------|---------------|-------------------|-----------------|---------------------|
| EfficientNetB0       | 5e-05         | 8/15   | 0.0962        | 0.9821            | 0.2170          | 0.9362              |
| Custom EfficientNetB0| 5e-05         | 11/15  | 0.0357        | 0.9928            | 0.2401          | 0.9362              |

---

## 5. Evaluation
### 5.1 Testing
- The model was evaluated on a separate test set to gauge its performance on unseen data.
- Key metrics, including test loss and accuracy, precision, recall, and F1-score for each class were computed to assess model generalization.

### 5.2 Results Analysis
- Confusion matrices were generated to analyze misclassifications and identify which breeds were most frequently confused with one another.

## 6. Conclusion

1. **ResNet34 vs. Traditional Machine Learning Solutions:**
   Using ResNet34 yielded significantly higher accuracy compared to the machine learning solution presented in the original paper. This improvement can be attributed to ResNet34's deep architecture, which allows it to learn complex, hierarchical features from images. In contrast, the paper’s approach relied on simpler models that struggled to capture the intricate visual patterns associated with different breeds.

2. **ResNet50 vs. ResNet34:**
   The expectation that ResNet50 would outperform ResNet34 was grounded in its increased depth and capability to capture even more complex features. However, initial trials indicated that simply using ResNet50 without adjustments did not yield the anticipated improvements. This suggests that while deeper models can potentially enhance accuracy, they also require careful tuning to realize their benefits fully.

3. **Low Accuracy of ResNet50 without Additional Layers:**
   When ResNet50 was used without any additional layers, the model's performance was suboptimal. This can be attributed to its architecture being designed primarily for transfer learning; it may not have been tailored to the specific characteristics of the pet breed classification task. The model's default output layer may not have effectively captured the distribution of classes in this dataset.

4. **Impact of Additional Layers:**
   Adding additional layers significantly improved accuracy. These layers allowed the model to better adapt to the specific nuances of the dataset by learning more detailed representations and reducing overfitting. The extra complexity enabled the model to generalize better, leading to improved performance on validation and test datasets.

5. **Best Model:**
   After thorough experimentation, the best-performing model was identified as EfficientNetB0. This configuration achieved the highest validation and test accuracies, and here's a comparison between ResNet50 and EfficientNetB0 explaiing why it achieved better accuracies.
   The difference in accuracy between ResNet50 and EfficientNetB0 can be attributed to several factors related to their architectures, training strategies, and design philosophies. Here are some key reasons why EfficientNetB0 might outperform ResNet50 in your specific case:

#### 1. **Model Architecture**:
   - **EfficientNetB0**: EfficientNet models are designed using a compound scaling method that balances network depth, width, and resolution. This allows EfficientNet to achieve better performance with fewer parameters compared to traditional architectures.
   - **ResNet50**: ResNet50 is a deeper network with 50 layers, primarily focusing on depth to improve performance. While it is effective, it may not be as efficient in terms of parameter usage and computational cost as EfficientNet.

#### 2. **Parameter Efficiency**:
   - **EfficientNetB0**: EfficientNetB0 is known for its parameter efficiency, achieving high accuracy with fewer parameters and FLOPs (floating-point operations per second). This efficiency can lead to better generalization and performance on various tasks.
   - **ResNet50**: ResNet50 has more parameters and may require more computational resources, which can sometimes lead to overfitting if not managed properly.

#### 3. **Pretraining and Transfer Learning**:
   - **EfficientNetB0**: EfficientNet models are pretrained on large datasets like ImageNet, and their efficient architecture allows them to transfer learned features effectively to new tasks.
   - **ResNet50**: While also pretrained on large datasets, ResNet50 may not transfer features as efficiently as EfficientNet due to its larger size and different architectural design.


### Conclusion:
EfficientNetB0's superior performance in this case can be attributed to its efficient architecture, balanced scaling strategy. While ResNet50 is a powerful model, EfficientNetB0's design allows it to achieve higher accuracy with fewer parameters and computational resources.


7. **Future Experiments:**
   To further increase accuracy, future experiments could explore the following avenues:
   - **Data Augmentation:** Applying more aggressive data augmentation strategies to increase the diversity of training samples and improve model robustness.
   - **Experiment with Different Models**: experimenting with different models and architectures to find the best fit for the dataset.
   - **Ensemble Methods:** Combining predictions from multiple models to enhance overall accuracy and reduce variance in predictions.
   - **Experiment with Other Architectures:** Exploring more recent models could yield better results.
   - **Fine-tuning Pretrained Models:** Instead of training only the final layers, fine-tuning the entire model could help capture more nuanced features from the images.
   - **Hyperparameter Tuning**: Perform thorough hyperparameter tuning for each model to ensure optimal performance.
   - **Data Augmentation and Regularization**: Use advanced data augmentation and regularization techniques to improve model generalization and robustness.


