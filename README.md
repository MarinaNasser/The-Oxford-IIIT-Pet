# The-Oxford-IIIT-Pet


## Dataset description
The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200 images for each class created by the Visual Geometry Group at Oxford. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation. [[kaggle.com](https://www.robots.ox.ac.uk/~vgg/data/pets/)]

Original dataset was downloaded from [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/).


![image](https://github.com/user-attachments/assets/55957e1c-adb4-4309-a561-63b77d5860fe)
The following annotations are available for every image in the dataset: (a) species and breed name; (b) a tight bounding box (ROI) around the head of the animal; and (c) a pixel level foreground-background segmentation (Trimap).

![image](https://github.com/user-attachments/assets/ae31a3b0-805f-405e-8add-dd1c21b2f43a)

## Models Used
### ResNet-34
The [ResNet-34](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html#torchvision.models.resnet34) architecture pre-trained on the ImageNet-1K dataset. This pre-trained model is used for linear evaluation and fine-tuning on the Oxford IIIT Pet Dataset.
