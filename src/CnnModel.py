import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG19_Weights


class SkinConditionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinConditionClassifier, self).__init__()
        # Load the pre-trained VGG19 model to leverage with transfer learning
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

        # Remove the last 4 layers from VGG19
        modules = list(vgg19.features)[:-4]
        self.vgg19_features = nn.Sequential(*modules)

        # Freeze the parameters (weights) of the VGG19 features
        for param in self.vgg19_features.parameters():
            param.requires_grad = False

        # Your custom deeper layers with dropout regularization
        self.custom_layers = nn.Sequential(
            # Add another convolutional block
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),  # Dropout in conv layer

            # Reduce channel size before flattening
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),  # Dropout in conv layer

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5),  # Optional

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 2048),  # Adjusted for additional max pooling
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout in dense layer

            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),

            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.vgg19_features(x)
        x = self.custom_layers(x)
        return x