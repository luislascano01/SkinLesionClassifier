import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os


class DataLoader:
    def __init__(self, images_dir, metadata: pd.DataFrame, image_column: str, target_column: str, augmentation=False):
        """
        A constructor method to create an object to load images for the skin cancer dataset. The metadata allows
        to construct and advanced tensor not only depending on the images but also considering the other columns
        such as age, gender, and lesion location in teh body.
        :param images_dir: The directory where images are located.
        :param metadata: Pandas dataframe containing the names of the images and their attributes.
        """
        current_directory = os.getcwd()
        self.images_dir = current_directory + "/" + images_dir
        self.metadata = metadata
        self.image_column = image_column
        self.target_column = target_column
        self.augmentation = augmentation
        self.transform = transforms.Compose([
            transforms.Resize((450, 450)),
            transforms.ToTensor(),
        ])
        self.encoder = LabelEncoder()

        self.X = None
        self.y = None

    def load_images(self, data_fraction: float):
        """
        This method loads the images from the dataset and automatically performs data augmentation.
        :param data_fraction:
        :return:
        """
        sampled_metadata = self.metadata.sample(frac=data_fraction)

        # Load images and targets
        images = [self.__load_image__(image_name) for image_name in sampled_metadata[self.image_column]]
        self.X = torch.stack(images)
        self.y = torch.tensor(self.encoder.fit_transform(sampled_metadata[self.target_column].values), dtype=torch.long)

    def __load_image__(self, image_name):
        image_path = os.path.join(self.images_dir, image_name) + ".jpg"
        image = Image.open(image_path).convert('RGB')  # Assuming images are in RGB
        if self.augmentation:
            image = self.__perform_augmentation__(image)
        image = self.transform(image)
        return image

    def __perform_augmentation__(self, image: torch.tensor) -> torch.tensor:
        """
        Private method to perform data augmentation of a single image.
        :return:
        """
        augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0),
        ])

        return augmentation_transforms(image)

    def get_X(self) -> torch.tensor:
        """
        Return the X tensor of the dataset.

        :return: torch tensor of X or input
        """
        if self.X is None:
            raise ValueError("The images have not been loaded. Call load_images first.")
        return self.X

    def get_y(self) -> torch.tensor:
        """
        Getter method for the Y tensor of the dataset.
        :return:
        """
        if self.y is None:
            raise ValueError("The labels have not been loaded. Call load_images first.")
        return self.y

    def get_encoder(self) -> LabelEncoder:

        return self.encoder
