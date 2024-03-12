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
        self.images_dir = images_dir
        self.metadata = metadata
        self.image_column = image_column
        self.target_column = target_column
        self.augmentation = augmentation
        self.transform = transforms.Compose([
            transforms.Resize((450, 450)),
            transforms.ToTensor(),
        ])

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
        encoder = LabelEncoder()
        self.y = torch.tensor(encoder.fit_transform(sampled_metadata[self.target_column].values), dtype=torch.long)

        # Associate metadata if additional data is to be included
        self.__associate_images_with_metadata__(sampled_metadata)

    def __load_image__(self, image_name):
        image_path = os.path.join(self.images_dir, image_name)
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
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=10),
            transforms.AutoAugment()
            # Add more transformations here
        ])

        return augmentation_transforms(image)

    def __associate_images_with_metadata__(self, sampled_metadata):
        """
        Private method to associate images with metadata and create a torch tensor from it.
        :return:
        """
        # Normalize continuous features
        age = sampled_metadata['age'].values.astype(np.float32)
        age = (age - age.mean()) / age.std()  # Normalization

        # Encode categorical features
        sex = sampled_metadata['sex'].values.reshape(-1, 1)
        localization = sampled_metadata['localization'].values.reshape(-1, 1)

        categorical_features = np.concatenate((sex, localization), axis=1)
        encoder = OneHotEncoder(sparse_output=False)
        categorical_features_encoded = encoder.fit_transform(categorical_features)

        # Stack the continuous feature with the encoded categorical features
        additional_features = np.concatenate((age.reshape(-1, 1), categorical_features_encoded), axis=1)
        additional_features_tensor = torch.tensor(additional_features, dtype=torch.float)

        # Now, the additional features tensor should be combined with the images tensor
        # Assuming images tensor has shape (N, C, H, W) where N is the number of images,
        # C is the number of channels, and H, W are the height and width of the images
        # We'll need to expand the additional features tensor to match the images tensor shape
        additional_features_expanded = additional_features_tensor.unsqueeze(2).unsqueeze(3)
        additional_features_expanded = additional_features_expanded.expand(-1, -1, self.X.size(2), self.X.size(3))

        # Concatenate the additional features tensor to the images tensor along the channel dimension
        self.X = torch.cat((self.X, additional_features_expanded), dim=1)

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
        if self.y is None:
            raise ValueError("The labels have not been loaded. Call load_images first.")
        return self.y
        :return:
        """
