import math

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import os


class DataLoader:
    def __init__(self, images_dir, metadata: pd.DataFrame, image_column: str, target_column: str):
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
        self.transform = transforms.Compose([
            transforms.Resize((450, 450)),
            transforms.ToTensor(),
        ])
        self.encoder = LabelEncoder()

        self.X = None
        self.y = None

    def load_images(self, load_fraction=1):
        """
        Loads images from the dataset and augments a fraction of them.
        The augmented dataset will be larger by the specified fraction.
        :param load_fraction: Determines what fraction of the dataset to load. Usually used for testing purposes.
        """

        sampled_metadata = self.metadata.sample(frac=load_fraction)

        all_images = []
        all_labels = []

        # First, add all original images with progress bar
        for _, row in tqdm(sampled_metadata.iterrows(), total=sampled_metadata.shape[0], desc="Loading original images"):
            image_name = row[self.image_column]
            label = row[self.target_column]
            image = self.__load_image__(image_name)  # Load and transform the image
            all_images.append(image)
            all_labels.append(label)

        self.X = torch.stack(all_images)
        self.y = torch.tensor(self.encoder.fit_transform(all_labels), dtype=torch.long)

    def __load_image__(self, image_name):
        image_path = os.path.join(self.images_dir, image_name) + ".jpg"
        image = Image.open(image_path).convert('RGB')  # Assuming images are in RGB
        image = self.transform(image)
        return image

    def train_test_split_tensor(self, train_size=0.8):
        """
        Splits data and target tensors into training and test sets.

        :param data_tensor: A tensor containing the features.
        :param target_tensor: A tensor containing the targets.
        :param train_size: The proportion of the dataset to include in the train split.
        :return: A tuple containing split data and target tensors for training and testing.
        """
        # Calculate the number of training samples
        num_samples = self.X.size(0)
        num_train = int(num_samples * train_size)

        # Randomly shuffle indices
        indices = torch.randperm(num_samples)

        # Split indices for training and test sets
        train_indices, test_indices = indices[:num_train], indices[num_train:]

        # Split the data and targets into training and test sets
        data_train, data_test = self.X[train_indices], self.X[test_indices]
        target_train, target_test = self.y[train_indices], self.y[test_indices]

        return data_train, data_test, target_train, target_test



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

    def get_num_classes(self) -> int:
        return len(self.encoder.classes_)
