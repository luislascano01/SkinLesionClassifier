import numpy as np
import torch
from torchvision import transforms
from tqdm.auto import tqdm


class Augmentation:
    def __init__(self, X: torch.tensor, y: torch.tensor):
        self.X = X
        self.y = y
        self.X_aug = None
        self.y_aug = None

    def perform_augmentation(self, augment_fraction=0.3) -> torch.tensor:
        """
        Private method to perform data augmentation of a single image.
        augment_fraction: Value that represents the fraction increase in size of the dataset through augmentation.
        :return:
        """

        x_size = self.X.shape[0]

        if augment_fraction < 0:
            augment_fraction = 0.3

        sample_indices = np.random.choice(np.arange(x_size), int(augment_fraction * x_size), replace=False)

        X_sample = self.X[sample_indices]
        y_sample = self.y[sample_indices]

        augmentation_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-10, 10), expand=True), # Commented out due to execution error.
            transforms.ColorJitter(brightness=0.1, contrast=0.06, saturation=0.06),
            transforms.Resize((450, 450)),
            transforms.ToTensor()
        ])

        aug_X = []

        for image_tensor in tqdm(X_sample, "Performing Data Augmentation"):
            aug_X.append(augmentation_transforms(image_tensor))

        aug_X = torch.stack(aug_X)

        self.X_aug = aug_X
        self.y_aug = y_sample

        return aug_X, y_sample

    def stack_augmented_data(self):
        if self.X_aug is None or self.y_aug is None:
            self.perform_augmentation()

            # Calculate the total size of the combined dataset
        total_size = self.X.size(0) + self.X_aug.size(0)
        # print(f'Total size: {total_size}')

        # Generate a random permutation of indices for the combined dataset
        shuffling_indices = torch.randperm(total_size)

        X_combined = torch.cat((self.X_aug, self.X), dim=0)
        y_combined = torch.cat((self.y_aug, self.y), dim=0)

        return X_combined[shuffling_indices], y_combined[shuffling_indices]
