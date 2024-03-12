import unittest

import numpy as np
import pandas as pd
import torch

from src.DataLoader import DataLoader  # This assumes your class is in a file named image_loader.py
import os


class TestImageLoader(unittest.TestCase):

    loader = DataLoader

    @classmethod
    def setUpClass(cls):
        # Assuming you have a small dataset for testing purposes
        cls.test_data = pd.DataFrame({


            'image_id': ['ISIC_0027419', 'ISIC_0032258'],
            'dx': ['bkl', 'mel'],
            'age': [80, 70],
            'sex': ['male', 'female'],
            'localization': ['scalp', 'back']
        })
        cls.images_dir = "skin_conditions_dataset/HAM10000_images_part_1"

        # Instantiate the ImageLoader with the test data
        cls.loader = DataLoader(images_dir=cls.images_dir, metadata=cls.test_data, image_column='image_id',
                                 target_column='dx')
        cls.loader.load_images(1.0)

    def test_load_images(self):
        # Test image loading functionality
        self.loader.load_images(data_fraction=1.0)
        self.assertIsNotNone(self.loader.X, "X tensor should not be None after loading images.")
        self.assertEqual(self.loader.X.shape[0], len(self.test_data), "X tensor length should match number of images.")

    def test_get_X(self):
        # Ensure get_X returns the correct tensor
        self.loader.load_images(data_fraction=1.0)
        X = self.loader.get_X()
        self.assertEqual(X.shape[0], len(self.test_data), "get_X should return X tensor with correct length.")

    def test_get_y(self):
        # Ensure get_y returns the correct tensor
        y = self.loader.get_y()
        self.assertEqual(y.shape[0], len(self.test_data), "get_y should return y tensor with correct length.")

    def test_augmentation(self):
        # Test that augmentation is being applied when the flag is set
        self.loader.augmentation = True
        original_X = self.loader.X.clone() if self.loader.X is not None else None
        self.loader.load_images(data_fraction=1.0)
        self.loader.augmentation = False
        # This test assumes that the augmentation will change the image.
        self.assertNotEqual(torch.equal(original_X, self.loader.X), True,
                            "Augmented images should not be equal to original images.")

    def test_metadata_association(self):
        # This test assumes that the metadata is correctly associated and checks the shape of the X tensor
        expected_channels = 3  # Change this based on how you modify the tensor in __associate_images_with_metadata__
        self.assertEqual(self.loader.X.shape[1], expected_channels, "Metadata association did not adjust the channel size correctly.")

    def test_invalid_image_path(self):
        # Test handling of invalid image paths
        self.loader.images_dir = "nonexistent_directory"
        with self.assertRaises(Exception):
            self.loader.load_images(data_fraction=1.0)

    def test_partial_data_loading(self):
        # Test loading a fraction of the data
        fraction = 0.5
        self.loader.load_images(data_fraction=fraction)
        self.assertTrue(len(self.loader.X) <= len(self.test_data) * fraction, "Loaded data does not match the specified fraction.")

    def test_normalization(self):
        # Directly test the normalization of age (or any continuous feature)
        ages = np.array([80, 70], dtype=np.float32)
        normalized_ages = (ages - ages.mean()) / ages.std()
        self.assertTrue(np.allclose(self.loader.X[:, -1], torch.tensor(normalized_ages, dtype=torch.float), atol=1e-6), "Age normalization does not match expected values.")

    def test_encoding(self):
        # Test the one-hot encoding of categorical features (sex, localization)
        unique_sex = ['male', 'female']
        unique_localization = ['scalp', 'back']
        # Assuming the encoded features are appended after age normalization in the tensor
        expected_shape = len(unique_sex) + len(unique_localization)  # Adjust based on actual implementation
        self.assertEqual(self.loader.X.shape[1] - 1, expected_shape, "Encoding of categorical features does not match expected number of columns.")




if __name__ == '__main__':
    unittest.main()
