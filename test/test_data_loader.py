import unittest
import pandas as pd
import torch

from src.DataLoader import DataLoader  # This assumes your class is in a file named image_loader.py
import os


class TestImageLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Assuming you have a small dataset for testing purposes
        cls.test_data = pd.DataFrame({


            'image_id': ['ISIC_0027419.jpg', 'ISIC_0032258.jpg'],
            'dx': ['bkl', 'mel'],
            'age': [80, 70],
            'sex': ['male', 'female'],
            'localization': ['scalp', 'back']
        })
        current_directory = os.getcwd()
        current_directory = os.path.dirname(current_directory)
        cls.images_dir = current_directory+"/skin_conditions_dataset/HAM10000_images_part_1"

        # Instantiate the ImageLoader with the test data
        cls.loader = DataLoader(images_dir=cls.images_dir, metadata=cls.test_data, image_column='image_id',
                                 target_column='dx')

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


if __name__ == '__main__':
    unittest.main()
