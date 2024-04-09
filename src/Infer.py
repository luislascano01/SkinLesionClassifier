
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class InferenceEngine:
    def __init__(self, model, device=None):
        """
        Initialize the inference engine with a pre-trained model.
        """
        self.lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'dermatofibroma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((450, 450)),  # Resize the input image to the expected size
            transforms.ToTensor(),  # Convert the image to a torch tensor
            # Add any additional normalization transforms here, if your model requires them.
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess the image before feeding it to the model.
        """
        image = Image.open(image_path).convert('RGB')  # Convert image to RGB
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        return image.to(self.device)

    def predict_batch(self, images):
        """
        Predict the skin conditions from a batch of images.
        :param images: a batch of images already preprocessed and moved to the correct device
        """
        with torch.no_grad():
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy()  # Return as a NumPy array for easy handling

    def predict_single_image(self, image_path):
        """
        Predict the skin condition from a single image.
        """
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

    def predict_probabilities_batch(self, images):
        """
        Predict the skin condition probabilities from a batch of images.
        :param images: a batch of images already preprocessed and moved to the correct device
        """
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    def predict_probabilities(self, image_path):
        """
        Predict the skin condition from an image and return the probabilities.
        """
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to convert to probabilities
            probabilities = probabilities.squeeze()  # Remove the batch dimension
        return probabilities.cpu().numpy()  # Return as a NumPy array for easy handling
