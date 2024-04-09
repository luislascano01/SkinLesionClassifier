import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm.auto import tqdm


class ModelEvaluator:
    def __init__(self, inference_engine, X, y, encoder):
        """
        Initialize the model evaluator.
        """
        self.inference_engine = inference_engine
        self.encoder = encoder
        self.X = X
        self.y = y

    def evaluate(self, batch_size=32):
        """
        Evaluate the model on the dataset using batches.
        """
        self.inference_engine.model.eval()  # Set the model to evaluation mode
        y_pred = []

        # Process the dataset in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(self.X), batch_size), desc="Evaluating batches"):
                batch_X = self.X[i:i + batch_size].to(self.inference_engine.device)  # Move batch of X to the model's device
                batch_predictions = self.inference_engine.predict_batch(batch_X)
                y_pred.extend(batch_predictions)  # Directly extend y_pred with batch_predictions if it's a numpy array
                batch_X = batch_X.cpu()  # Free up GPU memory by moving batch_X back to CPU

        return self.y, np.array(y_pred)

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot the confusion matrix using Plotly.
        """
        cm = confusion_matrix(y_true, y_pred)
        cm = cm[::-1]
        class_names_list = list(class_names)  # Ensure class names are in list format

        # Generate annotations based on the confusion matrix
        annotations = [[str(value) for value in row] for row in cm]
        reversed_class_named_list = class_names[::-1].tolist()

        # Generate the figure using the confusion matrix and annotations
        fig = ff.create_annotated_heatmap(z=cm, x=class_names_list, y=reversed_class_named_list, colorscale='Blues',
                                          annotation_text=annotations)

        # Update the layout to add titles and axis labels
        fig.update_layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted Label'),
            yaxis=dict(title='True Label'),
            xaxis_tickangle=-45,
            width=800,
            height=600
        )

        # Show the figure
        fig.show()

    def plot_class_distribution(self, y_true, class_names):
        """
        Plot the class distribution as a pie chart using Plotly.
        """
        counts = np.bincount(y_true)
        fig = go.Figure(data=[go.Pie(labels=class_names, values=counts, hole=0.3)])
        fig.update_layout(title_text='Class Distribution',
                          width=600,
                          height=700
                          )
        fig.show()



    def get_metrics(self, y_true, y_pred, zero_division=0):
        """
        Calculate and return precision, recall, F1 score, and accuracy.
        :param zero_division: Controls the behavior when there is a zero division situation.
                              If set to "0", it will return 0 for precision and F-score
                              if there are no predicted samples. If set to "1", it will return 1.
        """
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=zero_division)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=zero_division)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=zero_division)
        accuracy = accuracy_score(y_true, y_pred)

        return precision, recall, f1, accuracy

    def print_classification_report(self, y_true, y_pred):
        """
        Print the classification report.
        """
        report = classification_report(y_true, y_pred, target_names=self.encoder.classes_)
        print(report)
