import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset


class ModelTrainer:
    def __init__(self, model, X, y, num_classes, batch_size=32, lr=0.0005, weight_decay=2e-4, num_epochs=40,
                 train_fraction=0.9):
        # Ensure data is in the correct format and split
        X = X.float()
        dataset_size = len(X)
        train_size = int(dataset_size * train_fraction)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(TensorDataset(X, y), [train_size, val_size])

        # Create DataLoaders
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        # Define device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = num_classes

        # Define training parameters
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.num_epochs = num_epochs

        # Initialize placeholders for tracking progress
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_and_validate(self, epochs=None):
        if epochs is not None:
            self.num_epochs = epochs
        for epoch in range(self.num_epochs):

            # Training phase
            self.model.train()
            # Initialize metrics
            total_train_loss, total_train_correct, total_train = 0, 0, 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model.forward(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                _, preds = torch.max(y_pred, 1)
                total_train_correct += (preds == y_batch).sum().item()
                total_train += y_batch.size(0)

            avg_train_loss = total_train_loss / len(self.train_loader)
            train_accuracy = total_train_correct / total_train
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)

            # Validation phase
            self.model.eval()
            total_val_loss, total_val_correct, total_val = 0, 0, 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model.forward(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(y_pred, 1)
                    total_val_correct += (predicted == y_batch).sum().item()
                    total_val += y_batch.size(0)

            avg_val_loss = total_val_loss / len(self.val_loader)
            val_accuracy = total_val_correct / total_val
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Optional: Print progress
            print(f'Epoch [{epoch + 1}/{self.num_epochs}]')
            print(f'{"Metric":<15} {"Train":<15} {"Validation":<15}')
            print(f'{"Loss":<15} {avg_train_loss:<15.4f} {avg_val_loss:<15.4f}')
            print(f'{"Accuracy":<15} {train_accuracy:<15.4f} {val_accuracy:<15.4f}')
            print('-' * 45)

    # Getter methods
    def get_train_losses(self):
        return self.train_losses

    def get_val_losses(self):
        return self.val_losses

    def get_train_accuracies(self):
        return self.train_accuracies

    def get_val_accuracies(self):
        return self.val_accuracies

    def save_checkpoint(self, file_path):
        checkpoint = {
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1]
        }
        torch.save(checkpoint, file_path)
