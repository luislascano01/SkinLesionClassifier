import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from threading import Thread, Lock
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer:
    def __init__(self, model, file_directory, num_classes, batch_size=32, lr=0.0005, weight_decay=2e-4, num_epochs=40,
                 train_fraction=0.9, num_workers=2, pin_memory=True):

        self.dataset_lock = Lock()
        self.current_dataset = None
        self.loading_thread = None
        self.stop_thread = False

        self.file_directory = file_directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_fraction = train_fraction
        self.num_classes = num_classes

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

        self.x_filepaths = sorted(glob.glob(os.path.join(file_directory, 'X_train_combined_part*.pt')))
        self.y_filepaths = sorted(glob.glob(os.path.join(file_directory, 'y_train_combined_part*.pt')))
        self.dataset_size = len(self.x_filepaths)

    def train_and_validate(self, epochs=None):
        if epochs is not None:
            self.num_epochs = epochs
        for epoch in range(self.num_epochs):

            # Training phase
            self.model.train()
            # Initialize metrics
            total_train_loss, total_train_correct, total_train = 0, 0, 0
            total_val_loss, total_val_correct, total_val = 0, 0, 0

            self.start_loading_thread(0)
            self.loading_thread.join()

            for file_index in range(self.dataset_size):
                with self.dataset_lock:
                    train_dataset, val_dataset = self.current_dataset
                # Split into train and validation sets
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers, pin_memory=self.pin_memory)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers, pin_memory=self.pin_memory)

                # Start pre-loading the next dataset
                if file_index + 1 < self.dataset_size and not self.stop_thread:
                    self.start_loading_thread(file_index + 1)

                # Training phase
                self.model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    self.optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item()
                    _, preds = torch.max(y_pred, 1)
                    total_train_correct += (preds == y_batch).sum().item()
                    total_train += y_batch.size(0)

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self.model(X_batch)
                        loss = self.criterion(y_pred, y_batch)

                        total_val_loss += loss.item()
                        _, predicted = torch.max(y_pred, 1)
                        total_val_correct += (predicted == y_batch).sum().item()
                        total_val += y_batch.size(0)
                del train_dataset, val_dataset, train_loader, val_loader
                torch.cuda.empty_cache()

                # Calculate average loss and accuracy for the epoch
            avg_train_loss = total_train_loss / total_train
            avg_val_loss = total_val_loss / total_val
            train_accuracy = total_train_correct / total_train
            val_accuracy = total_val_correct / total_val

            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_accuracy)

            # Print formatted output for progress
            print(f'Epoch [{epoch + 1}/{self.num_epochs}]')
            print(f'{"Metric":<15} {"Train":<15} {"Validation":<15}')
            print(f'{"Loss":<15} {avg_train_loss:<15.4f} {avg_val_loss:<15.4f}')
            print(f'{"Accuracy":<15} {train_accuracy:<15.4f} {val_accuracy:<15.4f}')
            print('-' * 45)

            self.stop_thread = True
            if self.loading_thread is not None:
                self.loading_thread.join()

    # Getter methods

    def load_dataset(self, file_index):
        with self.dataset_lock:
            x_filepath = self.x_filepaths[file_index]
            y_filepath = self.y_filepaths[file_index]
            X = torch.load(x_filepath)
            y = torch.load(y_filepath)

            dataset_size = len(X)
            train_size = int(dataset_size * self.train_fraction)
            val_size = dataset_size - train_size

            train_dataset = TensorDataset(X[:train_size], y[:train_size])
            val_dataset = TensorDataset(X[train_size:], y[train_size:])

            self.current_dataset = (train_dataset, val_dataset)

    def start_loading_thread(self, file_index):
        if self.loading_thread is not None:
            self.loading_thread.join()

        self.loading_thread = Thread(target=self.load_dataset, args=(file_index,))
        self.loading_thread.start()

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
