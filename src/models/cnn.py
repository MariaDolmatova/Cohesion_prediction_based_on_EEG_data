import os
import torch
import pandas as pd
import numpy as np
import configparser
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import copy
from ..utils.random_seed import set_random_seed
from ..utils.visuals import plot_cross_validation_results
from ..utils.logger import get_logger

# Define a CNN model for 1D sequence classification (EEG Data)
class CNN_120dataset(nn.Module):
    def __init__(self):
        """Initialize a 1D CNN with batch normalization, dropout, and fully connected layers."""
        super(CNN_120dataset, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=40, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)  # Normalize activations
        self.dropout1 = nn.Dropout(0.3)  # Prevent overfitting
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Reduce spatial dimensions

        # Second convolutional block
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 30, 32)  # 30 comes from sequence length reduction by pooling
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)  # Output layer (binary classification)

    def forward(self, x):
        """Forward pass of the CNN model."""
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BatchNorm + ReLU
        x = self.dropout1(x)  # Apply dropout
        x = self.pool1(x)  # Max pooling

        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 + BatchNorm + ReLU
        x = self.dropout2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = self.dropout3(x)
        x = self.fc2(x)  # Output layer (raw logits)
        return x

def train_model_early_stopping(model, train_loader, val_loader,
                               optimizer, criterion, epochs, patience, min_delta):
    """
    Train the CNN model using early stopping.
    
    - Stops training if validation loss does not improve after `patience` epochs.
    - Logs training and validation loss, accuracy, and F1 score.
    """
    logger = get_logger()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    best_val_loss = float('inf')  # Best validation loss
    best_model_weights = copy.deepcopy(model.state_dict())  # Store best model state
    patience_counter = 0  # Tracks consecutive epochs without improvement

    for epoch in range(epochs):
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0.0

        # Training loop
        for X_batch, y_batch in train_loader:
            output = model(X_batch).squeeze()  # Forward pass
            y_batch = y_batch.squeeze().float()

            loss = criterion(output, y_batch)  # Compute loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)  # Accumulate batch loss

            predicted = (torch.sigmoid(output) >= 0.5).long()  # Convert logits to binary predictions
            train_labels.extend(y_batch.tolist())
            train_preds.extend(predicted.tolist())

        # Compute metrics
        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_epoch_acc = accuracy_score(train_labels, train_preds)
        train_epoch_f1 = f1_score(train_labels, train_preds, average='binary')

        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        train_f1s.append(train_epoch_f1)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                output_val = model(X_val).squeeze()
                y_val = y_val.squeeze().float()

                val_loss = criterion(output_val, y_val)
                val_running_loss += val_loss.item() * X_val.size(0)

                predicted_val = (torch.sigmoid(output_val) >= 0.5).long()
                val_preds.extend(predicted_val.tolist())
                val_labels.extend(y_val.tolist())

        # Compute validation metrics
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_labels, val_preds)
        val_epoch_f1 = f1_score(val_labels, val_preds, average='binary')

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        val_f1s.append(val_epoch_f1)

       # Early stopping 
        if val_epoch_loss < best_val_loss - min_delta:
            best_val_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'  >> Early stopping at epoch {epoch+1}!')
                break

    model.load_state_dict(best_model_weights) # Load best model weights
    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s


# Cross-validation function for CNN model
def fold_cross_validation_120dataset(model_class, data_X, data_Y, config_path, n_splits):
    """Performs K-Fold cross-validation on the CNN model using the given dataset."""
    set_random_seed()
    logger = get_logger()
    df_X = pd.read_csv(data_X)
    df_Y = pd.read_csv(data_Y)
    df_X.drop(columns=['Pair'], inplace=True, errors='ignore')
    df_X = df_X.dropna()
    scaler = StandardScaler()
    df_X = pd.DataFrame(scaler.fit_transform(df_X), columns=df_X.columns)
    X_np = df_X.values.astype(np.float32)
    y_np = df_Y.values.astype(np.int64)

    # Reshape for CNN input
    X_np = X_np.reshape(43, 5, 120, 8)
    X_np = X_np.reshape(43, 5 * 8, 120)
    
    X_tensor = torch.tensor(X_np)
    y_tensor = torch.tensor(y_np)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Read config
    config = configparser.ConfigParser()
    config.read(config_path)

    if "CNN" not in config:
        logger.error("Missing [CNN] section in config file.")
        raise ValueError("Missing [CNN] section in config file.") # error if no [CNN]

    batch_size = config.getint('CNN', 'batch_size', fallback=None)
    learning_rate = config.getfloat('CNN', 'learning_rate', fallback=None)
    epochs = config.getint('CNN', 'epochs', fallback=None)
    patience = config.getint('CNN', 'patience', fallback=None)
    min_delta = config.getfloat('CNN', 'min_delta', fallback=None)


    if None in [batch_size, learning_rate, epochs, patience, min_delta]:
        logger.error("Missing required values in the config file.")
        raise ValueError("One or more required parameters are missing in the [CNN] section.") # error if some parameter is missing
    
    num_pos = (y_np == 1).sum()
    num_neg = (y_np == 0).sum()
    pos_weight_value = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    kf = KFold(n_splits=n_splits, shuffle=True)
    fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []
    
    # Train the model
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logger.info(f"Fold {fold + 1}")
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        train_losses, val_losses, train_accs, val_accs, _, val_f1s = train_model_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta
        )
        
        fold_val_losses.append(val_losses[-1])
        fold_val_accs.append(val_accs[-1])
        fold_val_f1s.append(val_f1s[-1])

        # Plot each fold
        plot_cross_validation_results(fold, train_losses, val_losses, train_accs, val_accs)
      
    # Compute metrics
    mean_loss = np.mean(fold_val_losses)
    mean_acc = np.mean(fold_val_accs)
    mean_f1 = np.mean(fold_val_f1s)
    
    logger.info(f'Average loss: {mean_loss:.4f}')
    logger.info(f'Average accuracy: {mean_acc:.4f}')
    logger.info(f'Average F1 score: {mean_f1:.4f}')
    
    return fold_val_losses, fold_val_accs, fold_val_f1s